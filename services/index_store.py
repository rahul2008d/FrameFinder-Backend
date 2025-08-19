from __future__ import annotations

import io
import struct
from pathlib import PurePosixPath
from typing import Dict, List, Tuple

import faiss
import numpy as np
from botocore.exceptions import ClientError
from loguru import logger

from settings import settings
from utils.s3 import get_s3_client

MAGIC = b"FFI1"  # header for (timestamps np.save) + raw faiss bytes


# ---------- FAISS core ----------

def build_faiss_index(embeddings: Dict[int, np.ndarray]) -> Tuple[faiss.Index, List[int]]:
    dim = settings.embedding_dim
    index = faiss.IndexFlatIP(dim)  # IP == cosine similarity if vectors are L2-normalized
    timestamps = list(sorted(embeddings.keys()))
    vectors = np.stack([embeddings[t] for t in timestamps]).astype("float32")
    index.add(vectors)
    return index, timestamps


def search_faiss(index: faiss.Index, text_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    return index.search(text_vec.astype("float32"), k)


def serialize_index(index: faiss.Index, timestamps: List[int]) -> bytes:
    idx_blob = faiss.serialize_index(index)
    idx_bytes = idx_blob.tobytes() if isinstance(idx_blob, np.ndarray) else bytes(idx_blob)

    ts_buf = io.BytesIO()
    np.save(ts_buf, np.asarray(timestamps, dtype=np.int32))
    ts_bytes = ts_buf.getvalue()

    header = MAGIC + struct.pack(">I", len(ts_bytes))
    return header + ts_bytes + idx_bytes


def _faiss_deserialize(idx_bytes: bytes) -> faiss.Index:
    try:
        return faiss.deserialize_index(idx_bytes)
    except Exception:
        return faiss.deserialize_index(np.frombuffer(idx_bytes, dtype=np.uint8))


def deserialize_index(blob: bytes) -> Tuple[faiss.Index, List[int]]:
    try:
        if blob.startswith(MAGIC):
            if len(blob) < 8:
                raise ValueError("Blob too small for FFI1 header")
            ts_len = struct.unpack(">I", blob[4:8])[0]
            if 8 + ts_len > len(blob):
                raise ValueError(f"Bad header: ts_len={ts_len} exceeds blob len {len(blob)}")
            ts_bytes = blob[8:8 + ts_len]
            idx_bytes = blob[8 + ts_len:]
            timestamps = np.load(io.BytesIO(ts_bytes), allow_pickle=False).astype(int).tolist()
            index = _faiss_deserialize(idx_bytes)
            return index, timestamps

        # legacy fallback
        if b"\n---\n" not in blob:
            raise ValueError("Unknown index format: missing MAGIC and separator")
        raw_idx, raw_ts = blob.split(b"\n---\n", 1)
        index = _faiss_deserialize(raw_idx)
        timestamps = np.load(io.BytesIO(raw_ts), allow_pickle=False).astype(int).tolist()
        return index, timestamps
    except Exception as e:
        raise RuntimeError(f"deserialize_index error: {e}") from e


# ---------- S3 keys ----------

def s3_key_for_index(video_key: str) -> str:
    """
    uploads/<user>-<ts>/<ts>-<file>.ext -> uploads/<user>-<ts>/<ts>-<file>.faiss.bin
    """
    p = PurePosixPath(video_key)
    if not p.name:
        raise ValueError("video_key must include a file name")
    return str(p.with_name(f"{p.stem}.faiss.bin"))


def s3_key_for_sidecar(video_key: str) -> str:
    """
    Sidecar with timestamps + max-pooled vectors (npz).
    uploads/.../<stem>.sidecar.npz
    """
    p = PurePosixPath(video_key)
    if not p.name:
        raise ValueError("video_key must include a file name")
    return str(p.with_name(f"{p.stem}.sidecar.npz"))


# ---------- S3 upload/download ----------

def upload_index_to_s3(video_key: str, blob: bytes) -> str:
    s3 = get_s3_client()
    key = s3_key_for_index(video_key)
    bucket = settings.s3_bucket_videos
    s3.put_object(Bucket=bucket, Key=key, Body=blob, ContentType="application/octet-stream")
    logger.info(f"✅ Uploaded index to s3://{bucket}/{key}")
    return key


def upload_sidecar_to_s3(video_key: str, timestamps: List[int], max_vectors: np.ndarray) -> str:
    """
    Store tiny sidecar with aligned timestamps + max-pooled chunk vectors.
    - timestamps: list[int] of length N (order used by FAISS index)
    - max_vectors: np.ndarray shape (N, D) float32 (already L2-normalized)
    """
    if not isinstance(max_vectors, np.ndarray):
        max_vectors = np.asarray(max_vectors, dtype="float32")
    if max_vectors.dtype != np.float32:
        max_vectors = max_vectors.astype("float32")
    if max_vectors.ndim != 2:
        raise ValueError("max_vectors must be 2D (N, D)")

    buf = io.BytesIO()
    np.savez_compressed(buf, timestamps=np.asarray(timestamps, dtype=np.int32), maxp=max_vectors)
    payload = buf.getvalue()

    s3 = get_s3_client()
    key = s3_key_for_sidecar(video_key)
    bucket = settings.s3_bucket_videos
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/octet-stream")
    logger.info(f"✅ Uploaded sidecar to s3://{bucket}/{key}")
    return key


def download_index_from_s3(video_key: str) -> Tuple[faiss.Index, List[int]]:
    s3 = get_s3_client()
    key = s3_key_for_index(video_key)
    bucket = settings.s3_bucket_videos
    logger.info(f"⬇️  Downloading index from s3://{bucket}/{key}")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = e.response.get("Error", {}).get("Message")
        logger.warning(f"S3 get_object failed for {bucket}/{key}: {code} — {msg}")
        if code in ("NoSuchKey", "NotFound", "404"):
            raise FileNotFoundError(f"No index at s3://{bucket}/{key}") from e
        if code in ("AccessDenied", "403"):
            raise PermissionError(f"Access denied for s3://{bucket}/{key}") from e
        raise
    blob = obj["Body"].read()
    logger.info(f"📦 Index bytes: {len(blob)}")
    return deserialize_index(blob)


def download_sidecar_from_s3(video_key: str) -> Tuple[List[int], np.ndarray]:
    """
    Returns (timestamps, maxp) or raises FileNotFoundError/PermissionError.
    """
    s3 = get_s3_client()
    key = s3_key_for_sidecar(video_key)
    bucket = settings.s3_bucket_videos
    logger.info(f"⬇️  Downloading sidecar from s3://{bucket}/{key}")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = e.response.get("Error", {}).get("Message")
        logger.warning(f"S3 get_object failed for {bucket}/{key}: {code} — {msg}")
        if code in ("NoSuchKey", "NotFound", "404"):
            raise FileNotFoundError(f"No sidecar at s3://{bucket}/{key}") from e
        if code in ("AccessDenied", "403"):
            raise PermissionError(f"Access denied for s3://{bucket}/{key}") from e
        raise
    blob = obj["Body"].read()
    with np.load(io.BytesIO(blob)) as z:
        ts = z["timestamps"].astype(int).tolist()
        maxp = z["maxp"].astype("float32")
    return ts, maxp
