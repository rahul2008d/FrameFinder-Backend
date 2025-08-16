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

MAGIC = b"FFI1"


def build_faiss_index(embeddings: Dict[int, np.ndarray]) -> Tuple[faiss.Index, List[int]]:
    dim = settings.embedding_dim
    index = faiss.IndexFlatIP(dim)  # IP == cosine on normalized vectors
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

    header = MAGIC + struct.pack(">I", len(ts_bytes))  # big-endian length
    return header + ts_bytes + idx_bytes


def _faiss_deserialize(idx_bytes: bytes) -> faiss.Index:
    """
    Compat: Some faiss wheels accept bytes; others require a numpy uint8 array.
    Try bytes first, then fall back to np.frombuffer(..., uint8).
    """
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

            ts_bytes = blob[8:8+ts_len]
            idx_bytes = blob[8+ts_len:]

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


def s3_key_for_index(video_key: str) -> str:
    """
    uploads/<user>-<ts>/<ts>-<file>.ext  ->  uploads/<user>-<ts>/<ts>-<file>.faiss.bin
    """
    p = PurePosixPath(video_key)
    if not p.name:
        raise ValueError("video_key must include a file name")
    return str(p.with_name(f"{p.stem}.faiss.bin"))


def upload_index_to_s3(video_key: str, blob: bytes) -> str:
    s3 = get_s3_client()
    key = s3_key_for_index(video_key)
    bucket = settings.s3_bucket_videos
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=blob,
        ContentType="application/octet-stream",
    )
    logger.info(f"✅ Uploaded index to s3://{bucket}/{key}")
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
    try:
        index, timestamps = deserialize_index(blob)
        logger.info(f"✅ Deserialized index: ntotal={index.ntotal}, ts_len={len(timestamps)}")
        return index, timestamps
    except Exception as e:
        head = blob[:16]
        logger.error(f"❌ Deserialization failed: {e} | first16={head!r} | len={len(blob)}")
        raise
