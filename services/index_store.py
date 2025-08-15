from __future__ import annotations
import io
import faiss
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger
from settings import settings
from utils.s3 import get_s3_client

def build_faiss_index(embeddings: Dict[int, np.ndarray]) -> Tuple[faiss.Index, List[int]]:
    dim = settings.embedding_dim
    # cosine similarity with normalized vectors -> use Inner Product
    index = faiss.IndexFlatIP(dim)
    timestamps = list(sorted(embeddings.keys()))
    vectors = np.stack([embeddings[t] for t in timestamps]).astype("float32")
    index.add(vectors)
    return index, timestamps

def search_faiss(index: faiss.Index, text_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    return index.search(text_vec.astype("float32"), k)

def serialize_index(index: faiss.Index, timestamps: List[int]) -> bytes:
    idx_bytes = faiss.serialize_index(index)
    buf = io.BytesIO()
    # simple container: [faiss][\n][timestamps npy]
    np.save(buf, np.array(timestamps, dtype=np.int32))
    return idx_bytes + b"\n---\n" + buf.getvalue()

def deserialize_index(blob: bytes) -> Tuple[faiss.Index, List[int]]:
    raw_idx, raw_ts = blob.split(b"\n---\n", 1)
    index = faiss.deserialize_index(raw_idx)
    ts = np.load(io.BytesIO(raw_ts)).astype(int).tolist()
    return index, ts

def s3_key_for_index(video_key: str) -> str:
    base = video_key.rsplit("/", 1)[-1]
    return f"{settings.s3_index_prefix}{base}.faiss.bin"

def upload_index_to_s3(video_key: str, blob: bytes) -> str:
    s3 = get_s3_client()
    key = s3_key_for_index(video_key)
    s3.put_object(Bucket=settings.s3_bucket_indexes, Key=key, Body=blob, ContentType="application/octet-stream")
    return key

def download_index_from_s3(video_key: str) -> Tuple[faiss.Index, List[int]]:
    s3 = get_s3_client()
    key = s3_key_for_index(video_key)
    obj = s3.get_object(Bucket=settings.s3_bucket_indexes, Key=key)
    blob = obj["Body"].read()
    return deserialize_index(blob)
