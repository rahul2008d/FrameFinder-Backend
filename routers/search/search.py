from __future__ import annotations
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel
import numpy as np
from loguru import logger
from services.clip_encoder import encode_text
from services.index_store import download_index_from_s3, search_faiss
from settings import settings
from utils.s3 import create_presigned_get

router = APIRouter(prefix="/search", tags=["Search"])

class SearchRequest(BaseModel):
    video_key: str
    query: str
    k: int | None = None

class SearchHit(BaseModel):
    start_sec: int
    score: float
    deep_link: str

class SearchResponse(BaseModel):
    hits: list[SearchHit]

@router.post("/", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        index, timestamps = download_index_from_s3(req.video_key)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Index not found for {req.video_key}: {e}")

    text_vec_t = encode_text(req.query)  # torch (1,512) normalized
    text_vec = text_vec_t.cpu().numpy().astype("float32")

    k = min(req.k or settings.top_k, len(timestamps))
    distances, indices = search_faiss(index, text_vec, k)

    # distances are cosine similarity in [-1,1]; higher is better
    hits: list[SearchHit] = []
    try:
        base_url = create_presigned_get(req.video_key).split("?")[0]
    except Exception:
        base_url = ""

    for rank in range(k):
        ts = timestamps[int(indices[0, rank])]
        score = float(distances[0, rank])
        deep_link = f"{base_url}#t={ts}" if base_url else ""
        hits.append(SearchHit(start_sec=int(ts), score=score, deep_link=deep_link))

    return SearchResponse(hits=hits)
