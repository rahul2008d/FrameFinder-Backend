from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
import numpy as np

from services.clip_encoder import encode_text
from services.index_store import (
    download_index_from_s3,
    search_faiss,
    s3_key_for_index,
)
from settings import settings
from utils.s3 import create_presigned_get, get_s3_client

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


@router.get("/status")
def status(video_key: str):
    """
    Lightweight readiness probe: returns ready: true if the index object exists.
    """
    s3 = get_s3_client()
    bucket = settings.s3_bucket_videos
    key = s3_key_for_index(video_key)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return {"ready": True, "bucket": bucket, "key": key}
    except Exception:
        return {"ready": False, "bucket": bucket, "key": key}


@router.post("", response_model=SearchResponse)
@router.post("/", response_model=SearchResponse, include_in_schema=False)
async def search(req: SearchRequest):
    logger.info(f"🔎 SEARCH video_key={req.video_key} query={req.query!r}")

    # 1) Load index (or report proper status)
    try:
        index, timestamps = download_index_from_s3(req.video_key)
    except FileNotFoundError as e:
        logger.warning(str(e))
        # Index not uploaded yet
        raise HTTPException(status_code=202, detail="Index not ready. Retry shortly.")
    except PermissionError as e:
        logger.warning(str(e))
        raise HTTPException(status_code=403, detail="Access denied for index object.")
    except RuntimeError as e:
        # Deserialize or format error
        logger.error(f"Index decode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected index load error")
        raise HTTPException(status_code=500, detail=f"Index load error: {e}")

    # 2) Encode text
    text_vec_t = encode_text(req.query)  # torch tensor (1, D) normalized
    text_vec = text_vec_t.cpu().numpy().astype("float32")

    # Dimension sanity check
    if getattr(index, "d", None) != text_vec.shape[1]:
        raise HTTPException(
            status_code=500,
            detail=f"Dim mismatch: index.d={getattr(index, 'd', None)} vs text_vec={text_vec.shape[1]}",
        )

    # 3) Search
    k = int(min(req.k or settings.top_k, len(timestamps)))
    distances, indices = search_faiss(index, text_vec, k)

    # 4) Build deep links
    try:
        base_url = create_presigned_get(req.video_key).split("?")[0]
    except Exception:
        base_url = ""

    hits: list[SearchHit] = []
    for rank in range(k):
        ts = int(timestamps[int(indices[0, rank])])
        score = float(distances[0, rank])
        deep_link = f"{base_url}#t={ts}" if base_url else ""
        hits.append(SearchHit(start_sec=ts, score=score, deep_link=deep_link))

    return SearchResponse(hits=hits)
