# routers/search/search.py
from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from services.clip_encoder import encode_text
from services.planner_llm import plan_query
from services.index_store import (
    download_index_from_s3,
    search_faiss,
    s3_key_for_index,
    download_sidecar_from_s3,   # sidecar for max-pooled vectors
)
from services.ocr_sidecar import download_ocr_from_s3, score_ocr
from settings import settings
from utils.s3 import create_presigned_get, get_s3_client


router = APIRouter(prefix="/search", tags=["Search"])


# ---------- Schemas ----------

class SearchRequest(BaseModel):
    video_key: str
    query: str
    k: int | None = None  # top-k (defaults to settings.top_k)


class SearchHit(BaseModel):
    start_sec: int
    score: float  # fused score in [0..1]
    deep_link: str


class SearchResponse(BaseModel):
    hits: List[SearchHit]


# ---------- Helpers ----------

def _encode_many(texts: List[str]) -> List[np.ndarray]:
    """Encode a small list of texts with CLIP, returning unit-norm float32 vectors (D,)."""
    out: List[np.ndarray] = []
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        tv = encode_text(t)  # torch (1, D), already unit-norm
        out.append(tv.cpu().numpy().astype("float32")[0])
    return out


def _norm01_from_cos(x: float) -> float:
    """Cosine/IP in [-1, 1] -> [0, 1]."""
    return 0.5 * (x + 1.0)


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def _fuse_clip_and_ocr(
    clip_scores01: Dict[int, float],
    ocr_scores01_idx: Dict[int, float] | None,
    ocr_weight: float,
) -> Dict[int, float]:
    """
    Fuse normalized CLIP scores ([0..1]) with OCR scores already mapped to **chunk indices**:

      fused = (1 - w) * clip + w * ocr  (when OCR present for that idx)
           or = clip                     (when OCR missing)

    ocr_weight is clipped to [0,1].
    """
    w = max(0.0, min(1.0, float(ocr_weight)))
    if not ocr_scores01_idx:
        return dict(clip_scores01)

    fused: Dict[int, float] = {}
    idxs = set(clip_scores01.keys()) | set(ocr_scores01_idx.keys())
    for i in idxs:
        c = clip_scores01.get(i, 0.0)
        o = ocr_scores01_idx.get(i, 0.0)
        fused[i] = (1.0 - w) * c + w * o if o > 0.0 else c
    return fused


# ---------- Endpoints ----------

@router.get("/status")
def status(video_key: str):
    """
    Lightweight readiness probe: true if the FAISS index object exists.
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

    # 1) Load FAISS index
    try:
        index, timestamps = download_index_from_s3(req.video_key)
    except FileNotFoundError as e:
        logger.warning(str(e))
        raise HTTPException(status_code=202, detail="Index not ready. Retry shortly.")
    except PermissionError as e:
        logger.warning(str(e))
        raise HTTPException(status_code=403, detail="Access denied for index object.")
    except RuntimeError as e:
        logger.error(f"Index decode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected index load error")
        raise HTTPException(status_code=500, detail=f"Index load error: {e}")

    if not timestamps:
        return SearchResponse(hits=[])

    # Build quick mappings for alignment
    ts_to_idx: Dict[int, int] = {int(t): i for i, t in enumerate(timestamps)}

    # 2) LLM planning (best-effort)
    expansions: List[str] = [req.query]
    try:
        plan = plan_query(req.query)
    except Exception as e:
        plan = None
        logger.warning(f"planner_llm failed; using raw query only: {e}")

    if plan:
        if getattr(plan, "normalized", None):
            expansions.append(plan.normalized)
        if getattr(plan, "synonyms", None):
            expansions.extend(plan.synonyms)
        objs = (plan.objects or [])[:3] if getattr(plan, "objects", None) else []
        acts = (plan.actions or [])[:3] if getattr(plan, "actions", None) else []
        combo = " ".join([*objs, *acts]).strip()
        if combo:
            expansions.append(combo)
        if getattr(plan, "person_names", None):
            expansions.extend(plan.person_names[:2])

    # Deduplicate & cap expansions
    max_exp = int(getattr(settings, "planner_max_expansions", 6))
    seen: set[str] = set()
    cleaned: List[str] = []
    for t in expansions:
        s = " ".join((t or "").strip().split())
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        cleaned.append(s)
        if len(cleaned) >= max_exp:
            break

    logger.info(f"🧭 Expansions used: {cleaned}")

    # 3) Encode all expansions
    text_vecs = _encode_many(cleaned) or _encode_many([req.query])
    main_tv = _normalize_vec(np.mean(np.stack(text_vecs, axis=0), axis=0).astype("float32"))

    # 4) Multi-query FAISS search on mean-embeddings; take best score per chunk
    k = int(min(req.k or settings.top_k, len(timestamps)))
    if k <= 0:
        return SearchResponse(hits=[])

    topn = int(min(getattr(settings, "planner_topn_merge", max(10, 3 * k)), len(timestamps)))

    faiss_best_cos: Dict[int, float] = {}  # chunk_idx -> best cosine
    for tv in text_vecs:
        dists, idxs = search_faiss(index, tv.reshape(1, -1), topn)
        if dists.size == 0:
            continue
        for r in range(dists.shape[1]):
            ci = int(idxs[0, r])            # chunk index
            s = float(dists[0, r])          # cosine/IP
            if s > faiss_best_cos.get(ci, -1.0):
                faiss_best_cos[ci] = s

    if not faiss_best_cos:
        return SearchResponse(hits=[])

    # 5) Sidecar re-rank: fuse FAISS(mean) with cosine(main_tv, maxp_chunk)
    alpha = float(getattr(settings, "sidecar_alpha", 0.3))
    try:
        side_ts, side_max = download_sidecar_from_s3(req.video_key)  # (N,), (N, D)
        ts_to_row = {int(t): i for i, t in enumerate(side_ts)}
    except FileNotFoundError:
        side_max, ts_to_row = None, {}

    clip_fused01: Dict[int, float] = {}
    for ci, faiss_cos in faiss_best_cos.items():
        faiss01 = _norm01_from_cos(faiss_cos)
        if side_max is not None:
            t = int(timestamps[ci])
            row = ts_to_row.get(t)
            if row is not None and 0 <= row < side_max.shape[0]:
                maxp_vec = side_max[row]             # (D,)
                maxp_cos = float(maxp_vec @ main_tv) # both unit-norm
                maxp01   = _norm01_from_cos(maxp_cos)
                fused    = (1.0 - alpha) * faiss01 + alpha * maxp01
            else:
                fused = faiss01
        else:
            fused = faiss01
        clip_fused01[ci] = fused

    # 6) OCR fusion — align OCR {second->score} to **chunk indices** first
    ocr_scores01_idx: Dict[int, float] | None = None
    if getattr(settings, "ocr_enabled", False):
        try:
            ocr = download_ocr_from_s3(req.video_key)
            if ocr:
                sec_scores = score_ocr(ocr, req.query)  # e.g. {"18": 0.9, 23: 0.7, ...}
                mapped: Dict[int, float] = {}
                for sec_key, val in (sec_scores or {}).items():
                    try:
                        sec = int(sec_key)
                        idx = ts_to_idx.get(sec)
                        if idx is not None:
                            # if multiple OCR hits map to same idx, keep the max
                            mapped[idx] = max(mapped.get(idx, 0.0), float(val))
                    except Exception:
                        continue
                ocr_scores01_idx = mapped if mapped else None
        except Exception as e:
            logger.warning(f"OCR fusion skipped: {e}")

    final01 = _fuse_clip_and_ocr(
        clip_scores01=clip_fused01,
        ocr_scores01_idx=ocr_scores01_idx,
        ocr_weight=float(getattr(settings, "ocr_weight", 0.7)),
    )

    # 7) Rank by fused score and cap to k
    ranked: List[Tuple[int, float]] = sorted(final01.items(), key=lambda x: x[1], reverse=True)[:k]

    # 8) Deep links
    try:
        base_url = create_presigned_get(req.video_key).split("?")[0]
    except Exception:
        base_url = ""

    hits: List[SearchHit] = []
    for ci, score01 in ranked:
        # ci is a chunk index by construction now
        ts = int(timestamps[int(ci)])
        deep = f"{base_url}#t={ts}" if base_url else ""
        hits.append(SearchHit(start_sec=ts, score=float(score01), deep_link=deep))

    return SearchResponse(hits=hits)
