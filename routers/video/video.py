from __future__ import annotations
from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import os
from settings import settings
from utils.s3 import create_presigned_post, create_presigned_get, download_to_path
from services.video_embedder import extract_and_embed
from services.index_store import build_faiss_index, serialize_index, upload_index_to_s3

router = APIRouter(prefix="/video", tags=["Video"])

class UploadRequest(BaseModel):
    file_name: str = Field(..., description="S3 object key, e.g., uploads/video.mp4")
    content_type: str | None = None

class ProcessResponse(BaseModel):
    video_key: str
    presigned_get: str
    message: str
    total_chunks: int

@router.get("/health-check")
async def health_check():
    return {"status": "ok"}

@router.post("/get-signed-url")
async def get_signed_url(req: UploadRequest):
    try:
        # lock to a specific key (prefix + filename)
        key = req.file_name if req.file_name.startswith(settings.s3_key_prefix) else f"{settings.s3_key_prefix}{req.file_name}"
        post = create_presigned_post(key=key, content_type=req.content_type)
        return JSONResponse(content={"url": post["url"], "fields": post["fields"], "key": key})
    except Exception as e:
        logger.exception("signed-url error")
        raise HTTPException(status_code=500, detail=f"Failed to generate signed URL: {e}")

def _local_video_path(video_key: str) -> str:
    safe = video_key.replace("/", "_")
    os.makedirs(settings.tmp_dir, exist_ok=True)
    return os.path.join(settings.tmp_dir, safe)

def _process_video_sync(video_key: str) -> dict:
    # 1) download to temp
    local_path = _local_video_path(video_key)
    download_to_path(settings.s3_bucket_videos, video_key, local_path)

    # 2) extract+embed streaming
    embeddings = extract_and_embed(local_path)

    # 3) build faiss (cosine via IP on normalized vecs), serialize
    index, timestamps = build_faiss_index(embeddings)
    blob = serialize_index(index, timestamps)

    # 4) upload index artifact
    index_key = upload_index_to_s3(video_key, blob)

    # Cleanup local file
    try:
        os.remove(local_path)
    except Exception:
        pass

    return {"index_key": index_key, "total_chunks": len(timestamps)}

@router.post("/process", response_model=ProcessResponse)
async def process_video(req: UploadRequest, bg: BackgroundTasks):
    # normalize key
    video_key = req.file_name if req.file_name.startswith(settings.s3_key_prefix) else f"{settings.s3_key_prefix}{req.file_name}"

    # kick off background processing so the request returns fast
    bg.add_task(_process_video_sync, video_key)

    # return a GET url for preview and a message; client can poll /search/status if you add a job store
    try:
        presigned = create_presigned_get(video_key)
    except Exception:
        presigned = ""

    return ProcessResponse(
        video_key=video_key,
        presigned_get=presigned,
        message="Video accepted for processing; index will be available shortly.",
        total_chunks=0,
    )
