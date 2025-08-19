from __future__ import annotations
import os
import io
import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from settings import settings
from utils.s3 import create_presigned_post, create_presigned_get, download_to_path, get_s3_client
from services.video_embedder import extract_and_embed_with_sidecar 
from services.index_store import (
    build_faiss_index,
    serialize_index,
    upload_index_to_s3,
    upload_sidecar_to_s3,
)
from services.ocr_sidecar import build_ocr_sidecar_from_video, upload_ocr_to_s3

router = APIRouter(prefix="/video", tags=["Video"])


class UploadRequest(BaseModel):
    file_name: str = Field(..., description="S3 object key, e.g., uploads/user-ts/ts-file.mp4")
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
        bucket = settings.s3_bucket_videos
        key = req.file_name
        if not key.startswith(settings.s3_key_prefix):
            key = f"{settings.s3_key_prefix}{key}"
        presign = create_presigned_post(
            bucket,
            key,
            content_type=req.content_type or "video/mp4",
        )
        return {"url": presign["url"], "fields": presign["fields"], "key": key}
    except Exception as e:
        logger.exception("presign failed")
        raise HTTPException(500, f"Failed to presign: {e}")


def _local_video_path(video_key: str) -> str:
    safe = video_key.replace("/", "_")
    os.makedirs(settings.tmp_dir, exist_ok=True)
    return os.path.join(settings.tmp_dir, safe)


def _process_video_sync(video_key: str) -> dict:
    """
    Background job:
      1) download original
      2) compute mean & max-pooled embeddings per chunk
      3) build FAISS on means; upload index
      4) upload sidecar (timestamps + max-pooled vectors)
    """
    bucket = settings.s3_bucket_videos
    local_path = _local_video_path(video_key)

    try:
        # 1) Download
        logger.info(f"⬇️  Downloading video s3://{bucket}/{video_key} -> {local_path}")
        download_to_path(bucket, video_key, local_path)

        # 2) Embed
        logger.info("🧮 Extracting embeddings (mean + max) ...")
        means, maxp = extract_and_embed_with_sidecar(local_path)  # {t: 512}, {t: 512}

        # Align everything on the same ordered timestamps used for FAISS
        index, timestamps = build_faiss_index(means)  # FAISS on means
        blob = serialize_index(index, timestamps)

        # Prepare a (N, D) tensor for max-pooled in timestamp order
        D = settings.embedding_dim
        aligned_max = np.stack([maxp[t] for t in timestamps]).astype("float32").reshape(-1, D)

        ocr = None
        logger.info(f"🔍 Building OCR sidecar ... {settings.ocr_enabled}")
        if settings.ocr_enabled:
            try:
                ocr = build_ocr_sidecar_from_video(local_path)
            except Exception as e:
                # Log but don’t raise
                logger.warning(f"OCR sidecar generation failed for {video_key}: {e}")

        # 3) Upload artifacts
        index_key = upload_index_to_s3(video_key, blob)

        sidecar_key = upload_sidecar_to_s3(video_key, timestamps, aligned_max)

        ocr_key = None
        if ocr is not None:
            ocr_key = upload_ocr_to_s3(video_key, ocr)

        return {
            "index_key": index_key,
            "sidecar_key": sidecar_key,
            "ocr_key": ocr_key,
            "total_chunks": len(timestamps),
        }
    except Exception:
        logger.exception("Processing failed")
        raise
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass


@router.post("/process", response_model=ProcessResponse)
async def process_video(req: UploadRequest, bg: BackgroundTasks):
    # Normalize key to include uploads/ prefix
    video_key = req.file_name if req.file_name.startswith(settings.s3_key_prefix) else f"{settings.s3_key_prefix}{req.file_name}"

    # Fire-and-forget background processing
    bg.add_task(_process_video_sync, video_key)

    # Give the client a preview URL immediately (may 404 until the object is fully replicated)
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
