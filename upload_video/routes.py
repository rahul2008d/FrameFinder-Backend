import os
import faiss
import numpy as np
from fastapi import APIRouter 
from fastapi.responses import JSONResponse
from fastapi import HTTPException, status, UploadFile
from supabase import create_client, Client
from .schema import UploadRequest, VideoMetadata
from utils import create_presigned_post, download_video_from_s3, extract_frames, encode_chunks, build_index
from loguru import logger

router = APIRouter()

@router.get("/health-check")
async def health_check():
    return {"status": "ok"}
    

@router.post("/get-signed-url/")
async def get_signed_url(request: UploadRequest):
    try:
        bucket_name = "framefinder-videos-bucket"
        file_path = request.file_name  # Use actual file name

        boto3_response = create_presigned_post(bucket_name, file_path)

        region = "ap-south-1"  
        boto3_response["url"] = f"https://framefinder-videos-bucket.s3.{region}.amazonaws.com"

        logger.info(f"Signed URL: {boto3_response['url']}")

        # return {"signedUrl": response['signed_url'], "path": response['path']}
        return JSONResponse(content={"url": boto3_response["url"], "fields": boto3_response["fields"]})

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file: {str(e)}")


@router.post("/process_video/")
async def process_video(request: UploadRequest):
    # global faiss_index, timestamps

    bucket = "framefinder-videos-bucket"
    key = request.file_name
    video_path = os.path.join(os.getcwd(), request.file_name)  # Full path

    # Step 1: Check if video already exists
    if os.path.exists(video_path):
        logger.info("⚡ Video already exists, skipping download.")
    else:
        logger.info("Downloading video from S3...")
        download_video_from_s3(bucket, key, video_path)
        logger.info("✅ Video downloaded successfully.")

    # Step 2: Extract frames
    logger.info("Extracting frames from video...")
    frames_dict = extract_frames(video_path)
    logger.info(f"✅ Extracted {len(frames_dict)} frames.")

    # Step 3: Encode frames into embeddings
    logger.info("Encoding frames into embeddings...")
    chunk_embeddings = encode_chunks(frames_dict)
    logger.info(f"✅ Encoded {len(chunk_embeddings)} embeddings.")

    # Step 4: Build FAISS index
    logger.info("Building FAISS index...")
    faiss_index, timestamps = build_index(chunk_embeddings)

    index_path = "faiss_index.idx"
    timestamps_path = "timestamps.npy"

    faiss.write_index(faiss_index, index_path)
    np.save(timestamps_path, timestamps)

    logger.info(f"✅ FAISS index built successfully with {len(timestamps)} timestamps.")

    return {"message": "Video processed and indexed successfully", "total_chunks": len(timestamps)}




