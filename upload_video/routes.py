import os
import uuid
from fastapi import APIRouter
from fastapi import HTTPException, status, UploadFile
from supabase import create_client, Client
from .schema import UploadRequest, VideoMetadata
from utils import extract_frames, generate_vit_embedding, download_video
router = APIRouter()

# Load environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

@router.get("/health-check")
async def health_check():
    return {"status": "ok"}
    

@router.post("/get-signed-url/")
async def get_signed_url(request: UploadRequest):
    try:
        bucket_name = "videos"
        file_path = request.file_name  # Use actual file name
        # 🔹 Upload file to Supabase Storage
        response = supabase.storage.from_(bucket_name).create_signed_upload_url(file_path)

        if not response:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate signed URL")

        return {"signedUrl": response['signed_url'], "path": response['path']}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file: {str(e)}")


@router.post("/update-metadata/")
async def update_video_metadata(video: VideoMetadata):
    try:
        # 🔹 Get the public URL for the file
        public_url = supabase.storage.from_("videos").get_public_url(video.file_name)
        video_id = str(uuid.uuid4())
        # 🔹 Insert metadata into Supabase table
        response = supabase.table("videos").insert({"id": video_id, "file_name": video.file_name, "url": public_url}).execute()

        # ✅ Check for Supabase errors
        if isinstance(response, tuple):
            data, error = response
            if error:
                print("❌ Supabase error:", error)
                return {"status": "error", "message": error.message}        

        # ✅ Step 1: Download Video from Supabase URL
        file_path = download_video(public_url)

        # ✅ Step 2: Extract 5-second segments & frames
        segments = extract_frames(file_path)

        for segment in segments:
            video_id = str(uuid.uuid4())
            start_time, end_time, frame = segment["start_time"], segment["end_time"], segment["frame"]

            # ✅ Step 3: Generate ViT embeddings
            embedding = generate_vit_embedding(frame)

            # ✅ Step 4: Store in Supabase
            response = supabase.table("video_segments").insert({
                "id": str(uuid.uuid4()),
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "embedding": embedding
            }).execute()

        return {"message": "Video metadata updated successfully", "file_name": video.file_name, "url": public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")



