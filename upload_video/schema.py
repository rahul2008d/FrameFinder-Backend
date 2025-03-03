from pydantic import BaseModel

class VideoMetadata(BaseModel):
    file_name: str

class UploadRequest(BaseModel):
    file_name: str