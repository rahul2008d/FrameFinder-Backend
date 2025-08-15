from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    # API
    cors_origins: List[str] = ["http://localhost:5173"]
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    environment: str = "dev"  # dev|staging|prod

    # AWS / S3
    aws_region: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket_videos: str = "framefinder-videos-bucket"
    s3_bucket_indexes: str = "framefinder-indexes-bucket"  # could be same as videos
    s3_key_prefix: str = "uploads/"
    s3_index_prefix: str = "indexes/"

    # CLIP / Search
    clip_model_id: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    top_k: int = 5

    # Video processing defaults
    chunk_size_s: int = 5
    overlap_s: int = 2
    sample_fps: int = 1

    # Storage
    tmp_dir: str = "/tmp/framefinder"

    model_config = SettingsConfigDict(
        env_prefix="FRAMEFINDER_",
        env_file=".env",
        case_sensitive=False,
    )

settings = Settings()
