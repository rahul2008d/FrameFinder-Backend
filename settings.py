from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class Settings(BaseSettings):
    # API
    cors_origins: List[str] = ["http://localhost:5173"]
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    environment: str = "dev"

    # AWS / S3
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "eu-west-1"
    s3_bucket_videos: str = "framefinder-videos-rahul"
    s3_key_prefix: str = "uploads/"

    # CLIP / Search
    clip_model_id: str = "openai/clip-vit-base-patch32" #"openai/clip-vit-large-patch14"
    embedding_dim: int = 512
    top_k: int = 2

    # Rerank (optional)
    use_maxpool_rerank: bool = True
    sidecar_alpha: float = 0.3

    # Video processing defaults
    chunk_size_s: int = 5
    overlap_s: int = 2
    sample_fps: int = 1

    # Storage
    tmp_dir: str = "/tmp/framefinder"

    planner_enabled: bool = True
    planner_max_expansions: int = 6      
    planner_topn_merge: int = 50 
    openai_api_key: str | None = None
    planner_model: str = "gpt-4o-mini"

    # OCR
    ocr_enabled: bool = True
    ocr_lang: str = "eng"       
    tesseract_cmd: Optional[str] = None 
    ocr_weight: float = 0.7
    ocr_min_conf: int = 60
    ocr_fps: int = 1 

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

settings = Settings()
