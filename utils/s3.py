from __future__ import annotations
import os
from botocore.config import Config
from botocore.exceptions import ClientError
import boto3
from typing import Optional, Dict
from settings import settings
from loguru import logger

_config = Config(
    retries={"max_attempts": 5, "mode": "standard"},
    read_timeout=60, connect_timeout=10, region_name=settings.aws_region,
)

def get_s3_client():
    return boto3.client("s3", config=_config)

def create_presigned_post(
    key: str,
    content_type: Optional[str] = None,
    max_size_mb: int = 1024,  # 1GB default
    expires_s: int = 3600,
) -> Dict[str, str]:
    """
    Restrict uploads to a specific key prefix, content-type and size.
    """
    s3 = get_s3_client()
    conditions = [
        {"bucket": settings.s3_bucket_videos},
        ["starts-with", "$key", key],  # exact key or prefix
        ["content-length-range", 1, max_size_mb * 1024 * 1024],
    ]
    fields = {}
    if content_type:
        fields["Content-Type"] = content_type
        conditions.append({"Content-Type": content_type})

    try:
        resp = s3.generate_presigned_post(
            Bucket=settings.s3_bucket_videos,
            Key=key,
            Fields=fields or None,
            Conditions=conditions,
            ExpiresIn=expires_s,
        )
        # Construct region-specific URL
        resp["url"] = f"https://{settings.s3_bucket_videos}.s3.{settings.aws_region}.amazonaws.com"
        return resp
    except ClientError as e:
        logger.error(f"presigned_post error: {e}")
        raise

def create_presigned_get(key: str, expires_s: int = 3600) -> str:
    s3 = get_s3_client()
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.s3_bucket_videos, "Key": key},
        ExpiresIn=expires_s,
    )

def download_to_path(bucket: str, key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = get_s3_client()
    logger.info(f"Downloading s3://{bucket}/{key} -> {local_path}")
    s3.download_file(bucket, key, local_path)
