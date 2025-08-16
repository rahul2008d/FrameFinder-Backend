import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from settings import settings
from loguru import logger

def get_s3_client():
    # If you want to force the region: set FRAMEFINDER_AWS_REGION=eu-west-1
    cfg = Config(
        region_name=settings.aws_region or os.getenv("AWS_DEFAULT_REGION") or "eu-west-1",
        signature_version="s3v4",
        retries={"max_attempts": 5, "mode": "standard"},
        s3={"addressing_style": "virtual"},
    )
    # Don't pass keys explicitly; let boto3 use the default chain (AWS_PROFILE / env / IAM)
    return boto3.client("s3", config=cfg)

def create_presigned_post(bucket: str, key: str, *, content_type: str | None = None,
                          max_mb: int = 2048, expires_s: int = 3600):
    s3 = get_s3_client()
    conditions = [
        {"bucket": bucket},
        ["starts-with", "$key", key],  # lock to exactly this key (or a prefix if you wish)
        ["content-length-range", 1, max_mb * 1024 * 1024],
    ]
    fields = {}
    if content_type:
        fields["Content-Type"] = content_type
        conditions.append({"Content-Type": content_type})

    resp = s3.generate_presigned_post(
        Bucket=bucket,
        Key=key,
        Fields=fields or None,
        Conditions=conditions,
        ExpiresIn=expires_s,
    )

    # Force the **regional** endpoint to avoid redirects / signature mismatches
    region = settings.aws_region or os.getenv("AWS_DEFAULT_REGION") or "eu-west-1"
    resp["url"] = f"https://{bucket}.s3.{region}.amazonaws.com"

    # Helpful one-line debug (no secrets)
    creds = boto3.Session().get_credentials()
    logger.info(
        f"presign: bucket={bucket} key={key} region={region} "
        f"cred_method={getattr(creds,'method',None)} "
        f"fields={list(resp['fields'].keys())}"
    )
    return resp

def create_presigned_get(key: str, *, expires_s: int = 3600) -> str:
    s3 = get_s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket_videos, "Key": key},
        ExpiresIn=expires_s,
    )

def download_to_path(bucket: str, key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3 = get_s3_client()
    s3.download_file(bucket, key, local_path)
