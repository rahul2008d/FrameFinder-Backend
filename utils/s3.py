import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from settings import settings
from loguru import logger

def get_s3_client():
    cfg = Config(
        region_name=settings.aws_region or os.getenv("AWS_DEFAULT_REGION") or "eu-west-1",
        signature_version="s3v4",
        retries={"max_attempts": 5, "mode": "standard"},
        s3={"addressing_style": "virtual"},
    )

    # 1) If explicit keys are provided (best for local/dev via .env), use them.
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        session = boto3.Session(
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        method = "explicit-keys"

    # 2) Else if a profile is configured, pin to that profile from ~/.aws/credentials.
    elif getattr(settings, "aws_profile", None):
        session = boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
        method = f"profile:{settings.aws_profile}"

    # 3) Else fall back to the default chain (good for prod on AWS with instance/task role).
    else:
        session = boto3.Session(region_name=settings.aws_region)
        method = "default-chain"

    s3 = session.client("s3", config=cfg)

    # Lightweight diagnostics so you immediately see if you’re on temp tokens.
    creds = session.get_credentials()
    has_token = bool(getattr(creds, "token", None)) if creds else False
    used_method = getattr(creds, "method", method if creds else method)
    logger.info(f"AWS auth method={used_method} has_session_token={has_token} region={settings.aws_region}")

    return s3

def create_presigned_post(bucket: str, key: str, *, content_type: str | None = None,
                          max_mb: int = 2048, expires_s: int = 900):
    s3 = get_s3_client()
    conditions = [
        {"key": key},  # lock to exactly this key (or use ["starts-with","$key", prefix] if you need a prefix)
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
        ExpiresIn=expires_s,   # 15 minutes is typical
    )

    f = resp["fields"]
    logger.info(
    "presign url={} has_token={} alg={} date={} key={}",
    resp["url"], "x-amz-security-token" in f, f.get("x-amz-algorithm"),
    f.get("x-amz-date"), f.get("key"),
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
