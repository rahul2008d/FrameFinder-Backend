# services/ocr_sidecar.py
from __future__ import annotations
import io, json, re
from pathlib import PurePosixPath
from typing import Dict, List, Tuple

import cv2
import pytesseract
from loguru import logger

from settings import settings
from utils.s3 import get_s3_client
from services.video_embedder import iter_chunks  

TIME_RE = re.compile(r"\b([0-2]?\d[:.][0-5]\d)\b")  # 2:22, 02:22, 2.22 handled via variants

import shutil
import pytesseract
from loguru import logger
from settings import settings

def _init_tesseract():
    # allow override via settings
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    path = shutil.which("tesseract") or settings.tesseract_cmd
    if not path:
        raise RuntimeError(
            "tesseract is not installed or not on PATH. "
            "Install it (e.g., apt-get install tesseract-ocr or brew install tesseract) "
            "or set FRAMEFINDER_TESSERACT_CMD to the absolute binary path."
        )
    try:
        ver = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract found: {path}, version={ver}")
    except Exception as e:
        raise RuntimeError(f"tesseract is present but not callable: {e}")

# Call this once on module import or from your first OCR entry point:
_init_tesseract()

def s3_key_for_ocr(video_key: str) -> str:
    p = PurePosixPath(video_key)
    return str(p.with_name(f"{p.stem}.ocr.json"))

def upload_ocr_to_s3(video_key: str, sidecar: dict) -> str:
    key = s3_key_for_ocr(video_key)
    s3 = get_s3_client()
    payload = json.dumps(sidecar).encode("utf-8")
    s3.put_object(
        Bucket=settings.s3_bucket_videos,
        Key=key,
        Body=payload,
        ContentType="application/json",
        CacheControl="no-store"
    )
    logger.info(f"✅ Uploaded OCR sidecar to s3://{settings.s3_bucket_videos}/{key}")
    return key

def download_ocr_from_s3(video_key: str) -> dict | None:
    key = s3_key_for_ocr(video_key)
    s3 = get_s3_client()
    try:
        obj = s3.get_object(Bucket=settings.s3_bucket_videos, Key=key)
        raw = obj["Body"].read()
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        logger.warning(f"OCR sidecar missing for {key}: {e}")
        return None

def _norm_token(s: str, *, keep_colon: bool = False) -> str:
    allowed = {":"} if keep_colon else set()
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch in allowed)

def _ocr_tokens_and_lines(frames: List) -> Tuple[List[str], List[str]]:
    """
    Returns (tokens, lines). tokens are normalized (lowercased/alnum),
    lines keep punctuation for phrase/time matches.
    """
    tokens: List[str] = []
    lines: List[str] = []
    for img in frames:  # img is RGB
        bgr = img[:, :, ::-1]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(
            gray,
            lang=settings.ocr_lang,
            output_type=pytesseract.Output.DICT,
        )
        # tokens
        if "text" in data:
            for i, text in enumerate(data["text"]):
                if not text or not text.strip():
                    continue
                conf = int(data.get("conf", ["-1"] * len(data["text"]))[i]) if "conf" in data else -1
                if conf >= settings.ocr_min_conf:
                    t = _norm_token(text)
                    if t:
                        tokens.append(t)
        # lines
        if "text" in data and "line_num" in data:
            by_line: Dict[int, List[str]] = {}
            for i, text in enumerate(data["text"]):
                if not text or not text.strip():
                    continue
                conf = int(data.get("conf", ["-1"] * len(data["text"]))[i]) if "conf" in data else -1
                if conf >= settings.ocr_min_conf:
                    ln = data["line_num"][i]
                    by_line.setdefault(ln, []).append(text)
            for _, words in by_line.items():
                lines.append(" ".join(words).lower())
    return tokens, lines

def build_ocr_sidecar_from_video(video_path: str) -> dict:
    """
    Samples frames (reusing your chunking) and returns:
    {
      "tokens": { "term" : [secs...] },
      "lines":  { sec: ["raw line 1", "raw line 2", ...] }
    }
    """
    inv_tokens: Dict[str, set] = {}
    lines_by_sec: Dict[int, List[str]] = {}

    for sec, frames in iter_chunks(
        video_path,
        chunk_s=settings.chunk_size_s,
        overlap_s=settings.overlap_s,
        fps_sample=max(1, settings.ocr_fps),
    ):
        toks, lines = _ocr_tokens_and_lines(frames)
        lines_by_sec[sec] = lines
        for t in set(toks):  # dedupe within chunk
            inv_tokens.setdefault(t, set()).add(sec)

    return {
        "tokens": {t: sorted(list(secs)) for t, secs in inv_tokens.items()},
        "lines": {int(k): v for k, v in lines_by_sec.items()},
    }

def score_ocr(ocr: dict, raw_query: str) -> Dict[int, float]:
    """
    Very light scoring:
      - exact token match => 1.0
      - substring token match => 0.7
      - line-level substring match => 0.9
      - handles time-like variants 2:22 / 02:22 / 2-22 / 2.22
    Returns {sec -> score in [0..1]}.
    """
    tokens_inv = ocr.get("tokens", {})
    lines_by_sec = ocr.get("lines", {})

    q = raw_query.strip().lower()
    # detect time-like query
    time_like = bool(TIME_RE.search(q))
    # variants for “2:22”
    variants = {q}
    if time_like:
        variants.add(q.replace(".", ":").replace("-", ":"))
        if len(q) == 4 and q[1] == ":":
            variants.add("0" + q)  # 2:22 -> 02:22

    scores: Dict[int, float] = {}
    # token-level
    q_tokens = [_norm_token(v, keep_colon=time_like) for v in variants]
    all_tokens = list(tokens_inv.keys())
    for qt in q_tokens:
        if qt in tokens_inv:
            for sec in tokens_inv[qt]:
                scores[sec] = max(scores.get(sec, 0.0), 1.0)
        else:
            # light fuzzy: substring
            if len(qt) >= 3:
                for tok in all_tokens:
                    if qt in tok:
                        for sec in tokens_inv[tok]:
                            scores[sec] = max(scores.get(sec, 0.0), 0.7)

    # line-level
    for sec, lines in lines_by_sec.items():
        joined = " || ".join(lines).lower()
        for v in variants:
            if v and v in joined:
                scores[sec] = max(scores.get(sec, 0.0), 0.9)

    return scores
