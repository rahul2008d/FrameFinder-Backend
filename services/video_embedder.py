from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Iterable, Tuple
from loguru import logger
from services.clip_encoder import encode_images
from settings import settings

def iter_chunks(video_path: str, *, chunk_s: int, overlap_s: int, fps_sample: int) -> Iterable[Tuple[int, list]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = int(total_frames / frame_rate) if frame_rate else 0

    step_frames = max(int(frame_rate / max(fps_sample, 1)), 1)
    frames_per_chunk = max(int(chunk_s * fps_sample), 1)
    stride_s = max(chunk_s - overlap_s, 1)

    logger.info(f"duration={duration_s}s, fps={frame_rate:.2f}, step_frames={step_frames}, "
                f"chunk={chunk_s}s, overlap={overlap_s}s, sample_fps={fps_sample}")

    t = 0
    while t + chunk_s <= duration_s:
        # seek by time (ms)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        frames = []
        grabbed = 0
        while grabbed < frames_per_chunk:
            ret, frame = cap.read()
            if not ret:
                break
            # downsample by stepping frames to approximate fps_sample
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + (step_frames - 1))
            frames.append(frame[:, :, ::-1])  # BGR->RGB
            grabbed += 1

        if frames:
            yield t, frames
        t += stride_s

    cap.release()

def extract_and_embed(video_path: str) -> Dict[int, np.ndarray]:
    """
    Stream frames, compute normalized embeddings, average per chunk.
    Returns {start_sec: embedding(512,)} with float32.
    """
    chunk_embeddings: Dict[int, np.ndarray] = {}
    for start_s, frames in iter_chunks(
        video_path,
        chunk_s=settings.chunk_size_s,
        overlap_s=settings.overlap_s,
        fps_sample=settings.sample_fps,
    ):
        feats = encode_images(frames)  # (N, 512) normalized
        mean = feats.mean(dim=0).cpu().numpy().astype("float32")
        # mean of normalized vectors is not unit norm; renormalize
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        chunk_embeddings[start_s] = mean
    return chunk_embeddings
