from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, Iterable, Tuple, List
from loguru import logger
from services.clip_encoder import encode_images
from settings import settings


def iter_chunks(
    video_path: str,
    *,
    chunk_s: int,
    overlap_s: int,
    fps_sample: int
) -> Iterable[Tuple[int, List[np.ndarray]]]:
    """
    Yields (start_sec, [RGB frames]) for each chunk.
    - Seeks by time (ms) to reduce drift.
    - Downsamples frames to approximate `fps_sample`.
    """
    if chunk_s <= 0:
        raise ValueError("chunk_s must be > 0")
    if overlap_s < 0 or overlap_s >= chunk_s:
        # overlap can be 0..chunk_s-1
        overlap_s = max(0, min(overlap_s, chunk_s - 1))
    if fps_sample <= 0:
        fps_sample = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        frame_rate = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = int(total_frames / frame_rate) if frame_rate > 0 else 0

        # how many raw frames we skip between picks to approximate fps_sample
        step_frames = max(int(round(frame_rate / fps_sample)), 1)
        frames_per_chunk = max(int(round(chunk_s * fps_sample)), 1)
        stride_s = max(chunk_s - overlap_s, 1)

        logger.info(
            "duration=%ss, fps=%.2f, step_frames=%d, chunk=%ss, overlap=%ss, sample_fps=%d",
            duration_s, frame_rate, step_frames, chunk_s, overlap_s, fps_sample
        )

        t = 0
        while t < duration_s:
            if t + 0.5 > duration_s:  # avoid pointless last partial seek
                break

            # Seek by timestamp (ms)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            frames: List[np.ndarray] = []
            grabbed = 0

            while grabbed < frames_per_chunk:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                # Convert BGR -> RGB
                frame_rgb = frame_bgr[:, :, ::-1]
                frames.append(frame_rgb)
                grabbed += 1

                if step_frames > 1:
                    # advance by step_frames-1 to approximate fps_sample
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, cur + (step_frames - 1))

            if frames:
                yield t, frames

            t += stride_s
            if t + chunk_s > duration_s:
                # Stop if the next *full* chunk cannot fit; if you want a trailing partial chunk,
                # remove this check.
                break
    finally:
        cap.release()


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + 1e-12)


def extract_and_embed(video_path: str) -> Dict[int, np.ndarray]:
    """
    Legacy behavior (backward compatible):
    Stream frames, compute normalized CLIP embeddings, **mean** per chunk,
    and return {start_sec: embedding(512, float32)}.

    Use `extract_and_embed_with_sidecar(...)` if you also want a
    max-pooled vector per chunk for reranking.
    """
    means: Dict[int, np.ndarray] = {}

    for start_s, frames in iter_chunks(
        video_path,
        chunk_s=settings.chunk_size_s,
        overlap_s=settings.overlap_s,
        fps_sample=settings.sample_fps,
    ):
        # encode_images returns torch.Tensor (N, D) already L2-normalized per frame
        feats = encode_images(frames)  # (N, 512)
        f = feats.cpu().numpy().astype("float32")

        mean = _normalize(f.mean(axis=0))
        means[start_s] = mean

    return means


def extract_and_embed_with_sidecar(video_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    New: like extract_and_embed but also computes a **max-pooled** vector per chunk,
    which is useful for re-ranking top candidates at query time.

    Returns:
      (means, maxpool)
      - means:   {start_sec: mean(512,)}
      - maxpool: {start_sec: max(512,)}
    """
    means: Dict[int, np.ndarray] = {}
    maxp: Dict[int, np.ndarray] = {}

    for start_s, frames in iter_chunks(
        video_path,
        chunk_s=settings.chunk_size_s,
        overlap_s=settings.overlap_s,
        fps_sample=settings.sample_fps,
    ):
        feats = encode_images(frames)  # (N, 512)
        f = feats.cpu().numpy().astype("float32")

        mean = _normalize(f.mean(axis=0))
        mx   = _normalize(f.max(axis=0))

        means[start_s] = mean
        maxp[start_s] = mx

    return means, maxp
