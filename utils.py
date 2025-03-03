from typing import List
import requests
import tempfile
import cv2
import uuid
import numpy as np
import torch
from transformers import ViTModel, ViTFeatureExtractor
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/dino-vitb8")
vit_model = ViTModel.from_pretrained("facebook/dino-vitb8").to(device)

def extract_frames(video_path: str, chunk_duration: int = 5, fps: int = 1) -> List[dict]:
    """
    Splits video into 5-second chunks and extracts representative frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames // fps_video

    segments = []
    for start in range(0, duration, chunk_duration):
        end = min(start + chunk_duration, duration)
        frame_idx = (start + end) // 2 * fps_video  # Middle frame of the chunk

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            segments.append({"start_time": start, "end_time": end, "frame": frame_rgb})

    cap.release()
    return segments

def generate_vit_embedding(frame: np.ndarray) -> list:
    """
    Uses ViT to generate a high-quality embedding for the given frame.
    """
    inputs = feature_extractor(images=[frame], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vit_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
    return embedding    

def download_video(video_url: str) -> str:
    """
    Downloads the video from the provided Supabase URL and saves it locally.
    """
    response = requests.get(video_url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download video: {response.status_code}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        return temp_file.name