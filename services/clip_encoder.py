from __future__ import annotations
import torch
from typing import Tuple
from transformers import CLIPProcessor, CLIPModel
from functools import lru_cache
from settings import settings
from loguru import logger

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@lru_cache(maxsize=1)
def load_clip() -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    logger.info(f"Loading CLIP model: {settings.clip_model_id}")
    device = get_device()
    model = CLIPModel.from_pretrained(settings.clip_model_id)
    model.eval()
    model.to(device)
    processor = CLIPProcessor.from_pretrained(settings.clip_model_id)
    torch.set_grad_enabled(False)
    return model, processor, device

def encode_images(frames) -> torch.Tensor:
    model, processor, device = load_clip()
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    # Normalize for cosine similarity
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats

def encode_text(query: str) -> torch.Tensor:
    model, processor, device = load_clip()
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats  # shape (1, 512)
