from fastapi import APIRouter, HTTPException, status
from fastapi import APIRouter, HTTPException, Depends
from utils import model, processor
import os
import faiss
import numpy as np
import torch
from loguru import logger


search_router = APIRouter()

@search_router.get("/health")
async def health():
    return {"status": "ok"} 


@search_router.get("/search_clip/")
async def search_clip(query: str):
    index_path = "faiss_index.idx"
    timestamps_path = "timestamps.npy"

    if not (os.path.exists(index_path) and os.path.exists(timestamps_path)):
        return {"error": "No indexed video found. Please process a video first."}

    faiss_index = faiss.read_index(index_path)
    timestamps = np.load(timestamps_path)

    logger.info("Searching for query in indexed video...")
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs).numpy()

    _, indices = faiss_index.search(text_features, k=1)
    best_chunk_start = timestamps[indices[0][0]]

    return {
        "start_time": max(0, best_chunk_start - 2),  # Adding ±2 sec buffer
        "end_time": best_chunk_start + 3
    }

        