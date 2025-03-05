from fastapi import APIRouter, HTTPException, status
from fastapi import APIRouter, HTTPException, Depends
from utils import model, processor
import os
import faiss
import numpy as np
import torch
from loguru import logger

from shared import shared_data

search_router = APIRouter()

@search_router.get("/health")
async def health():
    return {"status": "ok"} 

@search_router.get("/search_clip/")
async def search_clip(query: str):
    if shared_data["faiss_index"] is None or shared_data["timestamps"] is None:
        return {"error": "No indexed video found. Please process a video first."}

    logger.info("Starting processor to run embeddings...")
    inputs = processor(text=query, return_tensors="pt", padding=True)
    logger.info(f"Query '{query}' embedding done...")

    with torch.no_grad():
        logger.info(f"Searching for query '{query}' in indexed video...")
        text_features = model.get_text_features(**inputs).numpy()

    _, indices = shared_data["faiss_index"].search(text_features, k=1)
    best_chunk_start = shared_data["timestamps"][indices[0][0]]

    return {
        "start_time": max(0, best_chunk_start - 2),
        "end_time": best_chunk_start + 3
    }
        