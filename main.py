from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from settings import settings
from utils.logging import setup_logging
from routers.video.video import router as video_router
from routers.search.search import router as search_router

logger = setup_logging()
app = FastAPI(title="FrameFinder", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(video_router)
app.include_router(search_router)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# dev: uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
