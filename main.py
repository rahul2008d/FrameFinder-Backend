from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from handlers import setup_exception_handlers
from search_api.routes import search_router
from upload_video.routes import router

app = FastAPI()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

# Include API routes
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(router, prefix="/video", tags=["Video"])

# Setup exception handlers
# setup_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

if __name__ == "__main__":    
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)