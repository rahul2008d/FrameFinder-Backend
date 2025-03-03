from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from exceptions import VideoNotFoundError, DatabaseConnectionError

def setup_exception_handlers(app: FastAPI):
    @app.exception_handler(VideoNotFoundError)
    async def video_not_found_handler(request: Request, exc: VideoNotFoundError):
        return JSONResponse(status_code=404, content={"error": "Video Not Found", "message": exc.message})

    @app.exception_handler(DatabaseConnectionError)
    async def database_error_handler(request: Request, exc: DatabaseConnectionError):
        return JSONResponse(status_code=500, content={"error": "Database Error", "message": exc.message})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"error": "Internal Server Error", "message": "An unexpected error occurred"})
