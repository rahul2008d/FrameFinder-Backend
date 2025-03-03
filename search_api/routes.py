from fastapi import APIRouter, HTTPException, status
from fastapi import APIRouter, HTTPException, Depends
from database import connect_db
from utils import generate_video_embedding

search_router = APIRouter()

@search_router.get("/health")
async def health():
    return {"status": "ok"} 


@search_router.get("/search/")
async def search_videos(query: str, db_pool=Depends(connect_db)):
    try:
        # Convert query to embedding
        query_embedding = await generate_video_embedding(query)

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT id, title, video_url 
                FROM videos 
                ORDER BY embedding <=> $1 
                LIMIT 5
                """, 
                query_embedding
            )

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Search failed: {str(e)}")

        