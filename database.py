import asyncpg
from fastapi import FastAPI
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_URL")

async def connect_db():
    return await asyncpg.create_pool(DATABASE_URL)

async def close_db(pool):
    await pool.close()
