from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes import generation, health, base
from rag.generator import Generator
from redis import Redis
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv('.env')

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.generator = Generator(summarize_retrievals = os.getenv('SUMMARIZE_RETRIEVALS'))
    app.redis_client = Redis(host=os.getenv('REDIS_HOST'), 
                                        port=int(os.getenv('REDIS_PORT')), 
                                        db=int(os.getenv('REDIS_DB')), 
                                        decode_responses=bool(os.getenv('DECODE_RESPONSE')))
    yield
    app.generator = None
    app.redis_client = None


app = FastAPI(lifespan=lifespan)


app.include_router(base.base_router)
app.include_router(generation.generation_router)
app.include_router(health.health_router)


uvicorn.run(app, host=os.getenv("APP_HOST"), port=int(os.getenv('APP_PORT')))