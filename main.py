from contextlib import asynccontextmanager
from fastapi import FastAPI
from deployment.routes import base_router, generation_router, health_router
from rag.generator import Generator
from redis import Redis
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.generator = Generator(
        top_k=int(os.getenv("TOP_K")),
        top_r=int(os.getenv("TOP_R")),
        verbose=True,
        summarize_retrievals=os.getenv("SUMMARIZE_RETRIEVALS"),
        retriever_device="cpu",
        vector_db_args={
            "qdrant_client_path": "./qdrant_db",
            "collection_name": os.getenv("EMBEDDINGS_COLLECTION_NAME"),
        },
        embedding_model=os.getenv("EMBEDDING_MODEL"),
        reranking_model=os.getenv("RERANKING_MODEL"),
        vector_db_url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        vector_db_api_key=os.getenv("QDRANT_API_KEY"),
    )

    app.redis_client = Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        db=int(os.getenv("REDIS_DB")),
        decode_responses=bool(os.getenv("DECODE_RESPONSE")),
    )
    yield
    app.generator = None
    app.redis_client = None


app = FastAPI(lifespan=lifespan)


app.include_router(base_router)
app.include_router(generation_router)
app.include_router(health_router)


uvicorn.run(app, host=os.getenv("APP_HOST"), port=int(os.getenv("APP_PORT")))
