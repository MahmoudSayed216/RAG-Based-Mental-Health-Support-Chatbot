from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from config import limiter

from deployment.routes import base_router, generation_router, health_router, feedback_router
from rag.generator import Generator
from redis import Redis
import os
from dotenv import load_dotenv
import uvicorn
import time

# ── NEW: Import logger ──
from logger import setup_logger, get_logger

load_dotenv()

# ── NEW: Initialize the root logger ONCE ──
setup_logger(
    name="app",
    log_dir="logs",
    console_level=os.getenv("LOG_CONSOLE_LEVEL", "INFO"),
    file_level=os.getenv("LOG_FILE_LEVEL", "DEBUG"),
)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup – initializing Generator and Redis")

    try:
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
        logger.info("Generator initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize Generator: {e}", exc_info=True)
        raise

    try:
        app.redis_client = Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            db=int(os.getenv("REDIS_DB")),
            decode_responses=bool(os.getenv("DECODE_RESPONSE")),
        )
        logger.info("Redis client initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize Redis: {e}", exc_info=True)
        raise

    yield

    logger.info("Application shutdown – cleaning up resources")
    app.generator = None
    app.redis_client = None


# Create the app instance
app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── NEW: Request/Response Logging Middleware ──
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    logger.info(f"→ {request.method} {request.url.path}")

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"← {request.method} {request.url.path} "
        f"| status={response.status_code} "
        f"| duration={duration_ms:.2f}ms"
    )

    return response


# ── NEW: Global exception logging ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(base_router)
app.include_router(generation_router)
app.include_router(feedback_router)
app.include_router(health_router)


logger.info(f"Starting server on {os.getenv('APP_HOST')}:{os.getenv('APP_PORT')}")
uvicorn.run(app, host=os.getenv("APP_HOST"), port=int(os.getenv("APP_PORT")))