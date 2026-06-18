from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from config import limiter

# ── Auto-instrument FastAPI spans ──
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from deployment.routes import (
    base_router,
    generation_router,
    health_router,
    feedback_router,
)
from rag.generator import Generator
from redis import Redis
import os
from dotenv import load_dotenv
import uvicorn
import time

from logger import setup_logger, get_logger
from telemetry import setup_telemetry, record_http_request, record_request_duration

load_dotenv()

# ── Initialize the root logger ONCE ──
setup_logger(
    name="app",
    log_dir="logs",
    console_level=os.getenv("LOG_CONSOLE_LEVEL", "INFO"),
    file_level=os.getenv("LOG_FILE_LEVEL", "DEBUG"),
)
logger = get_logger(__name__)

# ── Initialize telemetry ONCE, before the app starts ──
setup_telemetry(service_name="mental-health-rag-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Application startup – checking/downloading models, then initializing Generator and Redis"
    )

    # ── Handle Model Downloads Safely at Runtime ──
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        base = "rag/helper_models/model_objs"
        pkl = os.path.join(base, "language_detector.pkl")
        emotion = os.path.join(base, "final_mental_emotion_model")
        os.makedirs(base, exist_ok=True)

        if not os.path.exists(pkl):
            logger.info("Downloading language_detector.pkl from Hugging Face Hub...")
            hf_hub_download(
                repo_id="Abdellmohsennn/language_detector",
                filename="language_detector.pkl",
                local_dir=base,
            )
        else:
            logger.info("language_detector.pkl already exists locally.")

        if not os.path.exists(emotion):
            logger.info(
                "Downloading final_mental_emotion_model snapshot from Hugging Face Hub..."
            )
            snapshot_download(
                repo_id="Abdellmohsennn/final_mental_emotion_model", local_dir=emotion
            )
        else:
            logger.info("final_mental_emotion_model already exists locally.")

    except Exception as e:
        logger.critical(f"Runtime model download failed: {e}", exc_info=True)
        raise

    # ── Initialize Generator ──
    try:
        app.generator = Generator(
            top_k=int(os.getenv("TOP_K", 3)),
            top_r=int(os.getenv("TOP_R", 3)),
            verbose=True,
            summarize_retrievals=os.getenv("SUMMARIZE_RETRIEVALS", "false").lower()
            == "true",
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

    # ── Initialize Redis ──
    try:
        app.redis_client = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6380)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=os.getenv("DECODE_RESPONSE", "false").lower() == "true",
        )
        logger.info("Redis client initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize Redis: {e}", exc_info=True)
        raise

    yield

    logger.info("Application shutdown – cleaning up resources")
    app.generator = None
    app.redis_client = None


# ── Create the app instance ──
app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS Middleware ──
# Note: "allow_origins=['*']" is perfect for open deployment on GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    start_time = time.time()

    route = request.scope.get("route")
    endpoint_path = route.path if route else request.url.path

    skip_paths = ["/health", "/docs", "/redoc", "/openapi.json"]
    if endpoint_path in skip_paths:
        return await call_next(request)

    logger.info(f"→ {request.method} {endpoint_path}")

    response = await call_next(request)

    duration_s = time.time() - start_time

    logger.info(
        f"← {request.method} {endpoint_path} "
        f"| status={response.status_code} "
        f"| duration={duration_s:.2f}s"
    )

    record_http_request(
        method=request.method,
        endpoint=endpoint_path,
        status_code=response.status_code,
    )

    record_request_duration(endpoint=endpoint_path, duration_s=duration_s)

    # Record only real user traffic
    record_http_request(
        method=request.method,
        endpoint=endpoint_path,
        status_code=response.status_code,
    )

    record_request_duration(endpoint=endpoint_path, duration_s=duration_s)

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    record_http_request(
        method=request.method,
        endpoint=request.url.path,
        status_code=500,
    )
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Include routers ──
app.include_router(base_router)
app.include_router(generation_router)
app.include_router(feedback_router)
app.include_router(health_router, prefix="/health")

FastAPIInstrumentor.instrument_app(
    app,
    excluded_urls="/health,/docs,/openapi.json,/redoc",
)

# FIX: Extracted runtime environment evaluation so it falls back gracefully
# and doesn't print missing/None values on startup initialization.
app_host = "0.0.0.0"  # Hardcoded to 0.0.0.0 to fix the "Preparing Space" loop
app_port = int(os.getenv("APP_PORT", 7860))

logger.info(
    f"Server configuration forced. Binding network layer to {app_host}:{app_port}"
)

if __name__ == "__main__":
    uvicorn.run(app, host=app_host, port=app_port)
