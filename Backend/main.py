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

from logger import setup_logger, get_logger
from telemetry import setup_telemetry, record_http_request,record_request_duration

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
    logger.info("Application startup – initializing Generator and Redis")

    try:
        app.generator = Generator(
            top_k=int(os.getenv("TOP_K")),
            top_r=int(os.getenv("TOP_R")),
            verbose=True,
            # FIX: os.getenv returns a string; compare explicitly to avoid "False" being truthy
            summarize_retrievals=os.getenv("SUMMARIZE_RETRIEVALS", "false").lower() == "true",
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
            # FIX: os.getenv returns a string; compare explicitly to avoid "False" being truthy
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    # Skip tracking for CORS preflight requests
    if request.method == "OPTIONS":
        return await call_next(request)

    start_time = time.time()

    # Get the route template
    route = request.scope.get("route")
    endpoint_path = route.path if route else request.url.path

    # Skip tracking for internal/health endpoints that inflate metrics
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

    # Record only real user traffic
    record_http_request(
        method=request.method,
        endpoint=endpoint_path,
        status_code=response.status_code,
    )
    
    record_request_duration(endpoint=endpoint_path, duration_s=duration_s)

    return response

# ── Global exception logging ──
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
app.include_router(health_router)

# ── Auto-instrument FastAPI spans ──
# FIX: must be called BEFORE uvicorn.run(), which is blocking
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
FastAPIInstrumentor.instrument_app(
    app,
    excluded_urls="/health,/docs,/openapi.json,/redoc",
)

logger.info(f"Starting server on {os.getenv('APP_HOST')}:{os.getenv('APP_PORT')}")

# FIX: guard with __name__ check so importing this module doesn't start the server
if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("APP_HOST"), port=int(os.getenv("APP_PORT")))