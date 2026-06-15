from fastapi import APIRouter, Request
from uuid import uuid4
import os
from dotenv import load_dotenv

from deployment.models import GenerateRequest
from deployment.controllers import HistoryController
from rag.generator import Generator
from config import limiter

# from ..helpers.config import get_settings, Settings
load_dotenv(".env")

from telemetry import record_request_duration,record_session_created, record_messages_per_session
import time
from logger import get_logger
logger = get_logger("GenerationEndpoint:")
generation_router = APIRouter()


@generation_router.post("/generate")  # Fix: Get -> Post
@limiter.limit("7/minute")
def generate(request: Request, generate_request: GenerateRequest):
    start = time.time()
    history_controller = HistoryController(
        request.app.redis_client
    )  # it better just take the client as an input param
    generator: Generator = request.app.generator
    history = []
    general_history_str = ""
    intent_history_str = ""

    if generate_request.session_id is None:
        generate_request.session_id = str(uuid4())
        record_session_created()
    else:
        history = history_controller.get_history(generate_request.session_id)
        intent_history_length = -min(4, len(history))
        general_history_str = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
        )

        intent_history_str = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history[intent_history_length:]
        )
    record_messages_per_session(len(history) + 1)
    # with open("history_str.txt", "w") as f:
    #     f.write(history_str)
    logger.debug(f"QUERY: {generate_request.query}")
    
    answer = generator.answer(
        generate_request.query, general_history_str, intent_history_str
    )  ## AWAIT

    history_controller.save_history(
        generate_request.session_id,
        history,
        generate_request.query,
        answer,
        max_messages=int(os.getenv("MAX_MESSAGES")),
        ttl_seconds=int(os.getenv("TTL_SECONDS")),
    )
    record_request_duration("/generate", time.time() - start)
    return {"answer": answer, "session_id": generate_request.session_id}
