from fastapi import APIRouter, Request
from models.generate_request import GenerateRequest
from uuid import uuid4
from controllers.history_controller import HistoryController
from rag.generator import Generator
import os
from dotenv import load_dotenv

# from ..helpers.config import get_settings, Settings
load_dotenv(".env")


generation_router = APIRouter()


@generation_router.post("/generate")  # Fix: Get -> Post
def generate(request: Request, generate_request: GenerateRequest):
    history_controller = HistoryController(
        request.app.redis_client
    )  # it better just take the client as an input param
    generator: Generator = request.app.generator
    history = []
    history_str = ""

    if generate_request.session_id is None:
        generate_request.session_id = str(uuid4())
    else:
        history = history_controller.get_history(generate_request.session_id)

        history_str = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
        )
    # with open("history_str.txt", "w") as f:
    #     f.write(history_str)
    answer = generator.answer(generate_request.query, history_str)  ## AWAIT

    history_controller.save_history(
        generate_request.session_id,
        history,
        generate_request.query,
        answer,
        max_messages=int(os.getenv("MAX_MESSAGES")),
        ttl_seconds=int(os.getenv("TTL_SECONDS")),
    )
    return {"answer": answer, "session_id": generate_request.session_id}
