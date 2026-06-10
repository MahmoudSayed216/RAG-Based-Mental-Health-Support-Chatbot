from fastapi import APIRouter, Request
from dotenv import load_dotenv
import os
from config import limiter
# from ..helpers.config import get_settings, Settings

load_dotenv(".env")


health_router = APIRouter()


@health_router.get("/")
@limiter.limit("10/minute")
async def health(request: Request):
    app_name = os.getenv("APP_NAME")
    app_version = os.getenv("APP_VERSION")

    # return {"message" : f"{app_name} v{app_version}"}
    return {"app_name: ": app_name, "app_version": app_version}
