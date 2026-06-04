from fastapi import APIRouter
from dotenv import load_dotenv
import os
# from ..helpers.config import get_settings, Settings

load_dotenv(".env")


base_router = APIRouter()


@base_router.get("/")
async def info():
    app_name = os.getenv("APP_NAME")
    app_version = os.getenv("APP_VERSION")

    # return {"message" : f"{app_name} v{app_version}"}
    return {"app_name: ": app_name, "app_version": app_version}
