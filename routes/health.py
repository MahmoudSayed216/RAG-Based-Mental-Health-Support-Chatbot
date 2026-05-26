from fastapi import FastAPI, APIRouter, Depends
from dotenv import load_dotenv
import os
# from ..helpers.config import get_settings, Settings

load_dotenv('.env')


health_router = APIRouter()


@health_router.get("/")
async def health():
    app_name    = os.getenv("APP_NAME")
    app_version = os.getenv("APP_VERSION")

    # return {"message" : f"{app_name} v{app_version}"}
    return {
        "app_name: ": app_name,
        "app_version": app_version
    }
