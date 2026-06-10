import csv
from fastapi import APIRouter
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv(".env")


class FeedbackRequest(BaseModel):
    vote: str
    user_message: str
    bot_response: str
feedback_router = APIRouter()


@feedback_router.post("/feedback")
def collect_feedback(feedback: FeedbackRequest):
    file_exists = os.path.isfile("feedback_logs.csv")
    
    with open("feedback_logs.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Vote", "User Message", "Bot Response"]) # Header
            
        writer.writerow([feedback.vote, feedback.user_message, feedback.bot_response])
        
    return {"status": "success"}