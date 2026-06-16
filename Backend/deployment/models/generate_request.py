from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    session_id: str | None = None
    query: str = Field(default="", alias="message")

    class Config:
        populate_by_name = True