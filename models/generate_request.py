from pydantic import BaseModel

class GenerateRequest(BaseModel):
    session_id: str | None = None
    query: str = ""
