import redis
import json


class HistoryController:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def save_history(
        self,
        session_id: str,
        history: list[dict],
        query: str,
        response: str,
        max_messages: int = 10,
        ttl_seconds: int = 3600,
    ):
        key = f"chat_history:{session_id}"
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})
        if len(history) > max_messages:
            history = history[-max_messages:]
        self.redis_client.setex(key, ttl_seconds, json.dumps(history))

    def get_history(self, session_id: str):
        data = self.redis_client.get(f"chat_history:{session_id}")

        history = json.loads(data) if data else []

        print("HISTORY: ", history)
        return history
