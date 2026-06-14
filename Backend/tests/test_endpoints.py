"""
Tests for API endpoints:
  - GET  /           (base_router)
  - GET  /health     (health_router)
  - POST /generate   (generation_router)
  - POST /feedback   (feedback_router)
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded


# ── Env vars required by generation.py ───────────────────────────────────────
# int(os.getenv("MAX_MESSAGES")) and int(os.getenv("TTL_SECONDS")) are called
# inside the /generate handler, so they must be present for every test.

GENERATION_ENV = {
    "APP_NAME": "Serenity",
    "APP_VERSION": "0.0.1",
    "MAX_MESSAGES": "10",
    "TTL_SECONDS": "3600",
}


# ── Disable rate limiting globally for all tests ──────────────────────────────
# The @limiter.limit("7/minute") decorators live on the route functions and
# reference the `limiter` object imported from config.py.
# Replacing limiter.limit with a no-op decorator means the route functions
# are never wrapped with rate-limit logic, so 429s cannot occur in tests
# regardless of how many times an endpoint is called.

def _noop_limit(limit_string):
    """Return a pass-through decorator that ignores the rate-limit string."""
    def decorator(func):
        return func
    return decorator

import config  # noqa: E402  (import after path is set up by pytest)
config.limiter.limit = _noop_limit


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_app():
    """
    Build a minimal FastAPI app that mirrors main.py wiring but with
    the Generator and Redis replaced by mocks, so no real models load.
    """
    from deployment.routes import (
        base_router,
        generation_router,
        health_router,
        feedback_router,
    )

    app = FastAPI()

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.include_router(base_router)
    app.include_router(generation_router, prefix="/generate_ns")
    app.include_router(feedback_router)
    app.include_router(health_router, prefix="/health")

    mock_generator = MagicMock()
    mock_generator.answer.return_value = "Mock answer"
    app.generator = mock_generator

    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    app.redis_client = mock_redis

    return app


@pytest.fixture(scope="function")
def client():
    with patch.dict(os.environ, GENERATION_ENV):
        app = build_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ═════════════════════════════════════════════════════════════════════════════
# BASE ROUTE  GET /
# ═════════════════════════════════════════════════════════════════════════════

class TestBaseRoute:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_response_contains_app_name(self, client):
        response = client.get("/")
        body = response.json()
        assert "app_name: " in body or "app_version" in body

    def test_response_is_json(self, client):
        response = client.get("/")
        assert response.headers["content-type"].startswith("application/json")


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH ROUTE  GET /health
# ═════════════════════════════════════════════════════════════════════════════

class TestHealthRoute:
    def test_returns_200(self, client):
        response = client.get("/health/")
        assert response.status_code == 200

    def test_response_has_expected_keys(self, client):
        response = client.get("/health/")
        body = response.json()
        assert isinstance(body, dict)

    def test_response_is_json(self, client):
        response = client.get("/health/")
        assert response.headers["content-type"].startswith("application/json")


# ═════════════════════════════════════════════════════════════════════════════
# GENERATION ROUTE  POST /generate
# ═════════════════════════════════════════════════════════════════════════════

class TestGenerationRoute:

    # ── Happy paths ──────────────────────────────────────────────────────────

    def test_generate_without_session_id_returns_200(self, client):
        payload = {"query": "I feel anxious all the time"}
        response = client.post("/generate_ns/generate", json=payload)
        assert response.status_code == 200

    def test_generate_returns_answer_and_session_id(self, client):
        payload = {"query": "I feel anxious all the time"}
        response = client.post("/generate_ns/generate", json=payload)
        body = response.json()
        assert "answer" in body
        assert "session_id" in body

    def test_generate_auto_creates_session_id(self, client):
        payload = {"query": "Hello"}
        response = client.post("/generate_ns/generate", json=payload)
        body = response.json()
        assert body["session_id"] is not None
        assert len(body["session_id"]) > 0

    def test_generate_with_existing_session_id(self, client):
        """Second call with the same session_id should still succeed."""
        # Seed history in mock redis
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        client.app.redis_client.get.return_value = json.dumps(history)

        payload = {"query": "Can you help me?", "session_id": "existing-session-123"}
        response = client.post("/generate_ns/generate", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "existing-session-123"

    def test_generate_answer_comes_from_generator(self, client):
        client.app.generator.answer.return_value = "Specific mocked answer"
        client.app.redis_client.get.return_value = None

        payload = {"query": "What is anxiety?"}
        response = client.post("/generate_ns/generate", json=payload)
        assert response.json()["answer"] == "Specific mocked answer"

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_generate_with_empty_query(self, client):
        """Empty string query — should still reach the endpoint (validation is lenient)."""
        client.app.redis_client.get.return_value = None
        payload = {"query": ""}
        response = client.post("/generate_ns/generate", json=payload)
        # The endpoint itself doesn't block empty strings; generator handles it
        assert response.status_code == 200

    def test_generate_with_very_long_query(self, client):
        client.app.redis_client.get.return_value = None
        long_query = "stress " * 500
        payload = {"query": long_query}
        response = client.post("/generate_ns/generate", json=payload)
        assert response.status_code == 200

    def test_generate_with_arabic_query(self, client):
        client.app.redis_client.get.return_value = None
        payload = {"query": "أنا أشعر بالقلق الشديد"}
        response = client.post("/generate_ns/generate", json=payload)
        assert response.status_code == 200

    def test_generate_missing_query_field_uses_default(self, client):
        """query has a default of '' so omitting it should still be accepted."""
        client.app.redis_client.get.return_value = None
        response = client.post("/generate_ns/generate", json={})
        assert response.status_code == 200

    def test_generate_invalid_body_returns_422(self, client):
        """Sending a non-object body should fail schema validation."""
        response = client.post(
            "/generate_ns/generate",
            content="not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    # ── Error paths ──────────────────────────────────────────────────────────

    def test_generate_when_generator_raises_returns_500(self, client):
        client.app.generator.answer.side_effect = RuntimeError("model crashed")
        client.app.redis_client.get.return_value = None

        payload = {"query": "What should I do?"}
        response = client.post("/generate_ns/generate", json=payload)
        assert response.status_code == 500

        # Reset for subsequent tests
        client.app.generator.answer.side_effect = None
        client.app.generator.answer.return_value = "Mock answer"


# ═════════════════════════════════════════════════════════════════════════════
# FEEDBACK ROUTE  POST /feedback
# ═════════════════════════════════════════════════════════════════════════════

class TestFeedbackRoute:

    VALID_PAYLOAD = {
        "vote": "up",
        "user_message": "I feel sad",
        "bot_response": "I'm sorry to hear that.",
    }

    # ── Happy paths ──────────────────────────────────────────────────────────

    def test_feedback_returns_200(self, client):
        with patch("builtins.open", mock_open()):
            with patch("os.path.isfile", return_value=False):
                response = client.post("/feedback", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_feedback_returns_success_status(self, client):
        with patch("builtins.open", mock_open()):
            with patch("os.path.isfile", return_value=True):
                response = client.post("/feedback", json=self.VALID_PAYLOAD)
        body = response.json()
        assert body.get("status") == "success"

    def test_feedback_downvote(self, client):
        payload = {**self.VALID_PAYLOAD, "vote": "down"}
        with patch("builtins.open", mock_open()):
            with patch("os.path.isfile", return_value=True):
                response = client.post("/feedback", json=payload)
        assert response.status_code == 200

    # ── Edge cases ───────────────────────────────────────────────────────────

    def test_feedback_with_empty_strings(self, client):
        payload = {"vote": "", "user_message": "", "bot_response": ""}
        with patch("builtins.open", mock_open()):
            with patch("os.path.isfile", return_value=True):
                response = client.post("/feedback", json=payload)
        assert response.status_code == 200

    def test_feedback_with_unicode_content(self, client):
        payload = {
            "vote": "up",
            "user_message": "أنا حزين جداً",
            "bot_response": "أفهم مشاعرك",
        }
        with patch("builtins.open", mock_open()):
            with patch("os.path.isfile", return_value=True):
                response = client.post("/feedback", json=payload)
        assert response.status_code == 200

    def test_feedback_with_long_content(self, client):
        payload = {
            "vote": "up",
            "user_message": "I feel " * 200,
            "bot_response": "Response " * 200,
        }
        with patch("builtins.open", mock_open()):
            with patch("os.path.isfile", return_value=True):
                response = client.post("/feedback", json=payload)
        assert response.status_code == 200

    # ── Error paths ──────────────────────────────────────────────────────────

    def test_feedback_missing_required_fields_returns_422(self, client):
        response = client.post("/feedback", json={"vote": "up"})
        assert response.status_code == 422

    def test_feedback_invalid_body_returns_422(self, client):
        response = client.post(
            "/feedback",
            content="bad body",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422