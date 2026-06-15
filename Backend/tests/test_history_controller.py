"""
Tests for deployment/controllers/history_controller.py
Covers: save_history, get_history — happy paths, edge cases, error paths.
"""

import json
import pytest
from unittest.mock import MagicMock, call
from deployment.controllers.history_controller import HistoryController


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def redis_mock():
    return MagicMock()


@pytest.fixture
def controller(redis_mock):
    return HistoryController(redis_mock)


# ═════════════════════════════════════════════════════════════════════════════
# get_history
# ═════════════════════════════════════════════════════════════════════════════

class TestGetHistory:

    def test_returns_empty_list_when_no_history(self, controller, redis_mock):
        redis_mock.get.return_value = None
        result = controller.get_history("session-1")
        assert result == []

    def test_returns_parsed_history(self, controller, redis_mock):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        redis_mock.get.return_value = json.dumps(history)
        result = controller.get_history("session-1")
        assert result == history

    def test_queries_correct_redis_key(self, controller, redis_mock):
        redis_mock.get.return_value = None
        controller.get_history("abc-123")
        redis_mock.get.assert_called_once_with("chat_history:abc-123")

    def test_returns_list_of_dicts(self, controller, redis_mock):
        history = [{"role": "user", "content": "test"}]
        redis_mock.get.return_value = json.dumps(history)
        result = controller.get_history("s1")
        assert isinstance(result, list)
        assert all(isinstance(m, dict) for m in result)

    def test_handles_empty_json_array(self, controller, redis_mock):
        redis_mock.get.return_value = json.dumps([])
        result = controller.get_history("s1")
        assert result == []

    def test_multiple_turns_preserved(self, controller, redis_mock):
        history = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "msg2"},
            {"role": "assistant", "content": "resp2"},
        ]
        redis_mock.get.return_value = json.dumps(history)
        result = controller.get_history("s1")
        assert len(result) == 4


# ═════════════════════════════════════════════════════════════════════════════
# save_history
# ═════════════════════════════════════════════════════════════════════════════

class TestSaveHistory:

    def test_saves_to_correct_key(self, controller, redis_mock):
        controller.save_history("sess-42", [], "Hi", "Hello", max_messages=10, ttl_seconds=3600)
        key_used = redis_mock.setex.call_args[0][0]
        assert key_used == "chat_history:sess-42"

    def test_appends_user_and_assistant_messages(self, controller, redis_mock):
        controller.save_history("s1", [], "question", "answer", max_messages=10, ttl_seconds=3600)
        saved_data = json.loads(redis_mock.setex.call_args[0][2])
        assert saved_data[-2] == {"role": "user", "content": "question"}
        assert saved_data[-1] == {"role": "assistant", "content": "answer"}

    def test_respects_ttl(self, controller, redis_mock):
        controller.save_history("s1", [], "q", "a", max_messages=10, ttl_seconds=1800)
        ttl_used = redis_mock.setex.call_args[0][1]
        assert ttl_used == 1800

    def test_trims_history_to_max_messages(self, controller, redis_mock):
        old_history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(10)
        ]
        # max_messages=4: after appending 2 new entries → trim to last 4
        controller.save_history("s1", old_history, "new_q", "new_a", max_messages=4, ttl_seconds=3600)
        saved_data = json.loads(redis_mock.setex.call_args[0][2])
        assert len(saved_data) == 4

    def test_does_not_trim_when_under_limit(self, controller, redis_mock):
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        controller.save_history("s1", history, "q", "a", max_messages=10, ttl_seconds=3600)
        saved_data = json.loads(redis_mock.setex.call_args[0][2])
        assert len(saved_data) == 4  # 2 existing + 2 new

    def test_saves_with_empty_prior_history(self, controller, redis_mock):
        controller.save_history("s1", [], "first q", "first a", max_messages=10, ttl_seconds=3600)
        saved_data = json.loads(redis_mock.setex.call_args[0][2])
        assert len(saved_data) == 2

    def test_saves_with_empty_query_and_response(self, controller, redis_mock):
        """Edge case: empty strings should still be stored without error."""
        controller.save_history("s1", [], "", "", max_messages=10, ttl_seconds=3600)
        saved_data = json.loads(redis_mock.setex.call_args[0][2])
        assert saved_data[0]["content"] == ""
        assert saved_data[1]["content"] == ""

    def test_max_messages_of_one_keeps_only_last_entry(self, controller, redis_mock):
        """Extreme trim: only 1 message allowed after save."""
        history = [{"role": "user", "content": "old"}]
        controller.save_history("s1", history, "q", "a", max_messages=1, ttl_seconds=3600)
        saved_data = json.loads(redis_mock.setex.call_args[0][2])
        assert len(saved_data) == 1
        # The last message should be the assistant response
        assert saved_data[-1]["role"] == "assistant"

    def test_stored_value_is_valid_json_string(self, controller, redis_mock):
        controller.save_history("s1", [], "q", "a", max_messages=10, ttl_seconds=3600)
        raw = redis_mock.setex.call_args[0][2]
        parsed = json.loads(raw)  # must not raise
        assert isinstance(parsed, list)
