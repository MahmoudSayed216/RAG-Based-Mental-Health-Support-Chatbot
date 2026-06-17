"""
Tests for rag/helper_models/llm_caller/llm_caller.py  →  LLMCaller
ChatGroq is mocked so no real API calls are made.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


PATCH_GROQ = "rag.helper_models.llm_caller.llm_caller.ChatGroq"


# ── Factory ───────────────────────────────────────────────────────────────────


def make_caller(
    prompt="{text}", is_intent=False, llm_output="mocked output", verbose=False
):
    env = {
        "GROQ_API_KEY": "grok-api-key",
        "SIDE_MODEL": "meta-llama/llama-4-scout-17b-16e-instruct",
        "INTENT_CLASSIFICATION_MODEL": "llama-3.1-8b-instant",
    }
    with patch.dict(os.environ, env):
        with patch(PATCH_GROQ) as MockGroq:
            llm_instance = MagicMock()
            response = MagicMock()
            response.content = llm_output
            llm_instance.invoke.return_value = response
            MockGroq.return_value = llm_instance

            from rag.helper_models.llm_caller.llm_caller import LLMCaller

            caller = LLMCaller(
                prompt=prompt,
                identifier="TestCaller",
                isIntent=is_intent,
                verbose=verbose,
            )
            caller.llm = llm_instance  # keep ref for assertions
            return caller, llm_instance


# ═════════════════════════════════════════════════════════════════════════════
# call() — happy paths
# ═════════════════════════════════════════════════════════════════════════════


class TestLLMCallerCall:
    def test_returns_string(self):
        caller, _ = make_caller(llm_output="Some response")
        result = caller.call({})
        assert isinstance(result, str)

    def test_returns_stripped_output(self):
        caller, _ = make_caller(llm_output="  hello world  ")
        result = caller.call({})
        assert result == "hello world"

    def test_placeholder_substitution(self):
        caller, llm = make_caller(prompt="Translate {text} to English")
        caller.call({"{text}": "Bonjour"})
        invoked_prompt = llm.invoke.call_args[0][0][0].content
        assert "Bonjour" in invoked_prompt
        assert "{text}" not in invoked_prompt

    def test_multiple_placeholder_substitutions(self):
        caller, llm = make_caller(prompt="{src_lang} to {dst_lang}: {text}")
        caller.call(
            {"{src_lang}": "Arabic", "{dst_lang}": "English", "{text}": "مرحبا"}
        )
        invoked_prompt = llm.invoke.call_args[0][0][0].content
        assert "Arabic" in invoked_prompt
        assert "English" in invoked_prompt
        assert "مرحبا" in invoked_prompt

    def test_llm_invoke_called_once(self):
        caller, llm = make_caller()
        caller.call({})
        llm.invoke.assert_called_once()

    def test_empty_arguments_dict(self):
        caller, _ = make_caller(prompt="Static prompt with no placeholders")
        result = caller.call({})
        assert isinstance(result, str)

    def test_returns_correct_content(self):
        caller, _ = make_caller(llm_output="This is the answer")
        result = caller.call({})
        assert result == "This is the answer"

    def test_intent_caller_uses_intent_model(self):
        """When isIntent=True the caller should initialize with INTENT_CLASSIFICATION_MODEL."""
        env = {
            "GROQ_API_KEY": "test-key",
            "SIDE_MODEL": "meta-llama/llama-4-scout-17b-16e-instruct",
            "INTENT_CLASSIFICATION_MODEL": "llama-3.1-8b-instant",
        }
        with patch.dict(os.environ, env):
            with patch(PATCH_GROQ) as MockGroq:
                llm_instance = MagicMock()
                response = MagicMock()
                response.content = "greeting"
                llm_instance.invoke.return_value = response
                MockGroq.return_value = llm_instance

                from rag.helper_models.llm_caller.llm_caller import LLMCaller

                caller = LLMCaller(prompt="{text}", isIntent=True)
                caller.llm = llm_instance

                # The model name passed to ChatGroq should be the intent model
                init_kwargs = MockGroq.call_args[1]
                assert init_kwargs.get("model") == "llama-3.1-8b-instant"

    def test_non_intent_caller_uses_side_model(self):
        """When isIntent=False the caller should initialize with SIDE_MODEL."""
        env = {
            "GROQ_API_KEY": "test-key",
            "SIDE_MODEL": "meta-llama/llama-4-scout-17b-16e-instruct",
            "INTENT_CLASSIFICATION_MODEL": "llama-3.1-8b-instant",
        }
        with patch.dict(os.environ, env):
            with patch(PATCH_GROQ) as MockGroq:
                llm_instance = MagicMock()
                response = MagicMock()
                response.content = "some response"
                llm_instance.invoke.return_value = response
                MockGroq.return_value = llm_instance

                from rag.helper_models.llm_caller.llm_caller import LLMCaller

                LLMCaller(prompt="{text}", isIntent=False)

                init_kwargs = MockGroq.call_args[1]
                assert (
                    init_kwargs.get("model")
                    == "meta-llama/llama-4-scout-17b-16e-instruct"
                )


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════


class TestLLMCallerEdgeCases:
    def test_empty_prompt(self):
        caller, _ = make_caller(prompt="")
        result = caller.call({})
        assert isinstance(result, str)

    def test_very_long_prompt(self):
        long_prompt = "word " * 2000 + "{text}"
        caller, _ = make_caller(prompt=long_prompt)
        result = caller.call({"{text}": "hello"})
        assert isinstance(result, str)

    def test_unrecognised_placeholder_left_as_is(self):
        """An argument key not in the prompt should not cause an error."""
        caller, llm = make_caller(prompt="Hello world")
        caller.call({"{nonexistent}": "value"})
        # Prompt sent to LLM should still be "Hello world"
        invoked_prompt = llm.invoke.call_args[0][0][0].content
        assert invoked_prompt == "Hello world"

    def test_arabic_content_in_arguments(self):
        caller, _ = make_caller(prompt="{text}")
        result = caller.call({"{text}": "أنا أشعر بالحزن"})
        assert isinstance(result, str)

    def test_empty_llm_output_returns_empty_string(self):
        caller, _ = make_caller(llm_output="")
        result = caller.call({})
        assert result == ""

    def test_whitespace_only_llm_output_returns_empty_string(self):
        caller, _ = make_caller(llm_output="   \n\t  ")
        result = caller.call({})
        assert result == ""


# ═════════════════════════════════════════════════════════════════════════════
# Error paths
# ═════════════════════════════════════════════════════════════════════════════


class TestLLMCallerErrors:
    def test_llm_invoke_failure_raises(self):
        caller, llm = make_caller()
        llm.invoke.side_effect = RuntimeError("API timeout")
        with pytest.raises(RuntimeError):
            caller.call({})
