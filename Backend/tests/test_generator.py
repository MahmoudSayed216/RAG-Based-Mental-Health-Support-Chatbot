"""
Tests for rag/generator.py  →  Generator.answer()
All heavy dependencies (LLMCaller, Retriever, EmotionClassifier, LanguageDetector,
Preprocessor) are mocked so no models are loaded during testing.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open


# ── Shared mock targets ───────────────────────────────────────────────────────

PATCH_EMOTION      = "rag.generator.EmotionClassifier"
PATCH_LANG         = "rag.generator.LanguageDetector"
PATCH_LLM          = "rag.generator.LLMCaller"
PATCH_RETRIEVER    = "rag.generator.Retriever"
PATCH_PREPROCESSOR = "rag.generator.Preprocessor"
PATCH_OPEN         = "builtins.open"


# ── Factory ───────────────────────────────────────────────────────────────────

def make_generator(
    language="english",
    intent="asking_mental_health_question",
    emotion="sadness",
    llm_response="Here is your answer.",
    retrieved=None,
):
    """
    Build a Generator with all external deps mocked.
    Returns (generator, mocks_dict).
    """
    if retrieved is None:
        retrieved = [("What is anxiety?", "Anxiety is a feeling of worry.")]

    with (
        patch(PATCH_EMOTION) as MockEmotion,
        patch(PATCH_LANG)    as MockLang,
        patch(PATCH_LLM)     as MockLLM,
        patch(PATCH_RETRIEVER) as MockRetriever,
        patch(PATCH_PREPROCESSOR),
        patch(PATCH_OPEN, mock_open(read_data="Dummy prompt {text} {references} {history} {user_query} {emotion} {intent}")),
    ):
        # Language detector
        lang_instance = MagicMock()
        lang_instance.predict.return_value = {"language": language, "confidence": 0.99, "reliable": True}
        MockLang.return_value = lang_instance

        # Emotion classifier
        emotion_instance = MagicMock()
        emotion_instance.predict_emotion.return_value = (emotion, 0.95)
        MockEmotion.return_value = emotion_instance

        # LLMCaller — three instances: translator, summarizer, responseModel, intent_classifier
        llm_instances = []
        for _ in range(4):
            inst = MagicMock()
            inst.call.return_value = llm_response
            llm_instances.append(inst)
        # intent classifier returns intent string
        llm_instances[3].call.return_value = intent
        MockLLM.side_effect = llm_instances

        # Retriever
        retriever_instance = MagicMock()
        retriever_instance.retrieve.return_value = retrieved
        MockRetriever.return_value = retriever_instance

        from rag.generator import Generator

        gen = Generator(
            top_k=3,
            top_r=10,
            verbose=False,
            summarize_retrievals=False,
            retriever_device="cpu",
            vector_db_args={"collection_name": "test"},
            embedding_model="mock-embed",
            reranking_model="mock-rerank",
            vector_db_url="http://localhost",
            vector_db_api_key="key",
        )

        mocks = {
            "lang": lang_instance,
            "emotion": emotion_instance,
            "llm_instances": llm_instances,
            "retriever": retriever_instance,
        }

        return gen, mocks


# ═════════════════════════════════════════════════════════════════════════════
# answer() — happy paths
# ═════════════════════════════════════════════════════════════════════════════

class TestGeneratorAnswer:

    def test_returns_string(self):
        gen, _ = make_generator()
        result = gen.answer("I feel anxious", "", "")
        assert isinstance(result, str)

    def test_returns_llm_response(self):
        gen, _ = make_generator(llm_response="You are not alone.")
        result = gen.answer("I feel anxious", "", "")
        assert result == "You are not alone."

    def test_rag_used_for_asking_intent(self):
        gen, mocks = make_generator(intent="asking_mental_health_question")
        gen.answer("What is depression?", "", "")
        mocks["retriever"].retrieve.assert_called_once()

    def test_rag_not_used_for_greeting_intent(self):
        gen, mocks = make_generator(intent="greeting")
        gen.answer("Hello!", "", "")
        mocks["retriever"].retrieve.assert_not_called()

    def test_rag_not_used_for_gratitude_intent(self):
        gen, mocks = make_generator(intent="gratitude")
        gen.answer("Thank you!", "", "")
        mocks["retriever"].retrieve.assert_not_called()

    def test_rag_not_used_for_goodbye_intent(self):
        gen, mocks = make_generator(intent="goodbye")
        gen.answer("Bye!", "", "")
        mocks["retriever"].retrieve.assert_not_called()

    def test_rag_not_used_for_out_of_scope_intent(self):
        gen, mocks = make_generator(intent="out_of_scope")
        gen.answer("What is the weather?", "", "")
        mocks["retriever"].retrieve.assert_not_called()

    def test_language_detection_called(self):
        gen, mocks = make_generator()
        gen.answer("some query", "", "")
        mocks["lang"].predict.assert_called_once_with("some query")

    def test_emotion_detection_called(self):
        gen, mocks = make_generator()
        gen.answer("I feel terrible", "", "")
        mocks["emotion"].predict_emotion.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# Translation behaviour
# ═════════════════════════════════════════════════════════════════════════════

class TestGeneratorTranslation:

    def test_translation_triggered_for_non_english(self):
        gen, mocks = make_generator(language="arabic")
        gen.answer("أنا حزين", "", "")
        # translator is llm_instances[0]; it should have been called at least once
        translator = mocks["llm_instances"][0]
        assert translator.call.called

    def test_no_translation_for_english(self):
        gen, mocks = make_generator(language="english")
        gen.answer("I feel sad", "", "")
        translator = mocks["llm_instances"][0]
        # call() on translator should NOT have been invoked
        translator.call.assert_not_called()

    def test_translation_triggered_for_uncertain_language(self):
        """'uncertain' language → should NOT translate (treated as English)."""
        gen, mocks = make_generator(language="uncertain")
        gen.answer("some ambiguous text", "", "")
        translator = mocks["llm_instances"][0]
        translator.call.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestGeneratorEdgeCases:

    def test_empty_query_does_not_raise(self):
        gen, _ = make_generator()
        result = gen.answer("", "", "")
        assert isinstance(result, str)

    def test_very_long_query_does_not_raise(self):
        gen, _ = make_generator()
        result = gen.answer("stress " * 1000, "", "")
        assert isinstance(result, str)

    def test_empty_history_does_not_raise(self):
        gen, _ = make_generator()
        result = gen.answer("query", "", "")
        assert isinstance(result, str)

    def test_no_retrievals_returns_response(self):
        gen, _ = make_generator(retrieved=[])
        result = gen.answer("What is CBT?", "", "")
        assert isinstance(result, str)

    def test_multiple_retrievals_handled(self):
        retrieved = [
            ("Q1", "Answer 1"),
            ("Q2", "Answer 2"),
            ("Q3", "Answer 3"),
        ]
        gen, _ = make_generator(retrieved=retrieved)
        result = gen.answer("Tell me about therapy", "", "")
        assert isinstance(result, str)


# ═════════════════════════════════════════════════════════════════════════════
# Error paths
# ═════════════════════════════════════════════════════════════════════════════

class TestGeneratorErrorPaths:

    def test_retrieval_failure_is_handled_gracefully(self):
        # generator.py catches retrieval errors internally (sets references="")
        # and continues to generate a response rather than propagating the exception.
        gen, mocks = make_generator(intent="asking_mental_health_question")
        mocks["retriever"].retrieve.side_effect = RuntimeError("Qdrant down")
        result = gen.answer("What is depression?", "", "")
        # Should still return a string — the fallback path is used
        assert isinstance(result, str)

    def test_response_model_failure_raises(self):
        gen, mocks = make_generator()
        # responseModel is llm_instances[2]
        mocks["llm_instances"][2].call.side_effect = RuntimeError("LLM timeout")
        with pytest.raises(Exception):
            gen.answer("I feel hopeless", "", "")
