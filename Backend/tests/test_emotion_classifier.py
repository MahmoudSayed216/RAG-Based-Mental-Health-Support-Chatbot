"""
Tests for rag/helper_models/emotion_classifier/DistilBert_inference.py
→ EmotionClassifier
The transformers pipeline and tokenizer/model loading are mocked so no
actual model weights are downloaded or run.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


PATCH_TOKENIZER = "rag.helper_models.emotion_classifier.DistilBert_inference.AutoTokenizer"
PATCH_MODEL     = "rag.helper_models.emotion_classifier.DistilBert_inference.AutoModelForSequenceClassification"
PATCH_PIPELINE  = "rag.helper_models.emotion_classifier.DistilBert_inference.pipeline"


# ── Factory ───────────────────────────────────────────────────────────────────

def make_classifier(pipeline_output=None):
    """
    Returns (EmotionClassifier instance, pipeline_mock).
    pipeline_output: list of one dict, e.g. [{"label": "LABEL_0", "score": 0.95}]
    """
    if pipeline_output is None:
        pipeline_output = [{"label": "LABEL_0", "score": 0.95}]

    env = {"EMOTION_MODEL_PATH": "/fake/model/path"}
    with patch.dict(os.environ, env):
        with patch(PATCH_TOKENIZER), patch(PATCH_MODEL):
            with patch(PATCH_PIPELINE) as MockPipeline:
                pipe_instance = MagicMock(return_value=pipeline_output)
                MockPipeline.return_value = pipe_instance

                from rag.helper_models.emotion_classifier.DistilBert_inference import EmotionClassifier
                clf = EmotionClassifier()
                return clf, pipe_instance


# ═════════════════════════════════════════════════════════════════════════════
# predict_emotion() — happy paths
# ═════════════════════════════════════════════════════════════════════════════

class TestEmotionClassifierPredict:

    def test_returns_tuple(self):
        clf, _ = make_classifier()
        result = clf.predict_emotion("I feel sad")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_emotion_name_and_score(self):
        clf, _ = make_classifier([{"label": "LABEL_0", "score": 0.95}])
        emotion, score = clf.predict_emotion("I feel sad")
        assert emotion == "sadness"
        assert isinstance(score, float)

    @pytest.mark.parametrize("label_id,expected_emotion", [
        (0, "sadness"),
        (1, "joy"),
        (2, "love"),
        (3, "anger"),
        (4, "fear"),
        (5, "surprise"),
    ])
    def test_label_mapping(self, label_id, expected_emotion):
        clf, _ = make_classifier([{"label": f"LABEL_{label_id}", "score": 0.9}])
        emotion, _ = clf.predict_emotion("test text")
        assert emotion == expected_emotion

    def test_score_is_correct(self):
        clf, _ = make_classifier([{"label": "LABEL_1", "score": 0.87}])
        _, score = clf.predict_emotion("I am happy")
        assert score == pytest.approx(0.87)

    def test_pipeline_called_with_input_text(self):
        clf, pipe = make_classifier()
        clf.predict_emotion("my test sentence")
        pipe.assert_called_once_with("my test sentence")

    def test_non_label_format_returned_as_is(self):
        """If the model returns a plain label name instead of LABEL_N, return it directly."""
        clf, _ = make_classifier([{"label": "anger", "score": 0.77}])
        emotion, _ = clf.predict_emotion("I am angry")
        assert emotion == "anger"


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEmotionClassifierEdgeCases:

    def test_empty_string_does_not_raise(self):
        clf, _ = make_classifier()
        result = clf.predict_emotion("")
        assert isinstance(result, tuple)

    def test_very_long_text_does_not_raise(self):
        clf, _ = make_classifier()
        result = clf.predict_emotion("sad " * 512)
        assert isinstance(result, tuple)

    def test_arabic_text_does_not_raise(self):
        clf, _ = make_classifier([{"label": "LABEL_0", "score": 0.8}])
        result = clf.predict_emotion("أنا حزين جداً")
        assert isinstance(result, tuple)

    def test_unknown_label_id_returns_unknown(self):
        clf, _ = make_classifier([{"label": "LABEL_99", "score": 0.5}])
        emotion, _ = clf.predict_emotion("some text")
        assert emotion == "Unknown"

    def test_single_word_input(self):
        clf, _ = make_classifier([{"label": "LABEL_3", "score": 0.6}])
        emotion, _ = clf.predict_emotion("angry")
        assert emotion == "anger"


# ═════════════════════════════════════════════════════════════════════════════
# Error paths
# ═════════════════════════════════════════════════════════════════════════════

class TestEmotionClassifierErrors:

    def test_pipeline_failure_raises(self):
        clf, pipe = make_classifier()
        pipe.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(RuntimeError):
            clf.predict_emotion("I feel bad")
