"""
Tests for rag/helper_models/language_detector/language_detector.py
→ LanguageDetector  &  TextPreprocessor
joblib.load is mocked so no real model file is needed.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


PATCH_JOBLIB = "rag.helper_models.language_detector.language_detector.joblib.load"


# ── Factory ───────────────────────────────────────────────────────────────────


def make_detector(threshold=0.70, top_lang="en", top_conf=0.95):
    """
    Returns a LanguageDetector with a mocked sklearn-style pipeline inside.
    top_lang : ISO code returned by the mock
    top_conf : confidence returned by the mock
    """
    env = {"LANGUAGE_DETECTION_MODEL_PATH": "/fake/lang_model.pkl"}
    with patch.dict(os.environ, env):
        with patch(PATCH_JOBLIB) as mock_load:
            mock_model = MagicMock()
            classes = np.array(["ar", "en", "fr", "de"])
            lang_idx = list(classes).index(top_lang) if top_lang in classes else 1
            proba = np.zeros(len(classes))
            proba[lang_idx] = top_conf
            mock_model.predict_proba.return_value = [proba]
            mock_model.classes_ = classes
            mock_load.return_value = mock_model

            from rag.helper_models.language_detector.language_detector import (
                LanguageDetector,
            )

            detector = LanguageDetector(threshold=threshold)
            return detector, mock_model


# ═════════════════════════════════════════════════════════════════════════════
# TextPreprocessor (static method)
# ═════════════════════════════════════════════════════════════════════════════


class TestTextPreprocessor:
    from rag.helper_models.language_detector.language_detector import (
        TextPreprocessor as _TP,
    )

    def test_removes_url(self):
        from rag.helper_models.language_detector.language_detector import (
            TextPreprocessor,
        )

        result = TextPreprocessor.preprocess("visit https://example.com for info")
        assert "https" not in result
        assert "example.com" not in result

    def test_removes_hashtag(self):
        from rag.helper_models.language_detector.language_detector import (
            TextPreprocessor,
        )

        result = TextPreprocessor.preprocess("feeling #sad today")
        assert "#sad" not in result

    def test_removes_mention(self):
        from rag.helper_models.language_detector.language_detector import (
            TextPreprocessor,
        )

        result = TextPreprocessor.preprocess("thanks @doctor for the advice")
        assert "@doctor" not in result

    def test_collapses_whitespace(self):
        from rag.helper_models.language_detector.language_detector import (
            TextPreprocessor,
        )

        result = TextPreprocessor.preprocess("hello   world")
        assert "  " not in result

    def test_non_string_returns_empty(self):
        from rag.helper_models.language_detector.language_detector import (
            TextPreprocessor,
        )

        assert TextPreprocessor.preprocess(None) == ""
        assert TextPreprocessor.preprocess(123) == ""

    def test_empty_string_returns_empty(self):
        from rag.helper_models.language_detector.language_detector import (
            TextPreprocessor,
        )

        assert TextPreprocessor.preprocess("") == ""

    def test_plain_text_unchanged(self):
        from rag.helper_models.language_detector.language_detector import (
            TextPreprocessor,
        )

        result = TextPreprocessor.preprocess("I feel anxious")
        assert "anxious" in result


# ═════════════════════════════════════════════════════════════════════════════
# LanguageDetector.predict() — happy paths
# ═════════════════════════════════════════════════════════════════════════════


class TestLanguageDetectorPredict:
    def test_returns_dict(self):
        det, _ = make_detector(top_lang="en", top_conf=0.95)
        result = det.predict("Hello world")
        assert isinstance(result, dict)

    def test_returns_language_key(self):
        det, _ = make_detector(top_lang="en", top_conf=0.95)
        result = det.predict("Hello world")
        assert "language" in result

    def test_returns_confidence_key(self):
        det, _ = make_detector(top_lang="en", top_conf=0.95)
        result = det.predict("Hello world")
        assert "confidence" in result

    def test_returns_reliable_key(self):
        det, _ = make_detector(top_lang="en", top_conf=0.95)
        result = det.predict("Hello world")
        assert "reliable" in result

    def test_detects_english(self):
        det, _ = make_detector(top_lang="en", top_conf=0.97)
        result = det.predict("I feel very anxious")
        assert result["language"] == "english"
        assert result["reliable"] is True

    def test_detects_arabic(self):
        det, _ = make_detector(top_lang="ar", top_conf=0.92)
        result = det.predict("أنا أشعر بالقلق")
        assert result["language"] == "arabic"
        assert result["reliable"] is True

    def test_uncertain_when_below_threshold(self):
        det, _ = make_detector(threshold=0.80, top_lang="en", top_conf=0.50)
        result = det.predict("some ambiguous text")
        assert result["language"] == "uncertain"
        assert result["reliable"] is False

    def test_reliable_true_when_above_threshold(self):
        det, _ = make_detector(threshold=0.60, top_lang="fr", top_conf=0.85)
        result = det.predict("Je suis triste")
        assert result["reliable"] is True

    def test_confidence_value_correct(self):
        det, _ = make_detector(top_lang="en", top_conf=0.91)
        result = det.predict("Hello")
        assert result["confidence"] == pytest.approx(0.91, abs=1e-3)


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════


class TestLanguageDetectorEdgeCases:
    def test_empty_string_does_not_raise(self):
        det, _ = make_detector()
        result = det.predict("")
        assert isinstance(result, dict)

    def test_very_long_text_does_not_raise(self):
        det, _ = make_detector()
        result = det.predict("I feel sad. " * 300)
        assert isinstance(result, dict)

    def test_text_with_only_urls_does_not_raise(self):
        det, _ = make_detector()
        result = det.predict("https://example.com https://another.com")
        assert isinstance(result, dict)

    def test_threshold_boundary_exact_value(self):
        """Confidence exactly at threshold should be considered reliable."""
        det, _ = make_detector(threshold=0.70, top_lang="en", top_conf=0.70)
        result = det.predict("hello")
        # >= threshold → reliable
        assert result["reliable"] is True

    def test_threshold_just_below(self):
        det, _ = make_detector(threshold=0.70, top_lang="en", top_conf=0.699)
        result = det.predict("hello")
        assert result["reliable"] is False


# ═════════════════════════════════════════════════════════════════════════════
# Error paths
# ═════════════════════════════════════════════════════════════════════════════


class TestLanguageDetectorErrors:
    def test_model_load_failure_raises_runtime_error(self):
        env = {"LANGUAGE_DETECTION_MODEL_PATH": "/fake/path.pkl"}
        with patch.dict(os.environ, env):
            with patch(PATCH_JOBLIB, side_effect=FileNotFoundError("no file")):
                from rag.helper_models.language_detector.language_detector import (
                    LanguageDetector,
                )

                with pytest.raises(RuntimeError):
                    LanguageDetector(threshold=0.70)

    def test_predict_proba_failure_raises(self):
        det, mock_model = make_detector()
        mock_model.predict_proba.side_effect = RuntimeError("model exploded")
        with pytest.raises(RuntimeError):
            det.predict("I feel fine")
