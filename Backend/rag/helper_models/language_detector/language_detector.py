import joblib
import numpy as np
import re
from typing import Dict
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from logger import get_logger

logger = get_logger(__name__)

_LANGUAGE_DETECTOR_REPO = "Abdellmohsennn/language_detector"
_LANGUAGE_DETECTOR_FILE = "language_detector.pkl"
load_dotenv()


def _ensure_language_model(path: str) -> str:
    if path and os.path.exists(path):
        return path
    logger.info("Local model not found at %s — downloading from HF Hub", path)
    local_dir = os.path.dirname(path)
    os.makedirs(local_dir, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=_LANGUAGE_DETECTOR_REPO,
        filename=_LANGUAGE_DETECTOR_FILE,
        local_dir=local_dir,
    )
    logger.info("Downloaded language detector to %s", downloaded)
    return downloaded


class TextPreprocessor:
    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    HANDLE_RE = re.compile(r"[@#]\w+")
    SPACE_RE = re.compile(r"\s+")

    @staticmethod
    def preprocess(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = TextPreprocessor.URL_RE.sub(" ", text)
        t = TextPreprocessor.HANDLE_RE.sub(" ", t)
        t = TextPreprocessor.SPACE_RE.sub(" ", t).strip()
        return t


class LanguageDetector:
    def __init__(self, threshold: float = 0.70):
        self.model_path = _ensure_language_model(
            os.getenv("LANGUAGE_DETECTION_MODEL_PATH")
        )
        self.threshold = threshold
        logger.info(
            "Loading LanguageDetector from %s (threshold=%.2f)",
            self.model_path,
            threshold,
        )
        self.model = self._load_model()
        self.languages_map = {
            "ar": "arabic",
            "bg": "bulgarian",
            "de": "german",
            "el": "greek",
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "hi": "hindi",
            "it": "italian",
            "ja": "japanese",
            "nl": "dutch",
            "pl": "polish",
            "pt": "portuguese",
            "ru": "russian",
            "sw": "swahili",
            "th": "thai",
            "tr": "turkish",
            "ur": "urdu",
            "vi": "vietnamese",
            "zh": "chinese",
        }
        logger.info("LanguageDetector loaded successfully")

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
            logger.critical("Failed to load language detection model: %s", e)
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, text: str) -> Dict:
        clean_text = TextPreprocessor.preprocess(text)

        proba = self.model.predict_proba([clean_text])[0]
        classes = self.model.classes_
        top_idx = np.argmax(proba)
        top_conf = float(proba[top_idx])
        top_lang = classes[top_idx]
        top_lang = self.languages_map[top_lang]

        if top_conf < self.threshold:
            logger.debug("Language uncertain for text (conf=%.2f)", top_conf)
            return {"language": "uncertain", "confidence": top_conf, "reliable": False}

        logger.debug("Detected language: %s (conf=%.2f)", top_lang, top_conf)
        return {"language": top_lang, "confidence": top_conf, "reliable": True}


if __name__ == "__main__":
    lang_detector = LanguageDetector(threshold=0.60)
    text = "عليا الطلاق الملك نمبر وان"
    result = lang_detector.predict(text)
    print(result)
