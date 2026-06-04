import joblib
import numpy as np
import re
from typing import Dict


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
        self.model_path = "rag/helper_models/model_objs/language_detector.pkl"
        self.threshold = threshold
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

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as e:
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
            return {"language": "uncertain", "confidence": top_conf, "reliable": False}

        return {"language": top_lang, "confidence": top_conf, "reliable": True}


if __name__ == "__main__":
    lang_detector = LanguageDetector(threshold=0.60)
    text = "عليا الطلاق الملك نمبر وان"
    result = lang_detector.predict(text)
    print(result)
