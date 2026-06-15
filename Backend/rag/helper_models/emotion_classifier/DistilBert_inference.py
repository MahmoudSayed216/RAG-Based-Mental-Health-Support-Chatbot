from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from logger import get_logger

logger = get_logger(__name__)

load_dotenv(".env")

_EMOTION_MODEL_REPO = "Abdellmohsennn/final_mental_emotion_model"


def _ensure_emotion_model(path: str) -> str:
    if path and os.path.exists(path):
        return path
    logger.info("Local model not found at %s — downloading from HF Hub", path)
    os.makedirs(path, exist_ok=True)
    downloaded = snapshot_download(repo_id=_EMOTION_MODEL_REPO, local_dir=path)
    logger.info("Downloaded emotion model to %s", downloaded)
    return downloaded


class EmotionClassifier:
    def __init__(self):
        model_path = _ensure_emotion_model(os.getenv("EMOTION_MODEL_PATH"))
        logger.info("Loading EmotionClassifier from %s", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_token_type_ids=False,
        )
        self.label_map = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        }
        logger.info("EmotionClassifier loaded successfully")

    def predict_emotion(self, text):
        result = self.classifier(text)[0]
        raw_label = result["label"]

        if raw_label.startswith("LABEL_"):
            label_id = int(raw_label.split("_")[-1])
            emotion_name = self.label_map.get(label_id, "Unknown")
        else:
            emotion_name = raw_label

        logger.debug(
            "Emotion prediction: '%s' (confidence=%.4f)", emotion_name, result["score"]
        )
        return emotion_name, result["score"]
