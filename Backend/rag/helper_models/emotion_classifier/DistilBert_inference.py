from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import os
from dotenv import load_dotenv

# ── NEW ──
from logger import get_logger
logger = get_logger(__name__)

load_dotenv(".env")

class EmotionClassifier:
    def __init__(self):
        model_path = os.getenv("EMOTION_MODEL_PATH")
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

        logger.debug("Emotion prediction: '%s' (confidence=%.4f)", emotion_name, result["score"])
        return emotion_name, result["score"]