from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

load_dotenv(".env")

_EMOTION_MODEL_REPO = "Abdellmohsennn/final_mental_emotion_model"


def _ensure_emotion_model(incoming_val: str) -> str:
    # FIX 1: Use incoming_val to avoid UnboundLocalError
    if not incoming_val:
        path_str = "model_objs/final_mental_emotion_model"
    else:
        path_str = incoming_val
        
    incoming_path = Path(path_str)
    
    # FIX 2: Point cleanly to a root-level model_objs directory 
    # instead of the old nested 'rag/...' tree paths
    if "C:" in path_str or "\\" in path_str or not incoming_path.exists():
        folder_name = incoming_path.name if incoming_path.name else "final_mental_emotion_model"
        target_path = Path("model_objs") / folder_name
    else:
        target_path = incoming_path

    final_path = str(target_path.resolve())

    if os.path.exists(final_path) and os.listdir(final_path):
        return final_path

    logger.info("Local emotion model snapshot not found at %s — pulling from Hub", final_path)
    os.makedirs(final_path, exist_ok=True)
    
    downloaded = snapshot_download(
        repo_id=_EMOTION_MODEL_REPO,
        local_dir=final_path
    )
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
