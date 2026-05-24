from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


class EmotionClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "RAG/helper_models/model_objs/final_mental_emotion_model"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "RAG/helper_models/model_objs/final_mental_emotion_model"
        )
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

    def predict_emotion(self, text):
        result = self.classifier(text)[0]
        raw_label = result["label"]

        if raw_label.startswith("LABEL_"):
            label_id = int(raw_label.split("_")[-1])
            emotion_name = self.label_map.get(label_id, "Unknown")
        else:
            emotion_name = raw_label

        return emotion_name, result["score"]


# if __name__ == "__main__":

#     # Testing examples for each of the 6 classes
#     classifier = DistilBertEmotionClassifier()
#     test_sentences = [
#         "i feel awful about it too because it s my job to help her.",       # sadness
#         "i feel like i have performed well and achieved a huge milestone.",  # joy
#         "i feel romantic and nostalgic when thinking about our time.",       # love
#         "i feel so frustrated and irritated by the way things are handled.", # anger
#         "i remember feeling acutely distressed and anxious.",                # fear
#         "i keep feeling pleasantly surprised at the unexpected support."     # surprise
#     ]
#     for text in test_sentences:
#         emotion, confidence = classifier.predict_emotion(text)
#         print(f"Text: {text}")
#         print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.4f})\n")
