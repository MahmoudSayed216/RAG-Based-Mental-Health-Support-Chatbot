from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

model_path = "DistilModel/final_mental_emotion_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_token_type_ids=False)

# Testing examples for each of the 6 classes
test_sentences = [
    "i feel awful about it too because it s my job to help her.",       # sadness
    "i feel like i have performed well and achieved a huge milestone.",  # joy
    "i feel romantic and nostalgic when thinking about our time.",       # love
    "i feel so frustrated and irritated by the way things are handled.", # anger
    "i remember feeling acutely distressed and anxious.",                # fear
    "i keep feeling pleasantly surprised at the unexpected support."     # surprise
]
LabelMap = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

print("\n--- Inference Results ---")
for text in test_sentences:
    result = classifier(text)[0]
    raw_label = result['label']

    if raw_label.startswith("LABEL_"):
        label_id = int(raw_label.split('_')[-1])
        emotion_name = LabelMap.get(label_id, "Unknown")
    else:
        emotion_name = raw_label

    print(f"Text: {text}")
    print(f"Predicted: {emotion_name.upper()} (Confidence: {result['score']:.4f})\n")