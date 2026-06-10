import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv(".env")


DATA_SOURCE = (
    "hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json"
)
DS_NAME = os.getenv("RAG_DATASET")

df = pd.read_json(DATA_SOURCE, lines=True)
df.to_csv(f"RAG/{DS_NAME}.csv", index=False)
