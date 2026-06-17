import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv(".env")

OLD_DS_NAME = os.getenv("RAG_DATASET")
NEW_DS_NAME = os.getenv("GROUPED_RAG_DATASET")

duplicates_counter = 0


def count_duplicates(responses: list[str]):
    global duplicates_counter
    passed = set()
    for i in range(len(responses)):
        if i in passed:
            continue
        for j in range(i + 1, len(responses)):
            if j in passed:
                continue
            if responses[i] == responses[j]:
                duplicates_counter += 1
                passed.add(j)


def remove_duplicates(responses: list[str]):
    passed = set()
    for i in range(len(responses)):
        if i in passed:
            continue
        for j in range(i + 1, len(responses)):
            if j in passed:
                continue
            if responses[i] == responses[j]:
                passed.add(j)

    for i in sorted(list(passed), reverse=True):
        responses.pop(i)


df = pd.read_csv(f"RAG/{OLD_DS_NAME}.csv")

df.dropna(inplace=True)

grouped = df.groupby("Context")["Response"].apply(list).reset_index()


grouped["Response"].apply(count_duplicates)
print("Number of duplicates: ", duplicates_counter)
grouped["Response"].apply(remove_duplicates)
duplicates_counter = 0
grouped["Response"].apply(count_duplicates)
print("Number of duplicates: ", duplicates_counter)


grouped.to_csv(f"rag/{NEW_DS_NAME}.csv")
