import ast
import pandas as pd
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv('.env')

DS_NAME = os.getenv('GROUPED_RAG_DATASET')

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

client = QdrantClient(path="./qdrant_db")
client.create_collection(
    collection_name="mental_health",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

df = pd.read_csv(DS_NAME)
df = df.dropna(subset=["Response"])

docs = [
    Document(
        page_content=row["Context"],
        metadata={"Response": ast.literal_eval(row["Response"])}
    )
    for _, row in df.iterrows()
]

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="mental_health",
    embedding=embeddings,
)

vectorstore.add_documents(docs)
print(f"inserted {len(docs)} documents")