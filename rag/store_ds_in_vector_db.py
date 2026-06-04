import ast
import pandas as pd
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os



def main():

    load_dotenv(".env")

    DS_NAME = os.getenv("GROUPED_RAG_DATASET")

    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        model_kwargs={"device": os.getenv("DEVICE")},
        encode_kwargs={"normalize_embeddings": True},
    )

    client = QdrantClient(
        url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,  
    )

    collection_name = os.getenv("EMBEDDINGS_COLLECTION_NAME")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=int(os.getenv("EMBEDDING_SIZE")),
                distance=Distance.COSINE,
            ),
        )

    df = pd.read_csv(f"rag/{DS_NAME}")
    df = df.dropna(subset=["Response"])

    docs = [
        Document(
            page_content=row["Context"],
            metadata={"Response": ast.literal_eval(row["Response"])},
        )
        for _, row in df.iterrows()
    ]

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


    def chunk_data(data, size=256):
        for i in range(0, len(data), size):
            yield data[i : i + size]


    total_inserted = 0

    for batch_num, batch in enumerate(chunk_data(docs, size=256), start=1):
        vectorstore.add_documents(batch)
        total_inserted += len(batch)
        print(
            f"Batch {batch_num}: inserted {len(batch)} docs "
            f"(total={total_inserted}/{len(docs)})"
        )

    print(f"Inserted {total_inserted} documents")


if __name__ == "__main__":
    main()
    