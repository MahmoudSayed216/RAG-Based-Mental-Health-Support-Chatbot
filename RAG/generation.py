import os
import ast
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import google.generativeai as genai
from langchain_core.globals import set_verbose, set_debug

set_verbose(True)
set_debug(True)

load_dotenv(".env")

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_PATH     = "./qdrant_db"
COLLECTION_NAME = "mental_health"
TOP_K           = 3          # number of relevant questions to retrieve
EMBED_MODEL     = "BAAI/bge-large-en-v1.5"
# GEMINI_MODEL    = "gemini-2.0-flash"   # or "gemini-1.5-pro"
GEMINI_MODEL    = "gemini-2.5-flash"

# ── Init ──────────────────────────────────────────────────────────────────────

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

client = QdrantClient(path=QDRANT_PATH)
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

gemini = genai.GenerativeModel(GEMINI_MODEL)

# ── Core functions ────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return top-k matched contexts and their associated responses."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    retrieved = []
    print("____________RETRIEVED CONTEXT____________")
    for doc, score in results:
        responses = doc.metadata.get("Response", [])
        # stored as a list already, but guard against stringified lists
        if isinstance(responses, str):
            responses = ast.literal_eval(responses)
        print("")
        retrieved.append({
            "context": doc.page_content,
            "responses": responses,
            "score": score,
        })
        print(doc.page_content)
        print(score)
    print("____________END OF RETRIEVED CONTEXT____________")
    
    return retrieved


def build_prompt(user_query: str, retrieved: list[dict]) -> str:
    """Construct the prompt from retrieved context + responses."""
    blocks = []
    for i, item in enumerate(retrieved, 1):
        responses_text = "\n".join(f"  - {r}" for r in item["responses"])
        blocks.append(
            f"[Reference {i}]\n"
            f"Related question: {item['context']}\n"
            f"Counselor responses:\n{responses_text}"
        )

    references = "\n\n".join(blocks)

    prompt = f"""You are a compassionate and knowledgeable mental health support assistant.
Use the counselor references below to inform your answer. Synthesize them into a single,
coherent, empathetic response tailored to the user's specific question.
Do not copy the references verbatim — adapt and integrate them naturally.
If the references are not relevant, rely on your general knowledge and remain supportive.

───────────────────────────────────────
COUNSELOR REFERENCES
───────────────────────────────────────
{references}

───────────────────────────────────────
USER QUESTION
───────────────────────────────────────
{user_query}

───────────────────────────────────────
YOUR RESPONSE
───────────────────────────────────────"""
    return prompt


def answer(user_query: str, top_k: int = TOP_K, verbose: bool = False) -> str:
    """Full RAG pipeline: retrieve → build prompt → generate."""
    retrieved = retrieve(user_query, top_k=top_k)

    if verbose:
        print(f"\n📌 Top-{top_k} retrieved contexts:")
        for i, item in enumerate(retrieved, 1):
            print(f"  {i}. [{item['score']:.3f}] {item['context'][:80]}...")

    prompt = build_prompt(user_query, retrieved)
    response = gemini.generate_content(prompt)
    return response.text


# ── CLI loop ──────────────────────────────────────────────────────────────────

def main():
    print("🧠 Mental Health RAG Chatbot  (type 'quit' to exit, 'top_k=N' to change retrieval depth)\n")
    top_k = TOP_K

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye. Take care of yourself 💙")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye. Take care of yourself 💙")
            break

        # allow runtime top_k change: top_k=5
        if user_input.lower().startswith("top_k="):
            try:
                top_k = int(user_input.split("=")[1])
                print(f"  ✅ top_k set to {top_k}")
            except ValueError:
                print("  ❌ Usage: top_k=<integer>")
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            reply = answer(user_input, top_k=top_k, verbose=False)
            print(reply)
        except Exception as e:
            print(f"[Error] {e}")
        print()
    client.close()

if __name__ == "__main__":
    main()