import os
import ast
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import google.generativeai as genai
from langchain_core.globals import set_verbose, set_debug
from helper_models import (
    EmotionClassifier,
    IntentClassifier,
    LanguageDetector,
    Translator,
)
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ENUMS import LanguagesEnums, IntentEnums

set_verbose(True)
set_debug(True)

load_dotenv(".env")


class Generator:
    QDRANT_PATH = "./qdrant_db"
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
    TOP_K = int(os.getenv("TOP_K"))  # number of relevant questions to retrieve
    EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
    GEMINI_MODEL = os.getenv("GEMINI_GENERATION_MODEL")

    def __init__(self):
        self._initialize_helper_models()
        self._initalize_qdrant_db()
        self._initalize_prompt(file_path="rag/prompt.txt")

    def _initialize_helper_models(self):

        # Helper models
        self.emotion_classifier = EmotionClassifier()
        self.intent_classifier = IntentClassifier()
        self.language_classifier = LanguageDetector(threshold=0.6)
        self.translator = Translator()
        ## Answer Generation LLM
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.gemini = genai.GenerativeModel(self.GEMINI_MODEL)

    def _initalize_qdrant_db(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBED_MODEL,
            model_kwargs={"device": os.getenv("DEVICE")},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.client = QdrantClient(path=self.QDRANT_PATH)
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.COLLECTION_NAME,
            embedding=embeddings,
        )

    def _retrieve(self, query: str, top_k) -> list[dict]:
        """Return top-k matched contexts and their associated responses."""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        retrieved = []
        print("____________RETRIEVED CONTEXT____________")
        for doc, score in results:
            responses = doc.metadata.get("Response", [])
            # stored as a list already, but guard against stringified lists
            if isinstance(responses, str):
                responses = ast.literal_eval(responses)
            print("")
            retrieved.append(
                {
                    "context": doc.page_content,
                    "responses": responses,
                    "score": score,
                }
            )
            print(doc.page_content)
            print(score)
        print("____________END OF RETRIEVED CONTEXT____________")

        return retrieved

    def _initalize_prompt(self, file_path):
        file = open(file_path)
        self.prompt = file.read()
        file.close()

    def _build_prompt(
        self, user_query: str, retrieved: list[dict], emotion: str
    ) -> str:
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

        self.prompt = self.prompt.replace("{references}", references)
        self.prompt = self.prompt.replace("{user_query}", user_query)
        self.prompt = self.prompt.replace("{emotion}", emotion)

        return self.prompt

    def answer(self, user_query: str, top_k: int = TOP_K, verbose: bool = False) -> str:

        language = self.language_classifier.predict(user_query)

        if language != LanguagesEnums.ENGLISH.value:
            print("Langugae: ", language)
            user_query = self.translator.translate(
                src_lang=language["language"], dst_lang="English", text=user_query
            )

        intent = self.intent_classifier.classify(user_query)
        print("INTENT: ", intent)

        response = ""

        if intent == IntentEnums.ASKING.value:
            emotion = self.emotion_classifier.predict_emotion(text=user_query)[0]
            """Full RAG pipeline: retrieve → build prompt → generate."""
            retrieved = self._retrieve(user_query, top_k=self.TOP_K)

            if verbose:
                print(f"\n📌 Top-{top_k} retrieved contexts:")
                for i, item in enumerate(retrieved, 1):
                    print(f"  {i}. [{item['score']:.3f}] {item['context'][:80]}...")
            print("HHHHH")
            print("EMOTION: ", emotion)
            prompt = self._build_prompt(user_query, retrieved, emotion)
            print("PROMPT: ", prompt)

            response = self.gemini.generate_content(prompt)
            response = response.text

            if language != LanguagesEnums.ENGLISH.value:
                response = self.translator.translate(
                    src_lang="English", dst_lang=language["language"], text=response
                )

        elif intent == IntentEnums.GREETINGS.value:
            response = "Hello! I'm fine thank you!"  ## better be  AI GENERATED

        elif intent == IntentEnums.GRATITUDE.value:
            response = "You are welcome, Anytime :)"

        elif intent == IntentEnums.GOODBYE.value:
            response = "Bye, Have a good time :)"

        elif intent == IntentEnums.OUT_OF_SCOPE.value:
            response = "Your Question is out of the scope that I'm designed for"

        return response

    def __del__(self):
        self.client.close()


def main():
    generator = Generator()
    print(
        "🧠 Mental Health RAG Chatbot  (type 'quit' to exit, 'top_k=N' to change retrieval depth)\n"
    )
    top_k = 3

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
            reply = generator.answer(user_input, top_k=top_k, verbose=False)
            print(reply)
        except Exception as e:
            print(f"[Error] {e}")
        print()


if __name__ == "__main__":
    main()
