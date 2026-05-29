import os
import ast
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import random
import google.generativeai as genai

# from langchain_core.chat_history import InMemoryChatMessageHistory
from rag.helper_models import (
    EmotionClassifier,
    LanguageDetector,
    LLMCaller,
    Preprocessor,
)
import sys
import importlib

os.system("clear")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_enums = importlib.import_module("ENUMS")
LanguagesEnums = _enums.LanguagesEnums
IntentEnums = _enums.IntentEnums

# set_verbose(True)
# set_debug(True)

load_dotenv(".env")


class Generator:
    QDRANT_PATH = "./qdrant_db"
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
    TOP_K = int(os.getenv("TOP_K"))  # number of relevant questions to retrieve
    EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
    GEMINI_MODEL = os.getenv("GEMINI_GENERATION_MODEL")

    def __init__(
        self,
        top_k: int = TOP_K,
        verbose: bool = False,
        summarize_retrievals: bool = False,
    ):
        self.top_k = top_k
        self.verbose = verbose
        self.summarize_retrievals = summarize_retrievals
        self.Rag_Usage = False
        # Prompts
        self.prompt = self._initalize_prompt(file_path="rag/prompt.txt")
        self.IntentPrompt = self._initalize_prompt(
            file_path="rag/helper_models/prompts/intent_prompt.txt"
        )
        self.TranslationPrompt = self._initalize_prompt(
            file_path="rag/helper_models/prompts/translator_prompt.txt"
        )
        self.SummarizationPrompt = self._initalize_prompt(
            file_path="rag/helper_models/prompts/summarizer_prompt.txt"
        )

        self._initialize_helper_models()
        self._initalize_qdrant_db()
        self._initalize_prompt(file_path="rag/prompt.txt")
        # self.chat_history = InMemoryChatMessageHistory()

    def _initialize_helper_models(self):
        # Helper models
        self.emotion_classifier = EmotionClassifier()
        self.language_classifier = LanguageDetector(threshold=0.6)
        self.Preprocessor = Preprocessor()

        # self.intent_classifier = IntentClassifier()
        # self.translator = Translator()
        # self.summarizer = Summarizer()
        # self._initalize_prompt

        self.intent_classifier = LLMCaller(self.IntentPrompt, "Intent Classifier")
        self.translator = LLMCaller(self.TranslationPrompt, "Translator")
        self.summarizer = LLMCaller(self.SummarizationPrompt, "Summarizer")
        ## Answer Generation LLM
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # self.gemini = genai.GenerativeModel(self.GEMINI_MODEL)
        self.responseModel = LLMCaller(self.prompt, "Response Generator")

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
        # print("____________RETRIEVED CONTEXT____________")
        for doc, score in results:
            responses = doc.metadata.get("Response", [])
            # stored as a list already, but guard against stringified lists
            if isinstance(responses, str):
                responses = ast.literal_eval(responses)
            # print("")
            retrieved.append(
                {
                    "context": doc.page_content,
                    "responses": responses,
                    "score": score,
                }
            )
            # print(doc.page_content)
            # print(score)
        # print("____________END OF RETRIEVED CONTEXT____________")

        return retrieved

    def _initalize_prompt(self, file_path):
        file = open(file_path)
        prompt = file.read()
        file.close()
        return prompt

    def _extract_references(self, retrievals: list[dict]):
        blocks = []
        for i, item in enumerate(retrievals, 1):
            responses_text = "\n".join(f"  - {r}" for r in item["responses"])
            responses_text = self.Preprocessor.process(responses_text)

            # with open("retrievals.txt", "w") as f:
            #     f.write(f"Context:\n{item['context']}\n\nResponses:\n{responses_text}\n\n")

            blocks.append(
                f"[Reference {i}]\n"
                f"Related question: {item['context']}\n"
                f"Counselor responses:\n{responses_text}"
            )

        references = "\n\n".join(blocks)
        return references

    def answer(self, user_query: str, history: str) -> str:

        language = self.language_classifier.predict(user_query)

        if language != LanguagesEnums.ENGLISH.value:
            # print("Langugae: ", language)
            user_query = self.translator.call(
                {
                    "{src_lang}": language["language"],
                    "{dst_lang}": "English",
                    "{text}": user_query,
                },
                verbose=True,
            )
            # user_query = self.translator.translate(
            # src_lang=language["language"], dst_lang="English", text=user_query
            # )

        # intent = self.intent_classifier.classify(user_query)
        intent = self.intent_classifier.call({"text": user_query}, verbose=True)
        intent = intent.strip().lower() if intent else ""
        print("INTENT: ", intent)

        response = ""
        self.Rag_Usage = False

        if intent == IntentEnums.ASKING.value:
            self.Rag_Usage = True
            emotion = self.emotion_classifier.predict_emotion(text=user_query)[0]

            if self.Rag_Usage:
                """Full RAG pipeline: retrieve → build prompt → generate."""
                retrieved = self._retrieve(user_query, top_k=self.TOP_K)

                references = self._extract_references(retrieved)

                if self.summarize_retrievals:
                    # print("pre summarization references\n", references)
                    # references = self.summarizer.summarize(text=references)
                    references = self.summarizer.call(
                        {"references": references}, verbose=True
                    )
                    # print("post summarization references\n", references)
            else:
                references = ""

            # if self.verbose:
            # print(f"\n📌 Top-{self.top_k} retrieved contexts:")
            # for i, item in enumerate(retrieved, 1):
            # print(f"  {i}. [{item['score']:.3f}] {item['context'][:80]}...")
            # pass

            # history = str(self.chat_history.messages) ## will be replaced by redis

            # with open("final_prompt.txt", "w") as f:
            #     f.write(prompt)

            # response = self.gemini.generate_content(prompt)

            response = self.responseModel.call(
                {
                    "references": f"\t{references}",
                    "user_query": f"\t{user_query}",
                    "history": f"\t{history}",
                    "emotion": emotion,
                },
                verbose=True,
            )

            # response = response.text

            if language != LanguagesEnums.ENGLISH.value:
                # response = self.translator.translate(
                #     src_lang="English", dst_lang=language["language"], text=response
                # )
                if language["language"] == "uncertain":
                    language["language"] = "English"
                    # sanity check, should always be true
                response = self.translator.call(
                    {
                        "{src_lang}": "English",
                        "{dst_lang}": language["language"],
                        "{text}": response,
                    },
                    verbose=True,
                )

        elif intent == IntentEnums.GREETINGS.value:
            responses = [
                "Hello! How are you doing today?",
                "Hi there! How are you doing?",
                "Hey! Great to see you. How can I help?",
                "Welcome! What's on your mind today?",
                "Hi! I'm here to listen. How are you feeling?",
                "Hello! It's nice to chat with you. What brings you here?",
                "Hey there! How can I support you today?",
            ]
            response = random.choice(responses)
        elif intent == IntentEnums.GRATITUDE.value:
            response = "You are welcome, Anytime :)"

        elif intent == IntentEnums.GOODBYE.value:
            response = "Bye, Have a good time :)"

        elif intent == IntentEnums.OUT_OF_SCOPE.value:
            responses = [
                "Your Question is out of the scope that I'm designed for",
                "I'm not sure how to help with that, but I'm here if you want to talk about anything related to mental health.",
                "I wish I could help with that, but I'm really only equipped to talk about mental health. If you have any questions or just want to chat about that, I'm here for you!"
                "Sorry, I can't assist with that topic. However, if you have any questions or want to talk about mental health, I'm here to listen and help in any way I can.",
            ]
            response = random.choice(responses)

        # bring this back if you want a local chat history
        # self.chat_history.add_user_message(f"\nUser: {user_query}")
        # self.chat_history.add_ai_message(f"\nAI: {response}")

        # print()
        # print("CHAT HISTORY:")
        # for message in self.chat_history.messages:
        # print(f"{message.content}")
        # print()

        return response


# def main():
#     generator = Generator(summarize_retrievals=True)
#     print(
#         "🧠 Mental Health RAG Chatbot  (type 'quit' to exit, 'top_k=N' to change retrieval depth)\n"
#     )
#     top_k = 3

#     while True:
#         try:
#             user_input = input("You: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print("\nGoodbye. Take care of yourself 💙")
#             break

#         if not user_input:
#             continue

#         if user_input.lower() in {"quit", "exit", "q"}:
#             print("Goodbye. Take care of yourself 💙")
#             break

#         # allow runtime top_k change: top_k=5
#         if user_input.lower().startswith("top_k="):
#             try:
#                 top_k = int(user_input.split("=")[1])
#                 print(f"  ✅ top_k set to {top_k}")
#             except ValueError:
#                 print("  ❌ Usage: top_k=<integer>")
#             continue

#         print("\nAssistant: ", end="", flush=True)
#         try:
#             # print("XXXXXXXX")
#             reply = generator.answer(user_input, top_k=top_k, verbose=False)
#             print("Assistant's response: ", reply)
#         except Exception as e:
#             print(f"[Error] {e}")
#         print()


# if __name__ == "__main__":
#     main()
