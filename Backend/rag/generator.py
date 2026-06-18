import os
from dotenv import load_dotenv

from rag.helper_models import (
    EmotionClassifier,
    LanguageDetector,
    LLMCaller,
    Preprocessor,
)

from .retriever import Retriever
import sys
import importlib

import time

from logger import get_logger
from telemetry import (
    record_intent,
    record_query_length,
    record_rag_usage,
    check_and_record_refusal,
    record_llm_duration,
    record_emotion,
    record_language,
)

logger = get_logger(__name__)

# FIX: removed os.system("clear") — clears terminal on every instantiation,
#      wipes logs in production / containerized environments.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_enums = importlib.import_module("ENUMS")
LanguagesEnums = _enums.LanguagesEnums
IntentEnums = _enums.IntentEnums

load_dotenv(".env")


class Generator:
    def __init__(
        self,
        top_k: int = 3,
        top_r: int = 10,
        verbose: bool = False,
        summarize_retrievals: bool = False,
        retriever_device: str = "cpu",
        vector_db_args: str = "",
        embedding_model: str = "",
        reranking_model: str = "",
        vector_db_url: str = "",
        vector_db_api_key: str = "",
    ):
        self.top_k = top_k
        self.top_r = top_r
        self.verbose = verbose
        # FIX: caller (main.py) now passes a real bool after comparing the env string,
        #      so this assignment is always correct.
        self.summarize_retrievals = summarize_retrievals
        self.Rag_Usage = False

        logger.info("Initializing Generator with top_k=%d, top_r=%d", top_k, top_r)

        self._initialize_prompts()
        self._initialize_helper_models()

        self.retriever = Retriever(
            embedding_model=embedding_model,
            reranking_model=reranking_model,
            device=retriever_device,
            vector_db_args=vector_db_args,
            url=vector_db_url,
            api_key=vector_db_api_key,
        )

        self.Preprocessor = Preprocessor()
        logger.info("Generator fully initialized")

    def _initialize_prompts(self):
        self.system_prompt = self._initalize_prompt(file_path="rag/prompt.txt")
        self.intent_prompt = self._initalize_prompt(
            file_path="rag/helper_models/prompts/intent_prompt.txt"
        )
        self.translation_prompt = self._initalize_prompt(
            file_path="rag/helper_models/prompts/translator_prompt.txt"
        )
        self.summarization_prompt = self._initalize_prompt(
            file_path="rag/helper_models/prompts/summarizer_prompt.txt"
        )
        logger.debug("All prompts loaded")

    def _initialize_helper_models(self):
        self.emotion_classifier = EmotionClassifier()
        self.language_classifier = LanguageDetector(threshold=0.6)
        self.translator = LLMCaller(
            prompt=self.translation_prompt,
            identifier="Translator",
            verbose=self.verbose,
        )
        self.summarizer = LLMCaller(
            prompt=self.summarization_prompt,
            identifier="Summarizer",
            verbose=self.verbose,
        )
        self.responseModel = LLMCaller(
            prompt=self.system_prompt,
            identifier="Response Generator",
            verbose=self.verbose,
        )
        self.intent_classifier = LLMCaller(
            prompt=self.intent_prompt,
            isIntent=True,
            identifier="Intent Classifier",
            verbose=self.verbose,
        )
        logger.debug("All helper models initialized")

    def _initalize_prompt(self, file_path):
        try:
            with open(file_path) as file:
                prompt = file.read()
            logger.debug("Loaded prompt from %s", file_path)
            return prompt
        except Exception as e:
            logger.error("Failed to load prompt from %s: %s", file_path, e)
            raise

    def _format_references(self, retrievals: list[dict]):
        blocks = []
        for i, (question, response) in enumerate(retrievals):
            blocks.append(
                f"[Reference {i}]\n"
                f"Related Question: {question}\n"
                f"Counselor Response:\n {response}"
            )

        if self.verbose:
            for block in blocks:
                logger.debug("BLOCK: %s", block)

        references = "\n\n".join(blocks)
        return references

    def answer(self, user_query: str, history: str, intent_history: str) -> str:
        emotion = "Neutral"

        logger.info("Processing new query: '%s'", user_query[:100])

        # ── Language detection ──
        start_lang = time.time()
        language = self.language_classifier.predict(user_query)
        record_llm_duration("LanguageDetector", time.time() - start_lang)
        logger.debug(f"Language Detector Time: {time.time() - start_lang}")

        if self.verbose:
            logger.debug("Language detection result: %s", language)

        detected_lang = language.get("language")
        record_query_length(len(user_query), language=detected_lang)
        record_language(detected_lang)

        should_translate = detected_lang not in (
            LanguagesEnums.ENGLISH.value,
            "uncertain",
        )

        if should_translate:
            logger.info("Translating from %s to English", detected_lang)
            user_query = self.translator.call(
                {
                    "{src_lang}": detected_lang,
                    "{dst_lang}": "English",
                    "{text}": user_query,
                },
            )

        # ── Intent classification ──
        intent = self.intent_classifier.call(
            {"{text}": user_query, "{history}": intent_history}
        )
        intent = intent.strip().lower() if intent else ""
        logger.info("Detected intent: '%s'", intent)
        record_intent(intent)

        response = ""
        self.Rag_Usage = intent == IntentEnums.ASKING.value
        record_rag_usage(self.Rag_Usage)

        # FIX: emotion was detected but never sent to telemetry — now recorded
        record_emotion(emotion)

        # ── RAG retrieval (only for "asking" intent) ──
        if self.Rag_Usage:
            logger.info("Running RAG pipeline (retrieval + generation)")
            try:
                # ── Emotion classification ──
                emo_start = time.time()
                emotion = self.emotion_classifier.predict_emotion(text=user_query)[0]
                record_llm_duration("EmotionClassifier", time.time() - emo_start)
                logger.debug(f"Emotion time: {time.time() - emo_start}")
                logger.info("Detected emotion: '%s'", emotion)
                retrieved = self.retriever.retrieve(
                    user_query, max_context=self.top_k, max_responses=self.top_r
                )
                references = self._format_references(retrieved)
                logger.debug("Retrieved %d context blocks", len(retrieved))

                if self.summarize_retrievals:
                    summarize_time = time.time()
                    logger.info("Summarizing retrieved references")
                    references = self.summarizer.call({"{references}": references})
                    # FIX: identifier was "Summerizetion Time" (typo) — now matches
                    #      the LLMCaller identifier "Summarizer" for consistent Axiom labels
                    record_llm_duration("Summarizer", time.time() - summarize_time)
                    logger.debug(f"Summarization Time: {time.time() - summarize_time}")

            except Exception as e:
                logger.error("Retrieval failed: %s", e, exc_info=True)
                references = ""
        else:
            references = ""
            logger.info("Non-asking intent – skipping retrieval")

        # ── Response generation ──
        try:
            response = self.responseModel.call(
                {
                    "{references}": f"\t{references}",
                    "{user_query}": f"\t{user_query}",
                    "{history}": f"\t{history}",
                    "{emotion}": emotion,
                    "{intent}": intent,
                },
            )
            check_and_record_refusal(response, intent)

        except Exception as e:
            logger.error("Response generation failed: %s", e, exc_info=True)
            raise

        # ── Translate response back if needed ──
        if should_translate and response:
            logger.info("Translating response back to %s", detected_lang)
            response = self.translator.call(
                {
                    "{src_lang}": "English",
                    "{dst_lang}": detected_lang,
                    "{text}": response,
                },
            )

        logger.info("Response generated successfully (length=%d)", len(response))
        return response
