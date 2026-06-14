import sys
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# ── NEW ──
from logger import get_logger
logger = get_logger(__name__)
from telemetry import record_llm_duration,record_tokens
import time

class LLMCaller:
    def __init__(
        self,
        prompt: str,
        identifier: str = "None",
        isIntent: bool = False,
        verbose: bool = False,
    ):
        load_dotenv()
        self.model = os.getenv("SIDE_MODEL")
        self.identifier = identifier
        self.is_intent = isIntent
        self.verbose = verbose

        try:
            self.API_KEY = os.getenv("GROQ_API_KEY")
            if not self.API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
        except Exception as e:
            logger.critical("Error loading API key: %s", e)
            sys.exit(1)

        self._initialize_LLM()
        self.prompt = prompt

    def _initialize_LLM(self):
        model_name = (
            self.model if not self.is_intent else os.getenv("INTENT_CLASSIFICATION_MODEL")
        )
        self.llm = ChatGroq(
            model=model_name,
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=self.API_KEY,
        )
        logger.info("Initialized LLMCaller [%s] with model: %s", self.identifier, model_name)

    def call(self, arguments: dict):
        prompt = self.prompt
        for key, val in arguments.items():
            prompt = prompt.replace(key, val)

        logger.debug("[%s] Sending prompt to LLM (length=%d)", self.identifier, len(prompt))

        start = time.time()
        try:
            generated_text = self.llm.invoke([HumanMessage(content=prompt)])
            output = generated_text.content.strip()
            usage = getattr(generated_text, "usage_metadata", None) or {}
            logger.debug("[%s] usage_metadata: %s", self.identifier, usage)

            record_tokens(
                step=self.identifier,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
        except Exception as e:
            logger.error("[%s] LLM call failed: %s", self.identifier, e, exc_info=True)
            raise
        duration_s = time.time() - start

        record_llm_duration(self.identifier, duration_s)

        if self.verbose:
            logger.debug("[%s] PROMPT: %s", self.identifier, prompt[:500])
            logger.debug("[%s] OUTPUT: %s", self.identifier, output[:500])

        logger.info("[%s] LLM call completed (output length=%d)", self.identifier, len(output))
        return output