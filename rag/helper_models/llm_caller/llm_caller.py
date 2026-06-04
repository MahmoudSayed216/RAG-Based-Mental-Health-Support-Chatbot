import sys
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


class LLMCaller:
    def __init__(self, prompt:str, identifier: str="None", is_intent: bool=False, verbose:bool = False):
        load_dotenv()
        self.model = os.getenv("SIDE_MODEL")
        self.identifier = identifier
        self.is_intent = is_intent

        try:
            self.API_KEY = os.getenv("GROQ_API_KEY")
            if not self.API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
        except Exception as e:
            print(f"Error loading API key: {e}")
            sys.exit(1)

        self._initialize_LLM()
        self.prompt = prompt

    def _initialize_LLM(self):
        self.llm = ChatGroq(
            model=self.model if not self.is_intent else os.getenv("INTENT_CLASSIFICATION_MODEL"),
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=self.API_KEY,
        )

        print(
            f"Initialized LLMCaller with model: {self.model if not self.isIntent else os.getenv('INTENT_CLASSIFICATION_MODEL')}"
        )

    def call(self, arguments: dict):
        prompt = self.prompt
        for key, val in arguments.items():
            prompt = prompt.replace(key, val)
        print(f"___LOGS FROM {self.identifier} LLM___")
        print("PROMPT: ", prompt)
        generated_text = self.llm.invoke([HumanMessage(content=prompt)])
        output = generated_text.content.strip()
        print(f"OUTPUT OF {self.identifier} LLM")
        print(output)
        print(f"___END OF LOGS FROM {self.identifier} LLM___")

        # with open(f"{self.identifier}_prompt.txt", "w") as f:
        #     f.write(prompt)
        return output
