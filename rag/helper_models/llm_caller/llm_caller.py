import sys
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


class LLMCaller:
    def __init__(self, prompt, identifier="None"):
        load_dotenv()
        self.model = os.getenv("SIDE_MODEL")
        self.identifier = identifier
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
            model=self.model,
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=self.API_KEY,
        )

    def call(self, arguments: dict, verbose=False):
        prompt = self.prompt
        for key, val in arguments.items():
            prompt = prompt.replace(key, val)

        print(f"OUTPUT OF {self.identifier} LLM")
        print("PROMPT: ", prompt)
        generated_text = self.llm.invoke([HumanMessage(content=prompt)])
        output = generated_text.content.strip()
        print("__________")

        # with open(f"{self.identifier}_prompt.txt", "w") as f:
        #     f.write(prompt)
        return output
