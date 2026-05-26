import sys
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


class Summarizer:
    def __init__(self):
        load_dotenv()
        self.model = os.getenv("SIDE_MODEL")
        try:
            self.API_KEY = os.getenv("GROQ_API_KEY")
            if not self.API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
        except Exception as e:
            print(f"Error loading API key: {e}")
            sys.exit(1)

        self._initialize_LLM()
        self._initialize_prompt("./rag/helper_models/summarizer/prompt.txt")
    
    def _initialize_prompt(self, file_path):
        prompt_file = open(file=file_path)
        self.prompt = prompt_file.read()
        prompt_file.close()
    
    def _initialize_LLM(self):
        self.llm = ChatGroq(
            model=self.model,
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=self.API_KEY,
        )

    def summarize(self, text):
        prompt = self.prompt
        prompt = prompt.replace("{references}", text)

        print("TRANSLATOR PROMPT: ", prompt)
        generated_text = self.llm.invoke([HumanMessage(content=prompt)])
        return generated_text.content.strip()

