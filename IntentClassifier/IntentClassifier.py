from prompt import IntentClassificationPrompt
import subprocess
import sys
import tempfile
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv


class IntentClassifier:
    def __init__(self, model):
        self.model = model
        load_dotenv()
        self.API_KEY = os.getenv("GROQ_API_KEY")
        self.initialize_LLM()
        

    def classify(self, text):
        prompt = IntentClassificationPrompt.format(text=text)
        generated_text = self.llm.invoke([HumanMessage(content=prompt)])
        return generated_text.content.strip()
    
    def initialize_LLM(self):
        self.llm = ChatGroq(
            model=self.model,
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=self.API_KEY
        )
        
IntentClassClient = IntentClassifier(model="meta-llama/llama-4-scout-17b-16e-instruct")
print(IntentClassClient.classify("Hello, how are you?"))
print(IntentClassClient.classify("Who was in the kitchen with the knife?"))