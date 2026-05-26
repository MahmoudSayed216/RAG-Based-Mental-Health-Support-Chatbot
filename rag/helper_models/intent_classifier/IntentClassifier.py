from .prompt import IntentClassificationPrompt
import sys
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


class IntentClassifier:
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

        self.initialize_LLM()

    def initialize_LLM(self):
        self.llm = ChatGroq(
            model=self.model,
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=self.API_KEY,
        )

    def classify(self, text):
        prompt = IntentClassificationPrompt.format(text=text)
        generated_text = self.llm.invoke([HumanMessage(content=prompt)])
        return generated_text.content.strip()


if __name__ == "__main__":
    classifier = IntentClassifier()
    test_sentences = [
        "Hello, how are you?",
        "Thank you for your help!",
        "Can you tell me about anxiety?",
        "Okay! Goodbye!",
        "Who won the World Cup in 2022?",
    ]

    print("\n--- Intent Classification Results ---")
    for text in test_sentences:
        intent = classifier.classify(text)
        print(f"Text: {text}")
        print(f"Predicted Intent: {intent}\n")
