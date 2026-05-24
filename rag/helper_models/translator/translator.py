import sys
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


class Translator:
    def __init__(self):
        load_dotenv()
        self.model = os.getenv("INTENT_CLASSIFICATION_MODEL")
        prompt_file = open("./rag/helper_models/translator/prompt.txt")
        self.prompt = prompt_file.read()
        prompt_file.close()
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
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=5,
            api_key=self.API_KEY,
        )

    def translate(self, src_lang, dst_lang, text):
        prompt = self.prompt
        prompt = prompt.replace("{src_lang}", src_lang)
        prompt = prompt.replace("{dst_lang}", dst_lang)
        prompt = prompt.replace("{text}", text)

        print("TRANSLATOR PROMPT: ", prompt)
        generated_text = self.llm.invoke([HumanMessage(content=prompt)])
        return generated_text.content.strip()


# if __name__ == "__main__":
#     translator = Translator()

#     output = translator.translate("arabic", "english", "شعرت مؤخراً بانفصال تام عن كل من حولي. فقدت وظيفتي منذ شهرين، ومنذ ذلك الحين أقضي اليوم كله في السرير، ولا أجد في نفسي طاقة للقيام بأي شيء. لا أعرف من أتحدث إليه، ولا من أين أبدأ.")

#     print(output)
