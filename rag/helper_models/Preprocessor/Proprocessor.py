import re


class Preprocessor:
    def __init__(self):
        pass

    def process(self, retrieval: str) -> str:
        # remove closed tags and unclosed tags (e.g. <img with truncated base64)
        # since they overflow the context window limit for llms.
        result = re.sub(r"<.*?>", "", retrieval, flags=re.DOTALL)
        result = re.sub(r"<[^>]*$", "", result, flags=re.DOTALL)
        return result.strip()
