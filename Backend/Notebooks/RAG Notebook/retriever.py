from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder


class Retriever:
    def __init__(
        self, embedding_model, reranking_model, device, vector_db_args, url, api_key
    ):

        self.embedding_model = embedding_model
        self.reranking_model = reranking_model
        self.device = device
        self.vector_db_args = vector_db_args

        self._initialize_models()
        self._initialize_vector_db(url, api_key)

    def _initialize_models(self):

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.cross_encoder = CrossEncoder(self.reranking_model, device=self.device)

    def _initialize_vector_db(self, url, api_key):
        self.vector_db_client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=120,
        )

        self.vectorstore = QdrantVectorStore(
            client=self.vector_db_client,
            collection_name=self.vector_db_args["collection_name"],
            embedding=self.embedder,
        )

    def retrieve(self, query, max_context=3, max_responses=10):

        retrieved_docs = self.vectorstore.similarity_search_with_score(
            query, k=max_context
        )

        q_r_pairs = []  # [Q]uestion [R]esponse
        retrieved_questions = [
            question.page_content for question, _ in retrieved_docs
        ]  # extract the questions from the retrieved docs
        for i, doc in enumerate(retrieved_docs):
            doc = doc[0]  # extract the Document object from the (Document, score) tuple
            for response in doc.metadata["Response"]:
                q_r_pairs.append((i, response))

        scores = self.cross_encoder.predict(
            [(query, response) for _, response in q_r_pairs]
        )
        top_responses = sorted(
            zip(scores, q_r_pairs), key=lambda x: x[0], reverse=True
        )[:max_responses]
        responses = [
            response[1] for response in top_responses
        ]  # [(context1, response1), (context2, response2), ...]

        retrievals = [(retrieved_questions[i], q_r_pair) for i, q_r_pair in responses]

        return retrievals
