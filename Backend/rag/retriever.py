from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# ── NEW ──
from logger import get_logger
logger = get_logger(__name__)


class Retriever:
    def __init__(
            self, 
            embedding_model, 
            reranking_model, 
            device, 
            vector_db_args, 
            url, 
            api_key
        ):

        self.embedding_model = embedding_model
        self.reranking_model = reranking_model
        self.device = device
        self.vector_db_args = vector_db_args

        logger.info(
            "Initializing Retriever | embedding=%s | reranker=%s | device=%s",
            embedding_model, reranking_model, device,
        )

        self._initialize_models()
        self._initialize_vector_db(url, api_key)

        logger.info("Retriever fully initialized")

    def _initialize_models(self):
        logger.debug("Loading embedding model: %s", self.embedding_model)
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.debug("Loading cross-encoder reranker: %s", self.reranking_model)
        self.cross_encoder = CrossEncoder(self.reranking_model, device=self.device)

    def _initialize_vector_db(self, url, api_key):
        logger.info("Connecting to Qdrant at %s", url)
        self.vector_db_client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=120,
        )
        self.vectorstore = QdrantVectorStore(
            client=self.vector_db_client,
            collection_name=self.vector_db_args['collection_name'],
            embedding=self.embedder,
        )
        logger.info("Vector DB connection established (collection=%s)", self.vector_db_args['collection_name'])

    def retrieve(self, query, max_context=3, max_responses=10):
        logger.info(
            "Retrieving | query='%s' | max_context=%d | max_responses=%d",
            query[:80], max_context, max_responses,
        )

        retrieved_docs = self.vectorstore.similarity_search_with_score(query, k=max_context)
        logger.debug("Similarity search returned %d documents", len(retrieved_docs))

        q_r_pairs = []
        retrieved_questions = [question.page_content for question, _ in retrieved_docs]
        for i, doc in enumerate(retrieved_docs):
            doc = doc[0]
            for response in doc.metadata["Response"]:
                q_r_pairs.append((i, response))

        scores = self.cross_encoder.predict([(query, response) for _, response in q_r_pairs])
        top_responses = sorted(zip(scores, q_r_pairs), key=lambda x: x[0], reverse=True)[:max_responses]
        responses = [response[1] for response in top_responses]

        retrievals = [(retrieved_questions[i], q_r_pair) for i, q_r_pair in responses]

        logger.info("Retrieval complete – returning %d q-r pairs", len(retrievals))
        return retrievals