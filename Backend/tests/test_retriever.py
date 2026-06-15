"""
Tests for rag/retriever.py  →  Retriever
HuggingFaceEmbeddings, QdrantVectorStore, QdrantClient, and CrossEncoder
are all mocked so no real models or network connections are used.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


PATCH_EMBEDDINGS    = "rag.retriever.HuggingFaceEmbeddings"
PATCH_QDRANT_CLIENT = "rag.retriever.QdrantClient"
PATCH_VECTORSTORE   = "rag.retriever.QdrantVectorStore"
PATCH_CROSS_ENCODER = "rag.retriever.CrossEncoder"


# ── Factory ───────────────────────────────────────────────────────────────────

def make_retriever(docs=None, scores=None):
    """
    Build a Retriever with all external deps mocked.
    docs  : list of (Document-like mock, score) tuples returned by similarity_search_with_score
    scores: flat list of floats returned by cross_encoder.predict
    """
    if docs is None:
        doc1 = MagicMock()
        doc1.page_content = "What is anxiety?"
        doc1.metadata = {"Response": ["Anxiety is a feeling of worry.", "It can be treated."]}
        docs = [(doc1, 0.9)]

    if scores is None:
        # one score per (query, response) pair across all docs
        scores = [0.8, 0.7]  # matches two responses in default docs

    with (
        patch(PATCH_EMBEDDINGS)    as MockEmbed,
        patch(PATCH_QDRANT_CLIENT) as MockClient,
        patch(PATCH_VECTORSTORE)   as MockVS,
        patch(PATCH_CROSS_ENCODER) as MockCE,
    ):
        vs_instance = MagicMock()
        vs_instance.similarity_search_with_score.return_value = docs
        MockVS.return_value = vs_instance

        ce_instance = MagicMock()
        ce_instance.predict.return_value = scores
        MockCE.return_value = ce_instance

        from rag.retriever import Retriever

        retriever = Retriever(
            embedding_model="mock-model",
            reranking_model="mock-reranker",
            device="cpu",
            vector_db_args={"collection_name": "test-collection"},
            url="http://localhost:6333",
            api_key="test-key",
        )

        return retriever, vs_instance, ce_instance


# ═════════════════════════════════════════════════════════════════════════════
# retrieve() — happy paths
# ═════════════════════════════════════════════════════════════════════════════

class TestRetrieverRetrieve:

    def test_returns_list(self):
        retriever, _, _ = make_retriever()
        result = retriever.retrieve("I feel anxious", max_context=3, max_responses=10)
        assert isinstance(result, list)

    def test_returns_tuples_of_question_and_response(self):
        retriever, _, _ = make_retriever()
        result = retriever.retrieve("query", max_context=3, max_responses=10)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_similarity_search_called_with_correct_k(self):
        retriever, vs, _ = make_retriever()
        retriever.retrieve("query", max_context=5, max_responses=10)
        vs.similarity_search_with_score.assert_called_once_with("query", k=5)

    def test_cross_encoder_called(self):
        retriever, _, ce = make_retriever()
        retriever.retrieve("query", max_context=3, max_responses=10)
        ce.predict.assert_called_once()

    def test_results_limited_by_max_responses(self):
        doc1 = MagicMock()
        doc1.page_content = "Q1"
        doc1.metadata = {"Response": [f"R{i}" for i in range(10)]}
        scores = list(range(10, 0, -1))  # descending scores

        retriever, _, _ = make_retriever(docs=[(doc1, 0.9)], scores=scores)
        result = retriever.retrieve("query", max_context=1, max_responses=3)
        assert len(result) <= 3

    def test_results_sorted_by_score_descending(self):
        """The cross-encoder with higher scores should appear first."""
        doc1 = MagicMock()
        doc1.page_content = "Q1"
        doc1.metadata = {"Response": ["low quality answer", "high quality answer"]}
        # Second response gets higher score
        scores = [0.2, 0.9]

        retriever, _, _ = make_retriever(docs=[(doc1, 0.8)], scores=scores)
        result = retriever.retrieve("query", max_context=1, max_responses=2)
        # First returned response should correspond to score 0.9
        assert result[0][1] == "high quality answer"

    def test_question_text_preserved_in_result(self):
        doc1 = MagicMock()
        doc1.page_content = "What is depression?"
        doc1.metadata = {"Response": ["Depression is a mood disorder."]}
        scores = [0.95]

        retriever, _, _ = make_retriever(docs=[(doc1, 0.9)], scores=scores)
        result = retriever.retrieve("query", max_context=1, max_responses=1)
        assert result[0][0] == "What is depression?"


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestRetrieverEdgeCases:

    def test_empty_docs_returns_empty_list(self):
        retriever, _, _ = make_retriever(docs=[], scores=[])
        result = retriever.retrieve("query", max_context=3, max_responses=10)
        assert result == []

    def test_empty_query_does_not_raise(self):
        retriever, _, _ = make_retriever()
        result = retriever.retrieve("", max_context=3, max_responses=10)
        assert isinstance(result, list)

    def test_max_responses_larger_than_available(self):
        """Asking for more responses than exist should return all available."""
        doc1 = MagicMock()
        doc1.page_content = "Q1"
        doc1.metadata = {"Response": ["Only one response."]}
        scores = [0.8]

        retriever, _, _ = make_retriever(docs=[(doc1, 0.9)], scores=scores)
        result = retriever.retrieve("query", max_context=1, max_responses=100)
        assert len(result) == 1

    def test_multiple_docs_multiple_responses(self):
        doc1 = MagicMock()
        doc1.page_content = "Q1"
        doc1.metadata = {"Response": ["R1a", "R1b"]}

        doc2 = MagicMock()
        doc2.page_content = "Q2"
        doc2.metadata = {"Response": ["R2a"]}

        scores = [0.9, 0.7, 0.5]

        retriever, _, _ = make_retriever(
            docs=[(doc1, 0.9), (doc2, 0.8)], scores=scores
        )
        result = retriever.retrieve("query", max_context=2, max_responses=3)
        assert len(result) <= 3


# ═════════════════════════════════════════════════════════════════════════════
# Error paths
# ═════════════════════════════════════════════════════════════════════════════

class TestRetrieverErrors:

    def test_similarity_search_failure_raises(self):
        retriever, vs, _ = make_retriever()
        vs.similarity_search_with_score.side_effect = RuntimeError("Qdrant timeout")
        with pytest.raises(RuntimeError):
            retriever.retrieve("query")

    def test_cross_encoder_failure_raises(self):
        retriever, _, ce = make_retriever()
        ce.predict.side_effect = RuntimeError("Cross-encoder crashed")
        with pytest.raises(RuntimeError):
            retriever.retrieve("query")
