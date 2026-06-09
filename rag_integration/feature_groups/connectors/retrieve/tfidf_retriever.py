"""TF-IDF dense retrieve connector.

Second concrete for the ``retrieve`` family: embeds the corpus and query with
the repo's deterministic :class:`TfidfEmbedder` and ranks documents by cosine
similarity. Zero-download (no model, no network), pure-Python, deterministic. A
dense counterpart to the lexical ``bm25s`` backend that anchors the same
contract suite from a different ranking mechanism, and with no new dependency
(it reuses the existing TF-IDF embedder).
"""

from __future__ import annotations

from typing import List, Tuple

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.rag_pipeline.embedding.tfidf import TfidfEmbedder


class TfidfRetriever(BaseRetrieveConnector):
    """TF-IDF dense retrieval over an inline corpus (``retrieve_backend="tfidf"``).

    Embeds the corpus and the query together so they share one IDF/vocabulary,
    then ranks documents by cosine similarity to the query. The embedder
    L2-normalizes every vector, so cosine reduces to a dot product. Ties are
    broken by corpus index, so the ordering is stable and deterministic.
    """

    # The embedder hashes terms into a fixed-width vector; 384 is its own default
    # and is ample for the small inline corpora this family serves. ``model_name``
    # is ignored by the TF-IDF embedder, so the default is passed verbatim.
    EMBEDDING_DIM = 384

    RETRIEVE_BACKENDS = {
        "tfidf": "TF-IDF dense retrieval (cosine over hashed TF-IDF vectors)",
    }

    PROPERTY_MAPPING = {
        BaseRetrieveConnector.RETRIEVE_BACKEND: {"explanation": "Use 'tfidf' for TF-IDF dense retrieval"},
        BaseRetrieveConnector.QUERY_TEXT: {"explanation": "Raw text query to search the corpus"},
        BaseRetrieveConnector.TOP_K: {
            "explanation": f"Number of passages to return (default {BaseRetrieveConnector.DEFAULT_TOP_K})"
        },
        BaseRetrieveConnector.CORPUS: {"explanation": "Inline corpus: a list of {doc_id, text} dicts"},
    }

    @staticmethod
    def _cosine(query_vector: List[float], doc_vector: List[float]) -> float:
        # Both vectors are L2-normalized by the embedder, so the dot product is
        # already the cosine similarity.
        return sum(q * d for q, d in zip(query_vector, doc_vector))

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        # ``_embed_texts`` is the embedder's deterministic raw-text vectorization
        # entry point; embedding the corpus and query in one batch shares a
        # single IDF/vocabulary so the query and documents live in one space.
        vectors = TfidfEmbedder._embed_texts(list(texts) + [query], cls.EMBEDDING_DIM, "default")
        query_vector = vectors[-1]
        doc_vectors = vectors[:-1]

        # A query with no usable terms (empty, or only tokens the embedder drops)
        # embeds to an all-zero vector, leaving every cosine 0 and the ranking
        # meaningless. Nothing is rankable, so return nothing (mirrors how the
        # lexical bm25s sibling handles its degenerate input).
        if not any(query_vector):
            return []

        scored = [(idx, cls._cosine(query_vector, doc_vector)) for idx, doc_vector in enumerate(doc_vectors)]
        # Best score first; ties broken by original index for a stable order.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
