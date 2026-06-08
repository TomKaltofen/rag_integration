"""BM25 lexical retrieve connector, backed by the ``bm25s`` library.

Canonical concrete for the ``retrieve`` family: zero-download (no model, no
network), deterministic, and contract-canonical (bm25s ``retrieve`` returns
ranked indices with scores directly). MIT-licensed, numpy-only.
"""

from __future__ import annotations

from typing import Any, Dict, List

from mloda.provider import DefaultOptionKeys

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector


class Bm25sRetriever(BaseRetrieveConnector):
    """Lexical BM25 retrieval over an inline corpus.

    Selected with ``retrieve_backend="bm25s"``. Builds an in-memory BM25 index
    per call (the corpus is per-call), so there is no shared state to cache and
    repeated calls are idempotent.
    """

    RETRIEVE_BACKENDS = {
        "bm25s": "BM25 lexical retrieval (bm25s)",
    }

    PROPERTY_MAPPING = {
        BaseRetrieveConnector.RETRIEVE_BACKEND: {
            **RETRIEVE_BACKENDS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseRetrieveConnector.QUERY_TEXT: {
            "explanation": "Raw text query to search the corpus",
            DefaultOptionKeys.context: True,
        },
        BaseRetrieveConnector.TOP_K: {
            "explanation": "Number of passages to return",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: BaseRetrieveConnector.DEFAULT_TOP_K,
        },
        BaseRetrieveConnector.CORPUS: {
            "explanation": "Inline corpus: a list of {doc_id, text} dicts",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _retrieve(
        cls,
        query: str,
        corpus: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        import bm25s

        if not corpus:
            return []

        effective_k = min(top_k, len(corpus))
        if effective_k <= 0:
            return []

        texts = [str(doc.get("text", "")) for doc in corpus]
        doc_ids = [str(doc.get("doc_id", str(i))) for i, doc in enumerate(corpus)]

        corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)

        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        indices, scores = retriever.retrieve(query_tokens, k=effective_k, show_progress=False)

        passages: List[Dict[str, Any]] = []
        for rank in range(effective_k):
            corpus_idx = int(indices[0][rank])
            passages.append(
                {
                    "doc_id": doc_ids[corpus_idx],
                    "text": texts[corpus_idx],
                    "score": float(scores[0][rank]),
                    "rank": rank,
                }
            )
        return passages
