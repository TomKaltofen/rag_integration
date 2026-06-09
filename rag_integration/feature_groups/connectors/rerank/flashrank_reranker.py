"""FlashRank cross-encoder reranker.

Pedigree concrete for the ``rerank`` family: a real neural cross-encoder that
exercises the distinguishing semantics of reranking, kept light by FlashRank's
ONNX runtime (no torch). Behind the ``rerank`` extra. The model (~4 MB for the
default ``ms-marco-TinyBERT-L-2-v2``) downloads on first use and is cached, so
its contract test is skipped on CI (network) but runs locally; the zero-download
``LexicalReranker`` is the always-on CI anchor.
"""

from __future__ import annotations

import threading
from typing import Any, List, Tuple

from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector


class FlashRankReranker(BaseRerankConnector):
    """Cross-encoder reranking via FlashRank (``rerank_backend="flashrank"``).

    The default model is ``ms-marco-TinyBERT-L-2-v2`` (~4 MB, Apache-2.0). The
    ranker is cached at class level (keyed by model name) since constructing it
    loads the ONNX model; loading is guarded by a lock so concurrent callers do
    not build it twice.
    """

    DEFAULT_MODEL = "ms-marco-TinyBERT-L-2-v2"
    MODEL_NAME = "rerank_model"

    RERANK_BACKENDS = {
        "flashrank": "Cross-encoder reranking (FlashRank, ONNX)",
    }

    PROPERTY_MAPPING = {
        BaseRerankConnector.RERANK_BACKEND: {"explanation": "Use 'flashrank' for cross-encoder reranking"},
        BaseRerankConnector.QUERY_TEXT: {"explanation": "Query the candidates are reranked against"},
        BaseRerankConnector.TOP_K: {
            "explanation": f"Number of passages to return after reranking (default {BaseRerankConnector.DEFAULT_TOP_K})"
        },
        BaseRerankConnector.CANDIDATES: {"explanation": "Candidate passages: a list of {doc_id, text} dicts"},
        MODEL_NAME: {"explanation": f"FlashRank model name (default {DEFAULT_MODEL})"},
    }

    _ranker_cache: tuple[str, Any] | None = None
    _cache_lock = threading.Lock()

    @classmethod
    def _get_ranker(cls, model_name: str) -> Any:
        from flashrank import Ranker

        cache = cls._ranker_cache
        if cache is not None and cache[0] == model_name:
            return cache[1]
        with cls._cache_lock:
            cache = cls._ranker_cache
            if cache is not None and cache[0] == model_name:
                return cache[1]
            ranker = Ranker(model_name=model_name)
            cls._ranker_cache = (model_name, ranker)
            return ranker

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        from flashrank import RerankRequest

        ranker = cls._get_ranker(cls.DEFAULT_MODEL)
        # Use the candidate's list index as the passage id so results map back
        # to positions regardless of how FlashRank reorders them.
        passages = [{"id": str(idx), "text": text} for idx, text in enumerate(texts)]
        ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [(int(item["id"]), float(item["score"])) for item in ranked[:top_k]]
