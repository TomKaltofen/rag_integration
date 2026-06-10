"""Lexical token-overlap reranker.

Canonical concrete for the ``rerank`` family: zero-download, zero-dependency
(pure Python stdlib), deterministic. Scores each candidate by the number of
query tokens it contains and reorders best-first. A cheap, honest reranker that
anchors the CI contract suite with no model, network, or third-party library.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class LexicalReranker(BaseRerankConnector):
    """Token-overlap reranker (``rerank_backend="lexical"``).

    Score = number of distinct query tokens present in the candidate. Ties are
    broken by the candidate's original position, so the ordering is stable and
    deterministic.
    """

    RERANK_BACKENDS = {
        "lexical": "Token-overlap lexical reranking (pure Python)",
    }

    PROPERTY_MAPPING = {
        BaseRerankConnector.RERANK_BACKEND: {"explanation": "Use 'lexical' for token-overlap reranking"},
        BaseRerankConnector.QUERY_TEXT: {"explanation": "Query the candidates are reranked against"},
        BaseRerankConnector.TOP_K: {
            "explanation": f"Number of passages to return after reranking (default {BaseRerankConnector.DEFAULT_TOP_K})"
        },
        BaseRerankConnector.CANDIDATES: {"explanation": "Candidate passages: a list of {doc_id, text} dicts"},
    }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_TOKEN_RE.findall(text.lower()))

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        query_tokens = cls._tokenize(query)
        scored = [(idx, float(len(query_tokens & cls._tokenize(text)))) for idx, text in enumerate(texts)]
        # Best score first; ties broken by original index for a stable order.
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
