"""Contract test for :class:`FaissDenseRetriever` (canonical dense backend).

The whole suite is inherited from :class:`RetrieveConnectorContractBase`; this
class only wires up the five adapter methods. The corpus is crafted for the
hash embedder, which splits on whitespace without stripping punctuation: the
overlapping tokens are exact whitespace tokens (no trailing periods), the query
shares ``cat``/``pet`` with ``d2`` and just ``pet`` with ``d1`` (a positively
scoring runner-up for the score-margin assertion), and the distractors share no
token with the query, so their cosine is zero and the family drops them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.connectors.retrieve.faiss_retriever import FaissDenseRetriever
from tests.connectors.retrieve.retrieve_contract import RetrieveConnectorContractBase


class TestFaissDenseRetriever(RetrieveConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseRetrieveConnector]:
        return FaissDenseRetriever

    @classmethod
    def backend_value(cls) -> str:
        return "faiss"

    @classmethod
    def sample_corpus(cls) -> List[Dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "the mat lay flat on the floor by the window"},
            {"doc_id": "d1", "text": "a dog can be a loyal and energetic pet"},
            {"doc_id": "d2", "text": "a cat is an independent and curious pet"},
            {"doc_id": "d3", "text": "cars need regular engine oil and maintenance"},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "cat pet"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "d2"
