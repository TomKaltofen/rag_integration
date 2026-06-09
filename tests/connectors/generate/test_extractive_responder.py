"""Contract test for :class:`ExtractiveResponder` (zero-download CI anchor)."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector
from rag_integration.feature_groups.connectors.generate.extractive_responder import ExtractiveResponder
from tests.connectors.generate.generate_contract import GenerateConnectorContractBase


class TestExtractiveResponder(GenerateConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseGenerateConnector]:
        return ExtractiveResponder

    @classmethod
    def backend_value(cls) -> str:
        return "extractive"

    @classmethod
    def sample_passages(cls) -> List[Dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d1", "text": "Cats need fresh water, a clean litter box, and daily play."},
            {"doc_id": "d2", "text": "Dogs are loyal companions."},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "what do cats need"

    @classmethod
    def expected_citation_doc_id(cls) -> str:
        return "d1"

    @classmethod
    def expected_answer_substring(cls) -> str:
        return "fresh water"
