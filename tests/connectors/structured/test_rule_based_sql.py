"""Contract test for :class:`RuleBasedSql` (zero-download CI anchor)."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Type

from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector
from rag_integration.feature_groups.connectors.structured.rule_based_sql import RuleBasedSql
from tests.connectors.structured.structured_contract import StructuredConnectorContractBase


class TestRuleBasedSql(StructuredConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseStructuredConnector]:
        return RuleBasedSql

    @classmethod
    def backend_value(cls) -> str:
        return "rule_based"

    @classmethod
    def table_name(cls) -> str:
        return "pets"

    @classmethod
    def columns(cls) -> List[str]:
        return ["name", "species", "age"]

    @classmethod
    def rows(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "Whiskers", "species": "cat", "age": 3},
            {"name": "Rex", "species": "dog", "age": 5},
            {"name": "Felix", "species": "cat", "age": 2},
        ]

    @classmethod
    def key_column(cls) -> str:
        return "name"

    @classmethod
    def count_question(cls) -> str:
        return "how many pets are there"

    @classmethod
    def filter_question(cls) -> str:
        return "which pets have species cat"

    @classmethod
    def expected_filter_keys(cls) -> Set[str]:
        return {"Whiskers", "Felix"}
