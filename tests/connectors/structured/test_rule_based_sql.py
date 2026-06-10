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

    @classmethod
    def filter_value(cls) -> str:
        return "cat"

    # -- Backend-specific behavior ---------------------------------------------

    def test_count_with_filter_counts_only_matching_rows(self) -> None:
        """Count intent must keep the filter: COUNT over cats only, not all rows."""
        result = self._query("how many pets have species cat")
        assert "?" in result["sql"]
        (only_value,) = result["rows"][0].values()
        assert only_value == 2

    def test_snake_case_column_maps_to_filter(self) -> None:
        """The tokenizer keeps underscores, so snake_case columns are recognised."""
        rows = [{"pet_name": "Rex", "unit_price": 5}, {"pet_name": "Felix", "unit_price": 7}]
        result = RuleBasedSql._query("which items have unit_price 5", "items", ["pet_name", "unit_price"], rows)
        assert 'LOWER("unit_price") = ?' in result["sql"]
        assert [row["pet_name"] for row in result["rows"]] == ["Rex"]

    def test_decimal_filter_value_binds_full_number(self) -> None:
        """The tokenizer keeps decimals, so "2.5" binds whole (not "2" then "5")."""
        rows = [{"name": "Rex", "age": 2.5}, {"name": "Felix", "age": 3}]
        result = RuleBasedSql._query("which pets have age 2.5", "pets", ["name", "age"], rows)
        assert "?" in result["sql"]
        assert "2.5" not in result["sql"]
        assert [row["name"] for row in result["rows"]] == ["Rex"]
