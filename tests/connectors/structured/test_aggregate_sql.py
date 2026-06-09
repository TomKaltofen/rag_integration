"""Contract test for :class:`AggregateSql` (zero-download CI anchor).

Inherits the whole structured contract suite (count/filter/safety), then adds a
backend-specific proof: an aggregation question runs a real aggregate query and
returns a known computed value (avg of ``[2, 3, 5, 2]`` = ``3.0``), which the
count/filter-only sibling cannot answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Type

from rag_integration.feature_groups.connectors.structured.aggregate_sql import AggregateSql
from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector
from tests.connectors.structured.structured_contract import StructuredConnectorContractBase


class TestAggregateSql(StructuredConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseStructuredConnector]:
        return AggregateSql

    @classmethod
    def backend_value(cls) -> str:
        return "aggregate"

    @classmethod
    def table_name(cls) -> str:
        return "pets"

    @classmethod
    def columns(cls) -> List[str]:
        return ["name", "species", "age"]

    @classmethod
    def rows(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "Whiskers", "species": "cat", "age": 2},
            {"name": "Rex", "species": "dog", "age": 3},
            {"name": "Felix", "species": "cat", "age": 5},
            {"name": "Tom", "species": "cat", "age": 2},
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
        return {"Whiskers", "Felix", "Tom"}

    # -- Backend-specific proof: aggregation ----------------------------------

    def test_average_question_returns_computed_value(self) -> None:
        """Not-a-stub proof for this backend: an aggregation question runs a real
        AVG over the column and returns the known value (avg of [2,3,5,2] = 3.0)."""
        result = self._query("what is the average age")
        assert "AVG(age)" in result["sql"]
        assert len(result["rows"]) == 1
        (only_value,) = result["rows"][0].values()
        assert only_value == 3.0

    def test_max_question_returns_computed_value(self) -> None:
        """A second aggregate intent: MAX over the column returns the known max."""
        result = self._query("what is the maximum age")
        assert "MAX(age)" in result["sql"]
        (only_value,) = result["rows"][0].values()
        assert only_value == 5
