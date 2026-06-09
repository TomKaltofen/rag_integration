"""Base-level safety tests for the structured family, through the production path.

Pins that ``BaseStructuredConnector._query`` itself rejects a non-SELECT
statement produced by a backend (not just the isolated ``_validate_select``), so
a regression that dropped the guard from ``_query`` would fail here.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector


class _DeleteBackend(BaseStructuredConnector):
    """A deliberately malicious backend that emits a non-SELECT statement."""

    STRUCTURED_BACKENDS = {"_delete_stub": "test-only stub"}

    @classmethod
    def _to_sql(cls, question: str, table: str, columns: List[str]) -> Tuple[str, List[Any]]:
        return "DELETE FROM pets", []


def test_query_rejects_non_select_through_production_path() -> None:
    with pytest.raises(ValueError):
        _DeleteBackend._query("delete everything", "pets", ["name"], [{"name": "Rex"}])
