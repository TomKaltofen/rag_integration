"""Rule-based text-to-SQL backend.

Canonical concrete for the ``structured`` family: zero-download, deterministic,
no LLM. Translates a small set of natural-language intents (count, equality
filter, list-all) into parameterised SQL. There is no mature deterministic
NL->SQL library (every real one needs an LLM), so a transparent rule set is the
right CI anchor; LLM-backed translators are pedigree backends for later.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RuleBasedSql(BaseStructuredConnector):
    """Rule-based NL->SQL (``structured_backend="rule_based"``).

    Intents, in priority order:

    1. Count (``how many`` / ``count``): ``SELECT COUNT(*) AS cnt FROM <table>``.
    2. Equality filter (a column name followed by a value token):
       ``SELECT * FROM <table> WHERE LOWER(<col>) = ?`` (case-insensitive).
    3. Otherwise list all rows: ``SELECT * FROM <table>``.

    Table and column names are validated identifiers (by the base); values are
    always bound parameters, never interpolated.
    """

    STRUCTURED_BACKENDS = {
        "rule_based": "Rule-based natural-language-to-SQL (no LLM)",
    }

    PROPERTY_MAPPING = {
        BaseStructuredConnector.STRUCTURED_BACKEND: {"explanation": "Use 'rule_based' for rule-based text-to-SQL"},
        BaseStructuredConnector.QUESTION: {"explanation": "Natural-language question to answer over the table"},
        BaseStructuredConnector.TABLE: {"explanation": "Table name (a simple SQL identifier)"},
        BaseStructuredConnector.COLUMNS: {"explanation": "Column names (simple SQL identifiers)"},
        BaseStructuredConnector.ROWS: {"explanation": "Table rows: a list of {column: value} dicts"},
    }

    @classmethod
    def _to_sql(cls, question: str, table: str, columns: List[str]) -> Tuple[str, List[Any]]:
        tokens = _TOKEN_RE.findall(question.lower())
        token_set = set(tokens)

        # table and column are validated identifiers (by the base); the filter
        # value is always returned as a bound parameter, never interpolated.
        if "count" in token_set or ("how" in token_set and "many" in token_set):
            return f"SELECT COUNT(*) AS cnt FROM {table}", []  # nosec

        for column in columns:
            lowered = column.lower()
            if lowered in tokens:
                position = tokens.index(lowered)
                if position + 1 < len(tokens):
                    value = tokens[position + 1]
                    return f"SELECT * FROM {table} WHERE LOWER({column}) = ?", [value]  # nosec

        return f"SELECT * FROM {table}", []  # nosec
