"""Aggregation-aware rule-based text-to-SQL backend.

Second concrete for the ``structured`` family: zero-download, deterministic, no
LLM. Where :class:`RuleBasedSql` covers count/filter/list intents, this backend
adds *aggregation* (avg/min/max/sum over a numeric column named in the
question), translating it to a parameterised aggregate ``SELECT``. It reuses the
base's identifier whitelist, sqlglot read-only-SELECT guard, and sqlite
execution. The count/filter/list intents are reimplemented here too (rather than
inherited from ``RuleBasedSql``) so the backend subclasses the family base
directly and satisfies the shared contract suite on its own.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Natural-language aggregation cues -> SQL aggregate function. Each function is
# a fixed literal (never user text), so interpolating it is injection-safe.
_AGGREGATIONS = {
    "average": "AVG",
    "avg": "AVG",
    "mean": "AVG",
    "minimum": "MIN",
    "min": "MIN",
    "lowest": "MIN",
    "smallest": "MIN",
    "maximum": "MAX",
    "max": "MAX",
    "highest": "MAX",
    "largest": "MAX",
    "sum": "SUM",
    "total": "SUM",
}


class AggregateSql(BaseStructuredConnector):
    """Aggregation-aware rule-based NL->SQL (``structured_backend="aggregate"``).

    Intents, in priority order:

    1. Aggregate (an aggregation cue such as ``average``/``min``/``max``/``sum``
       plus a column named in the question):
       ``SELECT <FUNC>(<col>) AS result FROM <table>``.
    2. Count (``how many`` / ``count``): ``SELECT COUNT(*) AS cnt FROM <table>``.
    3. Equality filter (a column name followed by a value token):
       ``SELECT * FROM <table> WHERE LOWER(<col>) = ?`` (case-insensitive).
    4. Otherwise list all rows: ``SELECT * FROM <table>``.

    Table and column names are validated identifiers (by the base) and the
    aggregate function comes from a fixed whitelist; values are always bound
    parameters, never interpolated.
    """

    STRUCTURED_BACKENDS = {
        "aggregate": "Aggregation-aware rule-based natural-language-to-SQL (no LLM)",
    }

    PROPERTY_MAPPING = {
        BaseStructuredConnector.STRUCTURED_BACKEND: {
            "explanation": "Use 'aggregate' for aggregation-aware text-to-SQL"
        },
        BaseStructuredConnector.QUESTION: {"explanation": "Natural-language question to answer over the table"},
        BaseStructuredConnector.TABLE: {"explanation": "Table name (a simple SQL identifier)"},
        BaseStructuredConnector.COLUMNS: {"explanation": "Column names (simple SQL identifiers)"},
        BaseStructuredConnector.ROWS: {"explanation": "Table rows: a list of {column: value} dicts"},
    }

    @classmethod
    def _find_column(cls, tokens: List[str], columns: List[str]) -> str | None:
        """Return the first column named anywhere in the question, or None."""
        lowered = {column.lower(): column for column in columns}
        for token in tokens:
            if token in lowered:
                return lowered[token]
        return None

    @classmethod
    def _to_sql(cls, question: str, table: str, columns: List[str]) -> Tuple[str, List[Any]]:
        tokens = _TOKEN_RE.findall(question.lower())
        token_set = set(tokens)

        # 1. Aggregation: an aggregation cue plus a column named in the question.
        # Checked first by design, so "the average age" aggregates rather than
        # being read as a filter; the trade-off is that a filter whose *value*
        # token is itself a cue word (e.g. "... status max") would aggregate
        # instead. table, column, and the aggregate function are all whitelisted
        # (the function is a fixed literal), so the f-string is injection-safe.
        for token in tokens:
            function = _AGGREGATIONS.get(token)
            if function is not None:
                column = cls._find_column(tokens, columns)
                if column is not None:
                    return f"SELECT {function}({column}) AS result FROM {table}", []  # nosec
                break

        # 2. Count.
        if "count" in token_set or ("how" in token_set and "many" in token_set):
            return f"SELECT COUNT(*) AS cnt FROM {table}", []  # nosec

        # 3. Equality filter: a column name followed by a value token. The value
        # is always returned as a bound parameter, never interpolated.
        for column in columns:
            lowered = column.lower()
            if lowered in tokens:
                position = tokens.index(lowered)
                if position + 1 < len(tokens):
                    value = tokens[position + 1]
                    return f"SELECT * FROM {table} WHERE LOWER({column}) = ?", [value]  # nosec

        # 4. List all.
        return f"SELECT * FROM {table}", []  # nosec
