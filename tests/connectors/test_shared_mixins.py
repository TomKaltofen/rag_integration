"""Tests for the shared connector mixins and error types.

These pin the cross-cutting behaviour extracted from the per-family ``base.py``
files into ``connectors/mixins.py`` and ``connectors/errors.py``:

- every connector error subclasses the built-in ``ValueError`` (so existing
  ``pytest.raises(ValueError)`` callers keep catching them), and
- ``top_k`` parsing is now uniform across every ``TopKMixin`` family: a
  non-integer value is rejected as an ``InvalidOptionError`` naming the key and
  the offending value, rather than each family surfacing a different raw
  ``int()`` failure. Retrieve already pinned this in
  ``test_base_validation.py``; this locks the same contract for the rerank,
  graph_rag, and orchestrator families that the refactor unified onto it.
"""

from __future__ import annotations

from typing import Type

import pytest

from mloda.user import Options

from rag_integration.feature_groups.connectors.errors import (
    ConnectorError,
    DuplicateDocIdError,
    GroundingError,
    InvalidOptionError,
    MissingOptionError,
    RankingContractError,
    SqlSafetyError,
)
from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector
from rag_integration.feature_groups.connectors.mixins import TopKMixin
from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector
from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector


class TestErrorHierarchy:
    """Every connector error is a ``ValueError`` so existing callers keep catching it."""

    @pytest.mark.parametrize(
        "error_cls",
        [
            ConnectorError,
            MissingOptionError,
            InvalidOptionError,
            DuplicateDocIdError,
            RankingContractError,
            GroundingError,
            SqlSafetyError,
        ],
    )
    def test_subclasses_value_error(self, error_cls: Type[Exception]) -> None:
        assert issubclass(error_cls, ValueError)


_TOPK_FAMILIES = [BaseRetrieveConnector, BaseRerankConnector, BaseGraphRagConnector, BaseOrchestratorConnector]


class TestSharedTopK:
    """``TopKMixin._get_top_k`` parses uniformly across every family that mixes it in."""

    @pytest.mark.parametrize("base", _TOPK_FAMILIES)
    def test_garbage_top_k_raises_naming_key_and_value(self, base: Type[TopKMixin]) -> None:
        options = Options(context={base.TOP_K: "not-an-int"})
        with pytest.raises(InvalidOptionError, match="top_k.*not-an-int"):
            base._get_top_k(options)

    @pytest.mark.parametrize("base", _TOPK_FAMILIES)
    def test_non_coercible_type_raises_value_error(self, base: Type[TopKMixin]) -> None:
        # A list cannot become an int; before the refactor the non-validating
        # families surfaced a bare TypeError here, now it is a loud ValueError.
        options = Options(context={base.TOP_K: [1, 2]})
        with pytest.raises(ValueError):
            base._get_top_k(options)

    @pytest.mark.parametrize("base", _TOPK_FAMILIES)
    def test_absent_top_k_falls_back_to_default(self, base: Type[TopKMixin]) -> None:
        assert base._get_top_k(Options(context={})) == base.DEFAULT_TOP_K

    @pytest.mark.parametrize("base", _TOPK_FAMILIES)
    def test_string_integer_is_coerced(self, base: Type[TopKMixin]) -> None:
        assert base._get_top_k(Options(context={base.TOP_K: "3"})) == 3
