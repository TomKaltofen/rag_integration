"""Base class for the ``rerank`` connector family.

Contract: ``query_text + candidates + top_k -> reordered passages with scores``.

A rerank connector takes a set of candidate passages (already retrieved, e.g.
by the ``retrieve`` family) and reorders them by relevance to the query,
returning the top_k after reranking. It is a ROOT FeatureGroup here: candidates
are passed inline through ``Options`` so the family is self-contained and
contract-testable without a network or an upstream stage. In a two-stage
pipeline the candidates would come from a retrieve connector; the rerank logic
is identical either way.

Output (single row, keyed by the root feature name)::

    {"reranked_passages": [{"doc_id": ..., "text": ..., "score": ..., "rank": ...}, ...]}

``score`` is the rerank score (higher is more relevant); ``rank`` is 0-based,
ascending, best first.

This mirrors the ``retrieve`` family by design (selector-gated matching, the
``_rank`` hoist, base-side validation of returned indices). It deliberately
copies that pattern rather than subclassing ``BaseRetrieveConnector``: the
input is ``candidates`` not ``corpus``, and keeping the families decoupled lets
each evolve its own contract.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class BaseRerankConnector(FeatureGroup):
    """Root FeatureGroup for rerank-connector backends.

    A concrete backend declares its selector value in ``RERANK_BACKENDS`` and
    implements :meth:`_rank` (the only per-backend logic); the base owns the
    empty-input / ``top_k`` clamping, the passage-assembly contract, and the
    validation of returned indices. Selection is done entirely by
    :meth:`match_feature_group_criteria`, gating on
    ``rerank_backend in cls.RERANK_BACKENDS``; disjoint selector values keep
    backends mutually exclusive. The base keeps ``RERANK_BACKENDS`` empty so it
    never matches.
    """

    ROOT_FEATURE_NAME = "reranked_passages"

    # Option keys.
    RERANK_BACKEND = "rerank_backend"
    QUERY_TEXT = "query_text"
    TOP_K = "top_k"
    CANDIDATES = "candidates"

    DEFAULT_TOP_K = 5

    # Filled per concrete: {backend_value: human-readable description}. Disjoint
    # across backends; empty on the base so it never matches.
    RERANK_BACKENDS: Dict[str, str] = {}

    # Declarative option documentation only; selection is via
    # ``match_feature_group_criteria`` (not the FeatureChainParser).
    PROPERTY_MAPPING = {
        RERANK_BACKEND: {"explanation": "Which rerank-connector backend to use"},
        QUERY_TEXT: {"explanation": "Query the candidates are reranked against"},
        TOP_K: {"explanation": f"Number of passages to return after reranking (default {DEFAULT_TOP_K})"},
        CANDIDATES: {"explanation": "Candidate passages to rerank: a list of {doc_id, text} dicts"},
    }

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({cls.ROOT_FEATURE_NAME})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match the root feature name only for a backend this concrete declares.

        Gating on ``rerank_backend`` keeps backends mutually exclusive; an
        unknown backend matches nothing (honest surface).
        """
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        backend = options.get(cls.RERANK_BACKEND)
        return backend in cls.RERANK_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features (candidates arrive via Options)."""
        return None

    @classmethod
    def _get_top_k(cls, options: Options) -> int:
        val = options.get(cls.TOP_K)
        return int(val) if val is not None else cls.DEFAULT_TOP_K

    @classmethod
    def _get_candidates(cls, options: Options) -> List[Dict[str, Any]]:
        candidates = options.get(cls.CANDIDATES)
        if candidates is None:
            raise ValueError(
                f"{cls.__name__} requires '{cls.CANDIDATES}' in options: a list of {{doc_id, text}} dicts."
            )
        return list(candidates)

    @classmethod
    @abstractmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        """Reorder ``texts`` by relevance to ``query``.

        Returns up to ``top_k`` ``(candidate_index, score)`` pairs, ordered
        best-first, where ``score`` is higher-is-more-relevant. Indices must be
        in range (``0 <= candidate_index < len(texts)``) and unique (validated
        by the base). ``top_k`` is already clamped to ``1 <= top_k <=
        len(texts)``. The base does not re-sort, so best-first is required.
        """
        ...

    @classmethod
    def _validate_ranking(cls, ranked: List[Tuple[int, float]], n_candidates: int) -> None:
        """Reject out-of-range or duplicate indices from a backend's ``_rank``."""
        seen: Set[int] = set()
        for idx, _score in ranked:
            if not 0 <= idx < n_candidates:
                raise ValueError(
                    f"{cls.__name__}._rank returned out-of-range index {idx} for {n_candidates} candidates."
                )
            if idx in seen:
                raise ValueError(f"{cls.__name__}._rank returned duplicate index {idx}.")
            seen.add(idx)

    @classmethod
    def _rerank(
        cls,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Assemble the reranked-passage contract around the backend's :meth:`_rank`."""
        if not candidates:
            return []
        effective_k = min(top_k, len(candidates))
        if effective_k <= 0:
            return []

        texts = [str(doc.get("text", "")) for doc in candidates]
        doc_ids = [str(doc.get("doc_id", str(i))) for i, doc in enumerate(candidates)]

        ranked = cls._rank(query, texts, effective_k)
        cls._validate_ranking(ranked, len(candidates))

        passages: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(ranked):
            passages.append({"doc_id": doc_ids[idx], "text": texts[idx], "score": float(score), "rank": rank})
        return passages

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Rerank the candidates against the query, return reordered passages."""
        for feature in features.features:
            options = feature.options
            query = options.get(cls.QUERY_TEXT)
            if query is None:
                raise ValueError(f"{cls.__name__} requires '{cls.QUERY_TEXT}' in options.")
            candidates = cls._get_candidates(options)
            top_k = cls._get_top_k(options)
            passages = cls._rerank(str(query), candidates, top_k)
            return [{cls.ROOT_FEATURE_NAME: passages}]
        return []
