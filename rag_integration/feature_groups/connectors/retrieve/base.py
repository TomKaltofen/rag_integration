"""Base class for the ``retrieve`` connector family.

Contract: ``query_text + corpus + top_k -> ranked passages with scores``.

A retrieve connector is a ROOT FeatureGroup (no input features): it takes an
inline corpus and a query through ``Options`` and returns the passages ranked
best-first. Concrete backends (lexical, dense, hybrid, late-interaction) differ
only in the ranking they apply behind this one contract; they declare their
selector value in ``RETRIEVE_BACKENDS`` and implement :meth:`_rank`.

Output (single row, keyed by the root feature name)::

    {"retrieved_passages": [{"doc_id": ..., "text": ..., "score": ..., "rank": ...}, ...]}

``score`` is higher-is-more-relevant; ``rank`` is 0-based, ascending, best
first. Backends return at most ``top_k`` passages and only those with a
positive score, so a degenerate query (empty, or sharing no terms with the
corpus) yields no passages. ``PythonDictFramework`` slices the result to the
requested feature, so the ranked-passage list is the whole contract.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class BaseRetrieveConnector(FeatureGroup):
    """Root FeatureGroup for retrieve-connector backends.

    A concrete backend declares its selector value in ``RETRIEVE_BACKENDS`` and
    implements :meth:`_rank` (the only per-backend logic); the base owns the
    empty-corpus / ``top_k`` clamping and the passage-assembly contract so every
    backend returns an identically shaped result.

    Selection is unambiguous and done entirely by
    :meth:`match_feature_group_criteria`, which gates on
    ``retrieve_backend in cls.RETRIEVE_BACKENDS``. Because each backend declares
    a disjoint selector value and mloda raises when more than one feature group
    matches, at most one backend ever claims a given ``Options``. The base keeps
    ``RETRIEVE_BACKENDS`` empty so it never matches.

    Reuse note for sibling families (rerank, generate, ...): the carry-over is
    the *shape* (the ``RETRIEVE_BACKENDS`` selector dict, the
    ``match_feature_group_criteria`` gating, and the ``_rank`` hoist), not this
    class. The root/``DataCreator``/``_get_corpus`` triad below is
    retrieve-specific: rerank consumes candidate passages as input features
    rather than an inline corpus, so a sibling family copies this pattern, it
    does not subclass ``BaseRetrieveConnector``.
    """

    ROOT_FEATURE_NAME = "retrieved_passages"

    # Option keys.
    RETRIEVE_BACKEND = "retrieve_backend"
    QUERY_TEXT = "query_text"
    TOP_K = "top_k"
    CORPUS = "corpus"

    DEFAULT_TOP_K = 5

    # Filled per concrete: {backend_value: human-readable description}. The base
    # stays empty so it never matches a feature. Values must be disjoint across
    # backends (see the class docstring).
    RETRIEVE_BACKENDS: Dict[str, str] = {}

    # Declarative option documentation only. These root connector groups select
    # by ``match_feature_group_criteria`` (not the FeatureChainParser), so the
    # ``context``/``default``/``strict_validation`` flags that the parser would
    # consume are intentionally omitted here; defaulting and validation live in
    # the code below (``_get_top_k``) and in ``match_feature_group_criteria``.
    PROPERTY_MAPPING = {
        RETRIEVE_BACKEND: {"explanation": "Which retrieve-connector backend to use"},
        QUERY_TEXT: {"explanation": "Raw text query to search the corpus"},
        TOP_K: {"explanation": f"Number of passages to return (default {DEFAULT_TOP_K})"},
        CORPUS: {"explanation": "Inline corpus: a list of {doc_id, text} dicts"},
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

        Gating on ``retrieve_backend`` (rather than name alone) is what keeps
        concrete backends mutually exclusive, so enabling several at once is
        unambiguous. An unknown backend matches nothing (honest surface: the
        connector does not silently claim a backend it cannot serve).
        """
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        backend = options.get(cls.RETRIEVE_BACKEND)
        return backend in cls.RETRIEVE_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features."""
        return None

    @classmethod
    def _get_top_k(cls, options: Options) -> int:
        val = options.get(cls.TOP_K)
        if val is None:
            return cls.DEFAULT_TOP_K
        try:
            return int(val)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{cls.__name__} option '{cls.TOP_K}' must be an integer, got {val!r}.") from exc

    @classmethod
    def _get_corpus(cls, options: Options) -> List[Dict[str, Any]]:
        corpus = options.get(cls.CORPUS)
        if corpus is None:
            raise ValueError(f"{cls.__name__} requires '{cls.CORPUS}' in options: a list of {{doc_id, text}} dicts.")
        return list(corpus)

    @classmethod
    @abstractmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        """Rank ``texts`` against ``query``.

        Returns at most ``top_k`` ``(corpus_index, score)`` pairs, ordered
        best-first, where ``score`` is higher-is-more-relevant. The unified
        family rule: only pairs with a positive score are returned, so a
        degenerate query (empty, or sharing no terms with the corpus) yields no
        pairs. Requirements the base relies on and enforces (see
        :meth:`_validate_ranking`): indices are in range
        (``0 <= corpus_index < len(texts)``) and unique, at most ``top_k``
        pairs come back, and scores are non-increasing. ``top_k`` is already
        clamped to ``1 <= top_k <= len(texts)``, so backends need not re-check
        it. The base does not re-sort, so returning best-first is a hard
        requirement.
        """
        ...

    @classmethod
    def _validate_ranking(cls, ranked: List[Tuple[int, float]], corpus_size: int, top_k: int) -> None:
        """Reject a ``_rank`` result that breaks the contract.

        Enforces all four :meth:`_rank` requirements: indices in range, indices
        unique, at most ``top_k`` pairs, and scores non-increasing (best-first).
        """
        if len(ranked) > top_k:
            raise ValueError(f"{cls.__name__}._rank returned {len(ranked)} pairs for top_k={top_k}.")
        seen: Set[int] = set()
        previous_score: Optional[float] = None
        for corpus_idx, score in ranked:
            if not 0 <= corpus_idx < corpus_size:
                raise ValueError(
                    f"{cls.__name__}._rank returned out-of-range index {corpus_idx} for a corpus of size {corpus_size}."
                )
            if corpus_idx in seen:
                raise ValueError(f"{cls.__name__}._rank returned duplicate index {corpus_idx}.")
            seen.add(corpus_idx)
            if previous_score is not None and score > previous_score:
                raise ValueError(
                    f"{cls.__name__}._rank returned scores out of order: {score} after {previous_score} "
                    f"(scores must be non-increasing, best-first)."
                )
            previous_score = score

    @classmethod
    def _retrieve(
        cls,
        query: str,
        corpus: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Assemble the ranked-passage contract around the backend's :meth:`_rank`.

        Owns the cross-backend invariants: empty corpus and non-positive
        ``top_k`` return ``[]``; ``top_k`` is clamped to the corpus size;
        ``doc_id``/``text`` are read from the corpus; ``rank`` is assigned
        0-based ascending; ``score`` is coerced to ``float``. Corpus entries
        must be dicts, and the effective ``doc_id`` values (after ``str()``
        coercion and the positional-index fallback for a missing ``doc_id``)
        must be unique; either violation raises ``ValueError``. A missing
        ``text`` key stays lenient and coerces to ``""``. The ranking a backend
        returns is validated against the :meth:`_rank` contract, so a buggy
        ``_rank`` fails loudly here (for any input) instead of silently
        dropping or duplicating a passage.
        """
        if not corpus:
            return []

        for i, doc in enumerate(corpus):
            if not isinstance(doc, dict):
                raise ValueError(
                    f"{cls.__name__} corpus entry at index {i} is not a dict: {doc!r}. "
                    f"Each entry must be a {{doc_id, text}} dict."
                )

        doc_ids = [str(doc.get("doc_id", str(i))) for i, doc in enumerate(corpus)]
        seen_doc_ids: Set[str] = set()
        for doc_id in doc_ids:
            if doc_id in seen_doc_ids:
                raise ValueError(
                    f"{cls.__name__} corpus contains duplicate doc_id {doc_id!r} "
                    f"(after str() coercion and the positional-index fallback)."
                )
            seen_doc_ids.add(doc_id)

        effective_k = min(top_k, len(corpus))
        if effective_k <= 0:
            return []

        texts = [str(doc.get("text", "")) for doc in corpus]

        ranked = cls._rank(query, texts, effective_k)
        cls._validate_ranking(ranked, len(corpus), effective_k)

        passages: List[Dict[str, Any]] = []
        for rank, (corpus_idx, score) in enumerate(ranked):
            passages.append(
                {
                    "doc_id": doc_ids[corpus_idx],
                    "text": texts[corpus_idx],
                    "score": float(score),
                    "rank": rank,
                }
            )
        return passages

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Rank the corpus against the query, return ranked passages.

        The FeatureSet must contain exactly one feature: the family answers one
        query per run, so a set with several features would silently drop all
        but the first and instead raises ``ValueError``.
        """
        feature_list = list(features.features)
        if len(feature_list) > 1:
            raise ValueError(
                f"{cls.__name__} answers one query per run, but the FeatureSet contains {len(feature_list)} features."
            )
        for feature in feature_list:
            options = feature.options
            query = options.get(cls.QUERY_TEXT)
            if query is None:
                raise ValueError(f"{cls.__name__} requires '{cls.QUERY_TEXT}' in options.")
            corpus = cls._get_corpus(options)
            top_k = cls._get_top_k(options)
            passages = cls._retrieve(str(query), corpus, top_k)
            return [{cls.ROOT_FEATURE_NAME: passages}]
        return []
