"""Base class for the ``retrieve`` connector family.

Contract: ``query_text + corpus + top_k -> ranked passages with scores``.

A retrieve connector is a ROOT FeatureGroup (no input features): it takes an
inline corpus and a query through ``Options`` and returns the passages ranked
best-first. Concrete backends (lexical, dense, hybrid, late-interaction) differ
only in the ranking they apply behind this one contract; they are selected by
the ``retrieve_backend`` discriminator.

Output (single row, keyed by the root feature name)::

    {"retrieved_passages": [{"doc_id": ..., "text": ..., "score": ..., "rank": ...}, ...]}

``score`` is higher-is-more-relevant; ``rank`` is 0-based, ascending, best
first. ``PythonDictFramework`` slices the result to the requested feature, so
the ranked-passage list is the whole contract.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import DefaultOptionKeys
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class BaseRetrieveConnector(FeatureGroup):
    """Root FeatureGroup for retrieve-connector backends.

    Concrete backends override ``RETRIEVE_BACKENDS`` (and narrow the
    ``retrieve_backend`` enum to their own value with ``strict_validation``),
    then implement :meth:`_retrieve`. Selection is unambiguous: a concrete
    matches the root feature name only when ``retrieve_backend`` is one of the
    values it declares, so two backends never both claim the same options.
    """

    ROOT_FEATURE_NAME = "retrieved_passages"

    # Option keys.
    RETRIEVE_BACKEND = "retrieve_backend"
    QUERY_TEXT = "query_text"
    TOP_K = "top_k"
    CORPUS = "corpus"

    DEFAULT_TOP_K = 5

    # Filled per concrete: {backend_value: human-readable description}. The base
    # stays empty so it never matches a feature.
    RETRIEVE_BACKENDS: Dict[str, str] = {}

    PROPERTY_MAPPING = {
        RETRIEVE_BACKEND: {
            "explanation": "Which retrieve-connector backend to use",
            DefaultOptionKeys.context: True,
        },
        QUERY_TEXT: {
            "explanation": "Raw text query to search the corpus",
            DefaultOptionKeys.context: True,
        },
        TOP_K: {
            "explanation": "Number of passages to return",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: DEFAULT_TOP_K,
        },
        CORPUS: {
            "explanation": "Inline corpus: a list of {doc_id, text} dicts",
            DefaultOptionKeys.context: True,
        },
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
        return int(val) if val is not None else cls.DEFAULT_TOP_K

    @classmethod
    def _get_corpus(cls, options: Options) -> List[Dict[str, Any]]:
        corpus = options.get(cls.CORPUS)
        if corpus is None:
            raise ValueError(f"{cls.__name__} requires '{cls.CORPUS}' in options: a list of {{doc_id, text}} dicts.")
        return list(corpus)

    @classmethod
    @abstractmethod
    def _retrieve(
        cls,
        query: str,
        corpus: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Rank the corpus against the query.

        Returns at most ``top_k`` passages, best first, each a dict with
        ``doc_id``, ``text``, ``score`` (higher is more relevant) and ``rank``
        (0-based, ascending). An empty corpus returns an empty list.
        """
        ...

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Embed the query if needed, rank the corpus, return ranked passages."""
        for feature in features.features:
            options = feature.options
            query = options.get(cls.QUERY_TEXT)
            if query is None:
                raise ValueError(f"{cls.__name__} requires '{cls.QUERY_TEXT}' in options.")
            corpus = cls._get_corpus(options)
            top_k = cls._get_top_k(options)
            passages = cls._retrieve(str(query), corpus, top_k)
            return [{cls.ROOT_FEATURE_NAME: passages}]
        return []
