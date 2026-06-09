"""Base class for the ``generate`` connector family.

Contract: ``query_text + passages -> answer + citations``.

A generate connector takes a query and supporting passages (e.g. from the
retrieve or rerank families) and produces a grounded answer plus the passage
ids it drew from. It is a ROOT FeatureGroup here: passages are passed inline
through ``Options`` so the family is self-contained and contract-testable
without a network or an LLM.

Output (single row, keyed by the root feature name)::

    {"generated_answer": {"answer": "...", "citations": ["doc_id", ...]}}

The canonical concrete is deterministic and offline. LLM-backed generators are
pedigree backends that belong behind their own extra. The contract enforces
that the answer is *grounded*: every citation is one of the supplied passages.

This mirrors the retrieve/rerank families (selector-gated matching, a single
abstract hook, single-row output) but the output is an answer object, not a
ranked-passage list, so it copies the pattern rather than subclassing them.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class BaseGenerateConnector(FeatureGroup):
    """Root FeatureGroup for generate-connector backends.

    A concrete backend declares its selector value in ``GENERATE_BACKENDS`` and
    implements :meth:`_generate`; the base owns option extraction, the
    single-row assembly, and validation that every returned citation is one of
    the supplied passages (no hallucinated sources). Selection is via
    :meth:`match_feature_group_criteria`, gating on
    ``generate_backend in cls.GENERATE_BACKENDS``.
    """

    ROOT_FEATURE_NAME = "generated_answer"

    # Option keys.
    GENERATE_BACKEND = "generate_backend"
    QUERY_TEXT = "query_text"
    PASSAGES = "passages"

    # Filled per concrete; empty on the base so it never matches.
    GENERATE_BACKENDS: Dict[str, str] = {}

    # Declarative option documentation only; selection is via
    # ``match_feature_group_criteria`` (not the FeatureChainParser).
    PROPERTY_MAPPING = {
        GENERATE_BACKEND: {"explanation": "Which generate-connector backend to use"},
        QUERY_TEXT: {"explanation": "The question to answer"},
        PASSAGES: {"explanation": "Supporting passages: a list of {doc_id, text} dicts"},
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
        """Match the root feature name only for a backend this concrete declares."""
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        backend = options.get(cls.GENERATE_BACKEND)
        return backend in cls.GENERATE_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features (passages arrive via Options)."""
        return None

    @classmethod
    def _get_passages(cls, options: Options) -> List[Dict[str, Any]]:
        passages = options.get(cls.PASSAGES)
        if passages is None:
            raise ValueError(f"{cls.__name__} requires '{cls.PASSAGES}' in options: a list of {{doc_id, text}} dicts.")
        return list(passages)

    @classmethod
    @abstractmethod
    def _generate(cls, query: str, passages: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Answer ``query`` from ``passages``.

        Returns ``(answer, citations)`` where ``answer`` is the answer text and
        ``citations`` is the list of ``doc_id``s the answer draws from. Each
        citation must be the ``doc_id`` of one of the supplied passages (the
        base validates this). Empty passages yield ``("", [])``.
        """
        ...

    @classmethod
    def _validate_citations(cls, citations: List[str], passages: List[Dict[str, Any]]) -> None:
        """Reject any citation that is not one of the supplied passage doc_ids."""
        known = {str(p.get("doc_id", str(i))) for i, p in enumerate(passages)}
        for citation in citations:
            if citation not in known:
                raise ValueError(
                    f"{cls.__name__}._generate cited '{citation}', which is not among the supplied passages."
                )

    @classmethod
    def _answer(cls, query: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble the answer contract around the backend's :meth:`_generate`."""
        if not passages:
            return {"answer": "", "citations": []}
        answer, citations = cls._generate(query, passages)
        cls._validate_citations(citations, passages)
        return {"answer": answer, "citations": citations}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Generate an answer from the passages, return the answer object."""
        for feature in features.features:
            options = feature.options
            query = options.get(cls.QUERY_TEXT)
            if query is None:
                raise ValueError(f"{cls.__name__} requires '{cls.QUERY_TEXT}' in options.")
            passages = cls._get_passages(options)
            return [{cls.ROOT_FEATURE_NAME: cls._answer(str(query), passages)}]
        return []
