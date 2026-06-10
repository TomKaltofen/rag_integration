"""Base class for the ``graph_rag`` connector family.

Contract: ``query_text + nodes + edges + top_k -> ranked passages``.

A graph-RAG connector retrieves passages over a graph: the corpus is a set of
text nodes plus edges between them, and each node is scored by its own query
overlap plus a one-hop neighbour bonus for adjacent relevant nodes. The
distinguishing value over plain retrieval is *connected context*: a passage
with no query-term overlap can still be surfaced because it neighbours a
relevant one.

It is a ROOT FeatureGroup: nodes and edges are passed inline through
``Options`` so the family is self-contained and contract-testable without a
graph database. ``nodes`` is a list of ``{doc_id, text}``; ``edges`` is a list
of ``[doc_id_a, doc_id_b]`` pairs. ``edges`` is optional: omitting it degrades
scoring to lexical-only (no neighbour bonus).

Output (single row, keyed by the root feature name)::

    {"graph_passages": [{"doc_id": ..., "text": ..., "score": ..., "rank": ...}, ...]}

The base owns option extraction, edge resolution (doc_id pairs -> node-index
pairs), clamping, validation of returned indices, and passage assembly. A
backend implements only :meth:`_rank`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class BaseGraphRagConnector(FeatureGroup):
    """Root FeatureGroup for graph-RAG connector backends.

    A concrete backend declares its selector value in ``GRAPH_BACKENDS`` and
    implements :meth:`_rank`; selection is via
    :meth:`match_feature_group_criteria`, gating on
    ``graph_backend in cls.GRAPH_BACKENDS``.
    """

    ROOT_FEATURE_NAME = "graph_passages"

    # Option keys.
    GRAPH_BACKEND = "graph_backend"
    QUERY_TEXT = "query_text"
    TOP_K = "top_k"
    NODES = "nodes"
    EDGES = "edges"

    DEFAULT_TOP_K = 5

    # Filled per concrete; empty on the base so it never matches.
    GRAPH_BACKENDS: Dict[str, str] = {}

    # Declarative option documentation only; selection is via
    # ``match_feature_group_criteria`` (not the FeatureChainParser).
    PROPERTY_MAPPING = {
        GRAPH_BACKEND: {"explanation": "Which graph-RAG backend to use"},
        QUERY_TEXT: {"explanation": "Raw text query to search the graph"},
        TOP_K: {"explanation": f"Number of passages to return (default {DEFAULT_TOP_K})"},
        NODES: {"explanation": "Graph nodes: a list of {doc_id, text} dicts"},
        EDGES: {
            "explanation": "Graph edges: a list of [doc_id_a, doc_id_b] pairs."
            " Optional: omitting it degrades scoring to lexical-only (no neighbour bonus)"
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
        """Match the root feature name only for a backend this concrete declares."""
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        backend = options.get(cls.GRAPH_BACKEND)
        return backend in cls.GRAPH_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features (graph arrives via Options)."""
        return None

    @classmethod
    def _get_top_k(cls, options: Options) -> int:
        val = options.get(cls.TOP_K)
        return int(val) if val is not None else cls.DEFAULT_TOP_K

    @classmethod
    def _get_nodes(cls, options: Options) -> List[Dict[str, Any]]:
        nodes = options.get(cls.NODES)
        if nodes is None:
            raise ValueError(f"{cls.__name__} requires '{cls.NODES}' in options: a list of {{doc_id, text}} dicts.")
        return list(nodes)

    @classmethod
    def _resolve_edges(cls, options: Options) -> List[Tuple[str, str]]:
        """Resolve the optional ``EDGES`` option into ``(doc_id_a, doc_id_b)`` pairs.

        ``EDGES`` is optional: omitting it degrades scoring to lexical-only (no
        neighbour bonus). When present it must be a list/tuple of
        ``[doc_id_a, doc_id_b]`` pairs; any other container raises ``ValueError``
        (a string would otherwise silently drop every edge). Malformed elements
        and self-loops are skipped (they carry no usable context).
        """
        raw_edges = options.get(cls.EDGES)
        if raw_edges is None:
            return []
        if not isinstance(raw_edges, (list, tuple)):
            raise ValueError(
                f"{cls.__name__} '{cls.EDGES}' must be a list of [doc_id_a, doc_id_b] pairs, "
                f"got {type(raw_edges).__name__}."
            )
        resolved: List[Tuple[str, str]] = []
        for edge in raw_edges:
            # A real pair only: a length-2 string would otherwise fabricate an
            # edge between its two characters, and a non-sequence would crash len().
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            a, b = str(edge[0]), str(edge[1])
            if a != b:
                resolved.append((a, b))
        return resolved

    @classmethod
    @abstractmethod
    def _rank(cls, query: str, texts: List[str], edges: List[Tuple[int, int]], top_k: int) -> List[Tuple[int, float]]:
        """Rank nodes against the query using graph structure.

        ``edges`` are node-index pairs (already resolved from doc_ids). Returns
        up to ``top_k`` ``(node_index, score)`` pairs, best-first; indices must
        be in range and unique (validated by the base). ``top_k`` is clamped to
        ``1 <= top_k <= len(texts)``. The base does not re-sort, so returning
        best-first is a hard requirement.
        """
        ...

    @classmethod
    def _validate_ranking(cls, ranked: List[Tuple[int, float]], n_nodes: int) -> None:
        """Reject out-of-range or duplicate indices from a backend's ``_rank``."""
        seen: Set[int] = set()
        for idx, _score in ranked:
            if not 0 <= idx < n_nodes:
                raise ValueError(f"{cls.__name__}._rank returned out-of-range index {idx} for {n_nodes} nodes.")
            if idx in seen:
                raise ValueError(f"{cls.__name__}._rank returned duplicate index {idx}.")
            seen.add(idx)

    @classmethod
    def _retrieve(
        cls,
        query: str,
        nodes: List[Dict[str, Any]],
        edges: List[Tuple[str, str]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Assemble the ranked-passage contract around the backend's :meth:`_rank`.

        Pure data in, passages out: ``edges`` are already-resolved doc_id pairs
        (see :meth:`_resolve_edges`). A duplicate doc_id (including distinct
        values colliding after ``str()`` coercion) raises ``ValueError``: edges
        could not be attributed unambiguously, and the earlier node would become
        an unreachable isolated node that is still scored. Edges naming a
        doc_id outside the corpus are skipped (no usable context).
        """
        if not nodes:
            return []
        effective_k = min(top_k, len(nodes))
        if effective_k <= 0:
            return []

        texts = [str(node.get("text", "")) for node in nodes]
        doc_ids = [str(node.get("doc_id", str(i))) for i, node in enumerate(nodes)]
        doc_id_to_index: Dict[str, int] = {}
        for i, doc_id in enumerate(doc_ids):
            if doc_id in doc_id_to_index:
                raise ValueError(f"{cls.__name__}: duplicate doc_id '{doc_id}': edges would be ambiguous.")
            doc_id_to_index[doc_id] = i
        edge_indices = [
            (doc_id_to_index[a], doc_id_to_index[b]) for a, b in edges if a in doc_id_to_index and b in doc_id_to_index
        ]

        ranked = cls._rank(query, texts, edge_indices, effective_k)
        cls._validate_ranking(ranked, len(nodes))

        passages: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(ranked):
            passages.append({"doc_id": doc_ids[idx], "text": texts[idx], "score": float(score), "rank": rank})
        return passages

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Score nodes by query overlap plus a one-hop neighbour bonus, return ranked passages."""
        for feature in features.features:
            options = feature.options
            query = options.get(cls.QUERY_TEXT)
            if query is None:
                raise ValueError(f"{cls.__name__} requires '{cls.QUERY_TEXT}' in options.")
            nodes = cls._get_nodes(options)
            edges = cls._resolve_edges(options)
            top_k = cls._get_top_k(options)
            passages = cls._retrieve(str(query), nodes, edges, top_k)
            return [{cls.ROOT_FEATURE_NAME: passages}]
        return []
