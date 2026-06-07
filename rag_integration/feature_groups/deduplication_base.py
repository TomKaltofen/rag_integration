"""Shared base class for row-level deduplication feature groups.

The text and image deduplication pipelines follow the same shape: extract a comparable
item from each row, detect duplicates, attach duplicate metadata, then filter by a keep
strategy. That scaffolding used to be copied between
``rag_pipeline/deduplication/base.py`` and ``image_pipeline/deduplication/base.py``.

``BaseRowDeduplicator`` owns the shared parts (option getters, the metadata + keep-strategy
loop, and group-representative selection). Subclasses provide the per-row item extraction
(``_extract_items``: text strings vs image bytes), the duplicate-detection algorithm
(``_find_duplicates``), and the config/PROPERTY_MAPPING that selects an implementation.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Type

from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)


class BaseRowDeduplicator(FeatureChainParserMixin, FeatureGroup):
    """Shared scaffolding for text and image deduplication feature groups."""

    # Configuration keys
    SIMILARITY_THRESHOLD = "similarity_threshold"
    KEEP_STRATEGY = "keep_strategy"

    # Keep-strategy value that keeps the largest item from each duplicate group.
    # Text uses "longest", image uses "largest"; subclasses set this accordingly.
    KEEP_LARGEST_STRATEGY = "longest"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def _get_source_feature_name(cls, feature: Feature) -> str:
        """Extract source feature name from the feature."""
        source_features = cls._extract_source_features(feature)
        return source_features[0]

    @classmethod
    def _get_similarity_threshold(cls, feature: Feature) -> float:
        """Get similarity threshold from feature options."""
        threshold = feature.options.get(cls.SIMILARITY_THRESHOLD)
        return float(threshold) if threshold is not None else 1.0

    @classmethod
    def _get_keep_strategy(cls, feature: Feature) -> str:
        """Get keep strategy from feature options."""
        strategy = feature.options.get(cls.KEEP_STRATEGY)
        return str(strategy) if strategy is not None else "first"

    @classmethod
    @abstractmethod
    def _extract_items(cls, data: List[Dict[str, Any]], feature: Feature) -> List[Any]:
        """Extract the comparable item (text string or image bytes) from each row."""
        ...

    @classmethod
    @abstractmethod
    def _find_duplicates(cls, items: List[Any], threshold: float) -> List[Optional[int]]:
        """Find duplicates among items.

        Returns a list where each element is either None (not a duplicate) or the index
        of the earlier item it duplicates.
        """
        ...

    @classmethod
    def _item_size(cls, item: Any) -> int:
        """Size used to choose the representative when keeping the largest per group."""
        return len(item)

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Deduplicate rows: attach duplicate metadata and filter by keep strategy."""
        for feature in features.features:
            threshold = cls._get_similarity_threshold(feature)
            keep_strategy = cls._get_keep_strategy(feature)
            feature_name = feature.name

            items = cls._extract_items(data, feature)
            duplicate_of = cls._find_duplicates(items, threshold)

            result = []
            for i, row in enumerate(data):
                new_row = row.copy()
                new_row["is_duplicate"] = duplicate_of[i] is not None
                new_row["duplicate_of"] = duplicate_of[i]
                new_row[feature_name] = items[i]

                if keep_strategy == "all_unique":
                    result.append(new_row)
                elif keep_strategy == "first" and duplicate_of[i] is None:
                    result.append(new_row)
                elif keep_strategy == cls.KEEP_LARGEST_STRATEGY:
                    result.append(new_row)

            if keep_strategy == cls.KEEP_LARGEST_STRATEGY:
                result = cls._keep_largest_per_group(result, items)

            return result

        return data

    @classmethod
    def _keep_largest_per_group(cls, data: List[Dict[str, Any]], items: List[Any]) -> List[Dict[str, Any]]:
        """Keep only the largest item from each duplicate group."""
        groups: Dict[int, List[int]] = {}
        for i, row in enumerate(data):
            dup_of = row.get("duplicate_of")
            key = dup_of if dup_of is not None else i
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        keep_indices = set()
        for indices in groups.values():
            largest_idx = max(indices, key=lambda idx: cls._item_size(items[idx]))
            keep_indices.add(largest_idx)

        return [data[i] for i in sorted(keep_indices)]
