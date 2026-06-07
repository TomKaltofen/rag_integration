"""Exact hash-based image deduplication."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

from rag_integration.feature_groups.image_pipeline.deduplication.base import BaseImageDeduplicator


class ExactHashImageDeduplicator(BaseImageDeduplicator):
    """
    Exact hash-based image deduplicator.

    Uses MD5 hashing of raw image bytes for fast exact duplicate detection.
    Only detects byte-identical images (ignores similarity_threshold).

    Config-based matching:
        image_deduplication_method="exact_hash"
    """

    PROPERTY_MAPPING = BaseImageDeduplicator.build_property_mapping(
        "exact_hash", "MD5 hash-based exact duplicate detection"
    )

    @classmethod
    def _find_duplicates(
        cls,
        image_data_list: List[bytes],
        threshold: float,
    ) -> List[Optional[int]]:
        """
        Find exact duplicates using MD5 hashing.

        Args:
            image_data_list: List of image bytes
            threshold: Ignored for exact hash matching

        Returns:
            List where each element is either None (first occurrence)
            or the index of the first occurrence it duplicates.
        """
        hash_to_index: Dict[str, int] = {}
        result: List[Optional[int]] = []

        for i, image_data in enumerate(image_data_list):
            data_hash = hashlib.md5(image_data, usedforsecurity=False).hexdigest()

            if data_hash in hash_to_index:
                result.append(hash_to_index[data_hash])
            else:
                hash_to_index[data_hash] = i
                result.append(None)

        return result
