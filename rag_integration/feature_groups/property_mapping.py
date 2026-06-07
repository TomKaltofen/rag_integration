"""Shared helper for building config-based FeatureGroup PROPERTY_MAPPING dicts.

Most feature groups in this package follow the same shape: a discriminator option
(e.g. ``chunking_method``) selects one implementation, and every implementation in a
family shares the same set of additional option entries. Without help, each concrete
implementation has to repeat the entire mapping just to change the single discriminator
value.

``ConfigPropertyMappingMixin`` lets a base class declare the shared entries once
(``_SHARED_PROPERTY_MAPPING``) plus the discriminator key (``_DISCRIMINATOR_KEY``). Each
implementation then builds its mapping with a single call::

    PROPERTY_MAPPING = BaseChunker.build_property_mapping("sentence", "Sentence chunks")

``extra`` adds or overrides entries (algorithm-specific options or tweaked
explanations); ``exclude`` drops shared entries an implementation does not support.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from mloda.provider import DefaultOptionKeys


class ConfigPropertyMappingMixin:
    """Build a PROPERTY_MAPPING from shared entries plus a single discriminator value.

    Subclasses must define:
        _DISCRIMINATOR_KEY: the option key that selects the implementation.
        _SHARED_PROPERTY_MAPPING: option entries common to every implementation.
    """

    _DISCRIMINATOR_KEY: str
    _SHARED_PROPERTY_MAPPING: Dict[str, Any]

    @classmethod
    def build_property_mapping(
        cls,
        method: str,
        description: str,
        extra: Optional[Dict[str, Any]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Return a PROPERTY_MAPPING for one implementation.

        Args:
            method: The discriminator value this implementation handles (e.g. "sentence").
            description: Human-readable description of the method.
            extra: Entries to add or override on top of the shared mapping.
            exclude: Shared entry keys to drop (for implementations that do not use them).
        """
        mapping: Dict[str, Any] = {
            cls._DISCRIMINATOR_KEY: {
                method: description,
                DefaultOptionKeys.context: True,
                DefaultOptionKeys.strict_validation: True,
            },
            **cls._SHARED_PROPERTY_MAPPING,
        }
        for key in exclude or ():
            mapping.pop(key, None)
        if extra:
            mapping.update(extra)
        return mapping
