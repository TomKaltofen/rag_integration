"""Repo-wide invariant: every PROPERTY_MAPPING is a valid spec mapping whose defaults are accepted values.

mloda rejects a bad strict default when the ``property_spec`` is built and a non-spec value at
class definition; this sweep makes sure every feature_groups module is imported so none is skipped.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

import pytest
from mloda.provider import FeatureChainParser, FeatureGroup, property_spec

import rag_integration.feature_groups


def _validate(owner_name: str, property_mapping: Optional[Dict[str, Any]]) -> List[str]:
    """Run the upstream class-definition validator; returns violation messages."""
    try:
        FeatureChainParser.validate_property_mapping_defaults(owner_name, property_mapping)
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []


def _all_feature_groups() -> List[Type[FeatureGroup]]:
    """Import every feature_groups module and collect this package's FeatureGroup subclasses.

    Any import failure fails the test: a module that cannot import is a module whose
    PROPERTY_MAPPING this invariant silently skips.
    """
    import_failures: List[str] = []
    for module_info in pkgutil.walk_packages(
        rag_integration.feature_groups.__path__,
        prefix="rag_integration.feature_groups.",
        onerror=lambda name: import_failures.append(f"{name}: failed during package walk"),
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:
            import_failures.append(f"{module_info.name}: {exc!r}")
    assert not import_failures, "feature_groups modules failed to import:\n" + "\n".join(import_failures)

    collected: List[Type[FeatureGroup]] = []
    stack: List[Type[FeatureGroup]] = list(FeatureGroup.__subclasses__())
    seen: set[Type[FeatureGroup]] = set()
    while stack:
        candidate = stack.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        stack.extend(candidate.__subclasses__())
        if candidate.__module__.startswith("rag_integration."):
            collected.append(candidate)
    return sorted(collected, key=lambda c: f"{c.__module__}.{c.__name__}")


def test_spec_rejects_default_outside_allowed_values() -> None:
    """Guard against a vacuously green check: a default outside the accepted values must be rejected."""
    with pytest.raises(ValueError, match="turbo"):

        class _BadDefault(FeatureGroup):
            PROPERTY_MAPPING = {
                "mode": property_spec(
                    "Mode", allowed_values={"fast": "fast mode", "slow": "slow mode"}, strict=True, default="turbo"
                ),
            }


def test_spec_rejects_default_failing_element_validator() -> None:
    with pytest.raises(ValueError, match="element_validator"):

        class _BadValidatedDefault(FeatureGroup):
            PROPERTY_MAPPING = {
                "size": property_spec(
                    "Positive size",
                    strict=True,
                    element_validator=lambda value: isinstance(value, int) and value > 0,
                    default=-1,
                ),
            }


def test_validator_catches_dict_spec() -> None:
    assert _validate("DummyOwner", {"mode": {"explanation": "dict-form spec"}})


def test_all_property_mapping_defaults_are_accepted_values() -> None:
    feature_groups = _all_feature_groups()
    # 74 feature groups exist today; lower this only when groups are deliberately removed.
    assert len(feature_groups) >= 74, f"feature group discovery looks broken, found {len(feature_groups)}"

    violations: List[str] = []
    for feature_group in feature_groups:
        owner = f"{feature_group.__module__}.{feature_group.__name__}"
        violations.extend(_validate(owner, feature_group.PROPERTY_MAPPING))

    assert not violations, "PROPERTY_MAPPING violations:\n" + "\n".join(violations)
