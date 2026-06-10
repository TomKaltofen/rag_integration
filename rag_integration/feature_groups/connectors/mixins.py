"""Cross-cutting property mixins shared by the connector-family bases.

PR #31 implemented these recurring concerns inline in every family's
``base.py``: parsing ``top_k``, pulling a ``{doc_id, text}`` collection out of
``Options``, computing effective ``doc_id``s and rejecting duplicates, and
validating the ``(index, score)`` pairs a backend returns. This module hoists
each concern into a focused mixin so a family base declares it once by
inheriting it, mirroring open-kgo's ``EntityFilter`` / ``Pagination`` /
``Traversal`` / ``Inference`` mixins.

The mixins are deliberately plain classes (not ``FeatureGroup`` subclasses): a
family base lists the mixins it needs ahead of ``FeatureGroup`` in its bases, so
mloda's plugin discovery still sees only the ``FeatureGroup`` leaves and the
shared helpers resolve through the MRO. They use ``cls.__name__`` in messages,
so a rejection still names the concrete backend, exactly as the inline code did.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from mloda.user import Options

from rag_integration.feature_groups.connectors.errors import (
    InvalidOptionError,
    MissingOptionError,
    RankingContractError,
)


class OptionsMixin:
    """Read required scalar values out of a feature's ``Options``."""

    @classmethod
    def _require_option(cls, options: Options, key: str) -> Any:
        """Return ``options[key]``, or raise :class:`MissingOptionError` if absent (``None``)."""
        value = options.get(key)
        if value is None:
            raise MissingOptionError(f"{cls.__name__} requires '{key}' in options.")
        return value


class TopKMixin:
    """The ``top_k`` cut-off shared by retrieve, rerank, graph_rag, orchestrator."""

    TOP_K = "top_k"
    DEFAULT_TOP_K = 5

    @classmethod
    def _get_top_k(cls, options: Options) -> int:
        """Parse the ``top_k`` option, defaulting when absent.

        A present-but-non-integer value is a caller error, reported naming the
        key and the offending value rather than surfacing a raw ``int()``
        failure.
        """
        val = options.get(cls.TOP_K)
        if val is None:
            return cls.DEFAULT_TOP_K
        try:
            return int(val)
        except (ValueError, TypeError) as exc:
            raise InvalidOptionError(f"{cls.__name__} option '{cls.TOP_K}' must be an integer, got {val!r}.") from exc


class DocCollectionMixin:
    """A ``{doc_id, text}`` collection and its ``doc_id`` bookkeeping.

    Covers the corpus / candidates / nodes / passages every non-structured
    family pulls from ``Options``, the effective-``doc_id`` rule (explicit
    ``doc_id`` coerced to ``str``, falling back to the positional index), and the
    duplicate detection and known-id set the grounding and edge guards build on.
    """

    @staticmethod
    def _effective_doc_id(item: Dict[str, Any], index: int) -> str:
        """The ``doc_id`` an entry is keyed by: its explicit value, else its index."""
        return str(item.get("doc_id", str(index)))

    @classmethod
    def _effective_doc_ids(cls, items: Sequence[Dict[str, Any]]) -> List[str]:
        """Effective ``doc_id`` for every entry, positionally aligned with ``items``."""
        return [cls._effective_doc_id(item, i) for i, item in enumerate(items)]

    @classmethod
    def _known_doc_ids(cls, items: Sequence[Dict[str, Any]]) -> Set[str]:
        """The set of effective ``doc_id``s, for membership checks."""
        return set(cls._effective_doc_ids(items))

    @classmethod
    def _find_duplicate_doc_id(cls, items: Sequence[Dict[str, Any]]) -> Optional[str]:
        """Return the first effective ``doc_id`` that repeats, or ``None``.

        Centralises the seen-set scan; the caller raises
        :class:`~rag_integration.feature_groups.connectors.errors.DuplicateDocIdError`
        with its own family-specific rationale, the one part of the check that
        legitimately differs per family.
        """
        seen: Set[str] = set()
        for i, item in enumerate(items):
            doc_id = cls._effective_doc_id(item, i)
            if doc_id in seen:
                return doc_id
            seen.add(doc_id)
        return None

    @classmethod
    def _require_doc_list(cls, options: Options, key: str) -> List[Dict[str, Any]]:
        """Return the ``{doc_id, text}`` list under ``key``, or raise if absent."""
        value = options.get(key)
        if value is None:
            raise MissingOptionError(f"{cls.__name__} requires '{key}' in options: a list of {{doc_id, text}} dicts.")
        return list(value)


class RankingValidationMixin:
    """Validate the ``(index, score)`` pairs a backend ``_rank`` returns."""

    @classmethod
    def _validate_rank_indices(
        cls,
        ranked: List[Tuple[int, float]],
        count: int,
        extent: str,
        *,
        non_increasing: bool = False,
    ) -> None:
        """Reject out-of-range or duplicate indices (and, optionally, mis-ordered scores).

        ``count`` is the number of rankable items and ``extent`` is the
        human-readable description of that population used in the out-of-range
        message (e.g. ``"3 candidates"``). With ``non_increasing=True`` the same
        single pass also rejects a score that rises after a lower one, so a
        family that promises best-first output validates ordering without a
        second loop.
        """
        seen: Set[int] = set()
        previous_score: Optional[float] = None
        for idx, score in ranked:
            if not 0 <= idx < count:
                raise RankingContractError(f"{cls.__name__}._rank returned out-of-range index {idx} for {extent}.")
            if idx in seen:
                raise RankingContractError(f"{cls.__name__}._rank returned duplicate index {idx}.")
            seen.add(idx)
            if non_increasing:
                if previous_score is not None and score > previous_score:
                    raise RankingContractError(
                        f"{cls.__name__}._rank returned scores out of order: {score} after {previous_score} "
                        f"(scores must be non-increasing, best-first)."
                    )
                previous_score = score
