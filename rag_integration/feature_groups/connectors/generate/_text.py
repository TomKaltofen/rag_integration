"""Shared text helpers for the ``generate`` family's no-LLM responders.

Both deterministic baselines (extractive and template) tokenize and
sentence-split the same way; this private module holds the single copy so the
two backends cannot drift apart. The helpers are intentionally tiny: an
ASCII-lowercase token set and a punctuation-based sentence splitter.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
"""Punctuation-based sentence splitter (over-splits abbreviations)."""


def tokenize(text: str) -> set[str]:
    """Return the set of distinct lowercase ``[a-z0-9]+`` tokens in ``text``."""
    return set(_TOKEN_RE.findall(text.lower()))
