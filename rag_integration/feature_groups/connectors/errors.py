"""Centralised error types for the connector families.

PR #31 raised these validations inline in each family's ``base.py`` as bare
``ValueError``s, so the same rejection (a duplicate ``doc_id``, a backend that
breaks the ranking contract, a missing option) carried an ad-hoc, untyped shape
in every family. This module declares the shared hierarchy once.

Every connector error subclasses :class:`ConnectorError`, which subclasses the
built-in ``ValueError``: callers (and the contract tests) that catch
``ValueError`` keep working unchanged, while code that wants to distinguish a
connector-level rejection can catch the narrower type. The error *messages*
stay where they are raised, because their wording is the one part that
legitimately differs per family (a duplicate ``doc_id`` makes graph edges
ambiguous, but a corpus merely non-unique); only the *type* is centralised here.
"""

from __future__ import annotations


class ConnectorError(ValueError):
    """Base class for every connector-family validation / rejection error."""


class MissingOptionError(ConnectorError):
    """A required option is absent from the feature's ``Options``."""


class InvalidOptionError(ConnectorError):
    """An option is present but has the wrong type or an unusable value."""


class DuplicateDocIdError(ConnectorError):
    """Two corpus / passage / node entries share an effective ``doc_id``."""


class RankingContractError(ConnectorError):
    """A backend ``_rank`` result violates the base's ranking contract."""


class GroundingError(ConnectorError):
    """A generated answer cites or surfaces something not in the supplied input."""


class SqlSafetyError(ConnectorError):
    """Backend-produced SQL is unsafe or not a single bare ``SELECT``."""
