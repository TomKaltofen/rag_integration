"""R2R fixture-stub orchestrator backend.

Second concrete for the ``orchestrator`` family, and a different *integration
mode* from the in-process ``HaystackOrchestrator``: it models a server-shaped
RAG tool (R2R) over a static JSON fixture instead of running a library
in-process. There is no server and no network: the fixture holds canned R2R
``/rag``-style responses (a generated answer plus ranked source doc_ids), keyed
by query, exactly as the open-kgo ``rest_public`` file-fixture connectors model
a REST API from local files.

The honest-surface mechanism is **narrowing**: the corpus passed to the family
is treated as the documents ingested into R2R, and the stub surfaces only the
canned doc_ids that are actually in that corpus (with the corpus's own text), so
nothing is fabricated. A query with no canned response yields ``("", [])`` (the
server has nothing indexed for it). The canned answer is surfaced only when the
document it is drawn from (``answer_doc_id``) survives narrowing; otherwise the
result is retrieve-only (the surviving documents, an empty answer), so the
answer never rests on documents it was not drawn from.

Zero-download, zero-dependency (stdlib ``json``), deterministic; a CI anchor
alongside the Haystack backend.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "r2r_responses.json"


class R2RFixtureOrchestrator(BaseOrchestratorConnector):
    """R2R-shaped fixture-stub orchestrator (``orchestrator_backend="r2r"``).

    Answers from a bundled JSON fixture of canned R2R responses, narrowed to the
    supplied corpus. The fixture is loaded once and cached at class level; the
    read is deterministic, so repeated calls are idempotent.
    """

    ORCHESTRATOR_BACKENDS = {
        "r2r": "R2R-shaped server stub over a static JSON fixture (honest-surface narrowing)",
    }

    PROPERTY_MAPPING = {
        BaseOrchestratorConnector.ORCHESTRATOR_BACKEND: {"explanation": "Use 'r2r' for the R2R fixture-stub pipeline"},
        BaseOrchestratorConnector.QUERY_TEXT: {"explanation": "The query to look up in the canned R2R responses"},
        BaseOrchestratorConnector.TOP_K: {
            "explanation": f"Number of documents to surface (default {BaseOrchestratorConnector.DEFAULT_TOP_K})"
        },
        BaseOrchestratorConnector.CORPUS: {"explanation": "Inline corpus (the documents ingested into R2R)"},
    }

    _responses: Dict[str, Any] | None = None
    _cache_lock = threading.Lock()

    @classmethod
    def _get_responses(cls) -> Dict[str, Any]:
        """Load and cache the canned-response table from the bundled fixture.

        The returned table is the shared cache and must be treated as read-only;
        ``_run`` only reads from it and emits fresh dicts for surfaced documents,
        so the cache is never mutated through a result.
        """
        responses = cls._responses
        if responses is not None:
            return responses
        with cls._cache_lock:
            if cls._responses is None:
                with _FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
                    payload = json.load(fixture_file)
                cls._responses = dict(payload.get("responses", {}))
            return cls._responses

    @classmethod
    def _run(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
        effective_k = min(top_k, len(corpus))
        if not query.strip() or effective_k <= 0:
            return "", []

        response = cls._get_responses().get(query.strip().lower())
        if response is None:
            # The server has no canned response for this query (honest surface).
            return "", []

        text_by_doc_id = {str(doc.get("doc_id", str(i))): str(doc.get("text", "")) for i, doc in enumerate(corpus)}

        # Narrowing: keep only canned doc_ids that are in the ingested corpus,
        # surfacing the corpus's own text (never the fixture's), so a surfaced
        # document is always grounded in what was actually supplied.
        documents: List[Dict[str, Any]] = []
        for entry in response.get("documents", []):
            doc_id = str(entry.get("doc_id"))
            if doc_id in text_by_doc_id:
                documents.append(
                    {"doc_id": doc_id, "text": text_by_doc_id[doc_id], "score": float(entry.get("score", 0.0))}
                )
            if len(documents) >= effective_k:
                break

        # If narrowing removed every document, there is nothing to ground an
        # answer on, so return an empty result rather than an ungrounded answer.
        if not documents:
            return "", []

        # The canned answer is drawn from one source document (answer_doc_id).
        # Surface the answer only if that document survived narrowing; otherwise
        # the answer's support is not in the corpus, so return a retrieve-only
        # result (the surviving documents, no answer) rather than an answer
        # grounded on documents it was not drawn from.
        answer_doc_id = str(response.get("answer_doc_id", ""))
        surfaced_ids = {document["doc_id"] for document in documents}
        answer = str(response.get("answer", "")) if answer_doc_id in surfaced_ids else ""
        return answer, documents
