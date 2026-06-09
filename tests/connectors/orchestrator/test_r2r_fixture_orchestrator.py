"""Contract test for :class:`R2RFixtureOrchestrator` (zero-download CI anchor).

Runs entirely from a bundled JSON fixture (no server, no network), so it runs
on CI. Beyond the inherited contract, it adds the backend's own not-a-stub
proof: the honest-surface *narrowing* mechanism, where a canned response
document that is not in the supplied corpus is filtered out rather than
surfaced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.connectors.orchestrator.r2r_fixture_orchestrator import R2RFixtureOrchestrator
from tests.connectors.orchestrator.orchestrator_contract import OrchestratorConnectorContractBase


class TestR2RFixtureOrchestrator(OrchestratorConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseOrchestratorConnector]:
        return R2RFixtureOrchestrator

    @classmethod
    def backend_value(cls) -> str:
        return "r2r"

    @classmethod
    def sample_corpus(cls) -> List[Dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d1", "text": "A cat is an independent and curious pet."},
            {"doc_id": "d2", "text": "Dogs are loyal and energetic companions."},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "cat pet"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "d1"

    @classmethod
    def expected_answer_substring(cls) -> str:
        return "curious pet"

    # -- Backend-specific proof: honest-surface narrowing ---------------------

    def test_narrows_canned_docs_to_corpus(self) -> None:
        """The fixture's canned response for this query also ranks ``d2``, but a
        corpus that omits ``d2`` must not surface it: the stub narrows to the
        ingested corpus rather than echoing the fixture verbatim."""
        corpus = [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d1", "text": "A cat is an independent and curious pet."},
        ]
        result = self._answer(self.sample_query(), corpus, top_k=len(corpus))
        surfaced = {document["doc_id"] for document in result["documents"]}
        assert "d2" not in surfaced, "narrowing failed: surfaced a canned doc not in the corpus"
        assert surfaced == {"d0", "d1"}
        # The relevant doc survives narrowing, so the answer stays grounded.
        assert result["documents"][0]["doc_id"] == "d1"
        assert self.expected_answer_substring() in result["answer"]

    def test_answer_dropped_when_supporting_doc_narrowed_away(self) -> None:
        """If the document the answer is drawn from (``d1``) is not in the corpus,
        the answer is dropped even when other canned docs survive: the result is
        retrieve-only, never an answer grounded on documents it did not come from."""
        corpus = [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d2", "text": "Dogs are loyal and energetic companions."},
        ]
        result = self._answer(self.sample_query(), corpus, top_k=len(corpus))
        surfaced = {document["doc_id"] for document in result["documents"]}
        assert "d1" not in surfaced
        assert result["documents"], "surviving canned docs should still be surfaced (retrieve-only)"
        assert result["answer"] == "", "answer must be dropped when its supporting doc is narrowed away"

    def test_unknown_query_has_no_canned_response(self) -> None:
        """A query absent from the fixture yields an empty result (the server has
        nothing indexed for it), never a fabricated answer."""
        result = self._answer("a query the fixture has never seen", self.sample_corpus(), top_k=3)
        assert result == {"answer": "", "documents": []}
