"""The swap-backends demo is the runnable proof for issue #34, item 1.

Every swap below goes through the demo's single :func:`run_connector` helper, so
the test pins the actual claim: swapping a backend is a change to the options
dict, not to the calling code. Within a family only the ``<family>_backend``
selector changes; across families only the selector key and the root feature
name change, over identical query/corpus/top_k inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from cli.swap_demo import CORPUS, QUERY, SHARED_INPUTS, TOP_K, run_connector
from rag_integration.feature_groups.connectors.generate import ExtractiveResponder, TemplateResponder
from rag_integration.feature_groups.connectors.orchestrator import HaystackOrchestrator
from rag_integration.feature_groups.connectors.retrieve import Bm25sRetriever, TfidfRetriever

PASSAGES: List[Dict[str, str]] = [{"doc_id": doc["doc_id"], "text": doc["text"]} for doc in CORPUS]


def _assert_passage_shape(passages: List[Dict[str, Any]]) -> None:
    assert passages, "expected a non-empty ranking for the shape assertions to mean anything"
    for rank, passage in enumerate(passages):
        assert set(passage) == {"doc_id", "text", "score", "rank"}
        assert passage["rank"] == rank
    scores = [p["score"] for p in passages]
    assert scores == sorted(scores, reverse=True)


class TestWithinFamilyRetrieveSwap:
    """retrieve_backend="bm25s" -> "tfidf": same call shape, only the selector value moves."""

    def test_only_the_selector_value_differs_between_the_two_runs(self) -> None:
        bm25s_passages = run_connector(
            "retrieved_passages", {"retrieve_backend": "bm25s", **SHARED_INPUTS}, {Bm25sRetriever}
        )
        tfidf_passages = run_connector(
            "retrieved_passages", {"retrieve_backend": "tfidf", **SHARED_INPUTS}, {TfidfRetriever}
        )

        _assert_passage_shape(bm25s_passages)
        _assert_passage_shape(tfidf_passages)
        # Two different lexical mechanisms, but the same relevant documents in the
        # same order: the swap is safe, the downstream contract is unchanged.
        assert [p["doc_id"] for p in bm25s_passages] == ["d2", "d1"]
        assert [p["doc_id"] for p in tfidf_passages] == ["d2", "d1"]


class TestWithinFamilyGenerateSwap:
    """generate_backend="extractive" -> "template": same call shape, grounded both ways."""

    def test_both_backends_answer_under_one_contract(self) -> None:
        extractive = run_connector(
            "generated_answer",
            {"generate_backend": "extractive", "query_text": QUERY, "passages": PASSAGES},
            {ExtractiveResponder},
        )
        template = run_connector(
            "generated_answer",
            {"generate_backend": "template", "query_text": QUERY, "passages": PASSAGES},
            {TemplateResponder},
        )

        for answer in (extractive, template):
            assert set(answer) == {"answer", "citations"}
            assert answer["answer"], "a no-LLM responder still answers when the corpus is on-topic"
            # Grounded by construction: every citation is one of the supplied passages.
            known = {p["doc_id"] for p in PASSAGES}
            assert answer["citations"], "a non-empty answer must cite its source"
            assert all(c in known for c in answer["citations"])


class TestAcrossFamilySwap:
    """retrieve <-> orchestrator: identical inputs, only the selector key and root name change."""

    def test_the_two_families_share_their_inputs_verbatim(self) -> None:
        retrieve_options = {"retrieve_backend": "tfidf", **SHARED_INPUTS}
        orchestrator_options = {"orchestrator_backend": "haystack", **SHARED_INPUTS}

        # The only differences are the selector key and (downstream) the root
        # feature name; query/corpus/top_k are the same objects on both sides.
        shared_keys = {"query_text", "corpus", "top_k"}
        for key in shared_keys:
            assert retrieve_options[key] == orchestrator_options[key]
        assert set(retrieve_options) - shared_keys == {"retrieve_backend"}
        assert set(orchestrator_options) - shared_keys == {"orchestrator_backend"}

    def test_each_family_returns_its_own_shape_from_the_same_inputs(self) -> None:
        passages = run_connector("retrieved_passages", {"retrieve_backend": "tfidf", **SHARED_INPUTS}, {TfidfRetriever})
        answer = run_connector(
            "orchestrated_answer", {"orchestrator_backend": "haystack", **SHARED_INPUTS}, {HaystackOrchestrator}
        )

        _assert_passage_shape(passages)
        assert set(answer) == {"answer", "documents"}
        assert answer["answer"], "the orchestrator pipeline answers an on-topic query"
        corpus_ids = {doc["doc_id"] for doc in CORPUS}
        assert all(doc["doc_id"] in corpus_ids for doc in answer["documents"])


def test_top_k_constant_is_within_corpus_size() -> None:
    """Guard the demo's own fixture so the swaps above stay non-degenerate."""
    assert 0 < TOP_K <= len(CORPUS)
