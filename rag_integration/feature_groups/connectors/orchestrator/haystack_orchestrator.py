"""Haystack orchestrator backend.

Canonical concrete for the ``orchestrator`` family: runs a real Haystack 2.x
pipeline (``InMemoryDocumentStore`` + ``InMemoryBM25Retriever``) entirely
in-memory. Zero-download (BM25 needs no model and no API) and deterministic, so
it anchors the CI contract suite while exercising a genuine external framework.
Behind the ``orchestrator`` extra.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector


class HaystackOrchestrator(BaseOrchestratorConnector):
    """Whole-pipeline retrieval via Haystack (``orchestrator_backend="haystack"``).

    Builds an in-memory document store, writes the corpus, and runs a BM25
    retrieval pipeline. The answer (no LLM) is the top document's content; the
    surfaced documents carry the pipeline's BM25 scores.
    """

    ORCHESTRATOR_BACKENDS = {
        "haystack": "Haystack 2.x in-memory BM25 pipeline",
    }

    PROPERTY_MAPPING = {
        BaseOrchestratorConnector.ORCHESTRATOR_BACKEND: {"explanation": "Use 'haystack' for a Haystack BM25 pipeline"},
        BaseOrchestratorConnector.QUERY_TEXT: {"explanation": "The query to run through the pipeline"},
        BaseOrchestratorConnector.TOP_K: {
            "explanation": f"Number of documents to surface (default {BaseOrchestratorConnector.DEFAULT_TOP_K})"
        },
        BaseOrchestratorConnector.CORPUS: {"explanation": "Inline corpus: a list of {doc_id, text} dicts"},
    }

    @classmethod
    def _run(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
        from haystack import Document, Pipeline
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        entries = [(str(doc.get("doc_id", str(i))), str(doc.get("text", ""))) for i, doc in enumerate(corpus)]

        # Haystack's store requires unique ids; surface a clean error rather than
        # leaking the framework's DuplicateDocumentError.
        doc_ids = [doc_id for doc_id, _ in entries]
        if len(set(doc_ids)) != len(doc_ids):
            duplicate = next(doc_id for doc_id in doc_ids if doc_ids.count(doc_id) > 1)
            raise ValueError(f"{cls.__name__}: duplicate doc_id {duplicate!r} in corpus; ids must be unique.")

        # Nothing rankable -> empty result (rather than leaking a framework
        # error): an empty/whitespace query, an all-empty-text corpus (BM25
        # divides by the average document length), or a non-positive top_k.
        effective_k = min(top_k, len(entries))
        if not query.strip() or not any(text.strip() for _, text in entries) or effective_k <= 0:
            return "", []

        store = InMemoryDocumentStore()
        store.write_documents([Document(id=doc_id, content=text) for doc_id, text in entries])

        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=effective_k))
        result = pipeline.run({"retriever": {"query": query}})

        # document.score is Optional[float]; the BM25 retriever filters out
        # non-positive scores, so every surfaced document carries a float score.
        documents = [
            {"doc_id": document.id, "text": document.content, "score": float(document.score)}
            for document in result["retriever"]["documents"]
        ]
        answer = documents[0]["text"] if documents else ""
        return answer, documents
