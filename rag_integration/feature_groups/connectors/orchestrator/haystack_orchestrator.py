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

        store = InMemoryDocumentStore()
        store.write_documents(
            [
                Document(id=str(doc.get("doc_id", str(i))), content=str(doc.get("text", "")))
                for i, doc in enumerate(corpus)
            ]
        )

        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=top_k))
        result = pipeline.run({"retriever": {"query": query}})

        documents = [
            {"doc_id": document.id, "text": document.content, "score": float(document.score)}
            for document in result["retriever"]["documents"]
        ]
        answer = documents[0]["text"] if documents else ""
        return answer, documents
