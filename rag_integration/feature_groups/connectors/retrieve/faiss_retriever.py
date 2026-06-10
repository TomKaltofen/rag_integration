"""Dense FAISS retrieve connector: the repo's FAISS path under the family contract.

Canonical dense concrete for the ``retrieve`` family (issue #36): the same
FAISS nearest-neighbor search the stage pipeline's ``retrieval`` stage runs
(``feature_groups/rag_pipeline/retrieval``), folded in behind the family's
``query_text + corpus + top_k -> ranked passages`` contract. The stage searches
a pre-built on-disk index; this connector serves the family's inline-corpus
contract by embedding the corpus per call and searching an in-memory index.
Both paths emit the same passage row shape, so a downstream feature is agnostic
to which produced it (see ``tests/integration/test_stage_connector_parity.py``).

Embeddings come from the stage pipeline's deterministic
:class:`~rag_integration.feature_groups.rag_pipeline.embedding.hash_embed.HashEmbedder`
(zero-download, unit-normalized, query embedded independently of the corpus,
exactly how a dense bi-encoder behaves). Requires the ``faiss`` extra
(``faiss-cpu``), like the rest of the FAISS path.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.rag_pipeline.embedding.hash_embed import HashEmbedder


class FaissDenseRetriever(BaseRetrieveConnector):
    """Dense FAISS retrieval over an inline corpus (``retrieve_backend="faiss"``).

    Embeds corpus and query with the repo's deterministic ``HashEmbedder``,
    builds an in-memory ``IndexFlatIP`` over the corpus vectors, and ranks by
    inner product. The embedder L2-normalizes every vector, so the inner
    product is the cosine similarity and scores are higher-is-more-relevant.
    The corpus is per-call, so there is no shared state to cache and repeated
    calls are idempotent. Family rule: at most ``top_k`` passages come back and
    only those scoring positively, so a degenerate query (empty, or sharing no
    hashed terms with the corpus) yields no passages.
    """

    # The embedder hashes terms into a fixed-width vector; 384 is its own
    # default and is ample for the small inline corpora this family serves.
    # ``model_name`` is ignored by the hash embedder.
    _EMBED_DIM = 384

    RETRIEVE_BACKENDS = {
        "faiss": "Dense FAISS retrieval (cosine over deterministic hash embeddings)",
    }

    PROPERTY_MAPPING = {
        BaseRetrieveConnector.RETRIEVE_BACKEND: {"explanation": "Use 'faiss' for dense FAISS retrieval"},
        BaseRetrieveConnector.QUERY_TEXT: {"explanation": "Raw text query to search the corpus"},
        BaseRetrieveConnector.TOP_K: {
            "explanation": f"Number of passages to return (default {BaseRetrieveConnector.DEFAULT_TOP_K})"
        },
        BaseRetrieveConnector.CORPUS: {"explanation": "Inline corpus: a list of {doc_id, text} dicts"},
    }

    @classmethod
    def _rank(cls, query: str, texts: List[str], top_k: int) -> List[Tuple[int, float]]:
        import faiss

        # Embed corpus and query with the same stage-pipeline embedder. Each
        # text embeds independently (no shared vocabulary), so the query is
        # embedded exactly as a dense bi-encoder would embed it.
        vectors = HashEmbedder._embed_texts(list(texts) + [query], cls._EMBED_DIM, "default")
        corpus_array = np.array(vectors[:-1], dtype=np.float32)
        query_array = np.array([vectors[-1]], dtype=np.float32)

        # Vectors are unit-length, so inner product == cosine similarity and
        # FAISS returns the pairs best-first.
        index = faiss.IndexFlatIP(cls._EMBED_DIM)
        index.add(corpus_array)
        scores, indices = index.search(query_array, top_k)

        pairs = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        # Family rule: only positively scoring passages are returned. This also
        # covers the degenerate query: it embeds to a zero or orthogonal
        # vector, every cosine is <= 0, and no pair survives the filter.
        return [(idx, score) for idx, score in pairs if score > 0.0]
