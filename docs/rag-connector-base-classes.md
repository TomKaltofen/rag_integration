# RAG connector base classes (Phase 0: survey + family map)

This is the Phase 0 deliverable for the connector-family direction
([fork issue #25](https://github.com/TomKaltofen/rag_integration/issues/25)):
the survey and design doc that justify how the families were cut. It is the
RAG analogue of open-kgo's `docs/kg-connector-base-classes.md` (the doc whose
labeled family map drove `open_kgo/feature_groups/kg/`).

The code for the six families already shipped in
[PR #31](https://github.com/TomKaltofen/rag_integration/pull/31); this document
is the missing artifact that explains *why those six*, records the landscape it
was derived from, and reconciles the shipped cuts against the v0 hypothesis in
#25.

## Why connect instead of rebuild

Today `rag_integration` is a build-your-own-RAG toolkit: a user assembles a
pipeline stage by stage (`document_source` -> `pii_redaction` -> `chunking` ->
`deduplication` -> `embedding` -> `vector_store` -> `retrieval` ->
`llm_response`). That is valuable, but every retrieval or generation strategy
has to be re-implemented as a stage inside this repo.

The open-source RAG ecosystem already implements most of these strategies. The
bet, copied straight from open-kgo, is to **connect to** that ecosystem rather
than rebuild it, exposing each tool through one uniform mloda
`Feature -> run_all` surface so a user swaps retrievers, rerankers, or
generators by changing options, not by rewriting a pipeline.

## Method (mirrors open-kgo)

open-kgo derived 9 knowledge-graph families from a ~103-system survey. We follow
the same five steps:

1. **Enumerate the landscape.** Build a table of open-source RAG tools
   (frameworks, engines, retrievers, rerankers, graph-RAG, text-to-SQL, eval).
2. **Record the contract signature per system:** what you give a query call,
   what it returns, what state it needs (prebuilt index? corpus? model? running
   server?), and crucially **can it run in-memory / from a fixture with no
   Docker?**
3. **Cluster by contract shape.** Group systems whose in/out signature matches.
   Each cluster is a candidate family. Vendor and paradigm fall out as
   Pedigree/backend notes, not family boundaries.
4. **Pressure-test against no-Docker.** Every candidate family must have at least
   one member that runs in-memory or from a fixture. A family with no in-memory
   substrate is a flag to surface, not a family to ship.
5. **Write it up**, ending in a labeled family map with a Pedigree column.

### The discriminating test for a family

Cut families by **query-contract shape**, not by paradigm (lexical / dense /
hybrid / late-interaction) and not by vendor (LlamaIndex / Haystack / ...).

> Does a candidate need a *different reader contract*, or just a *different
> backend behind the same contract*? Same in/out shape -> it is a **backend**
> inside an existing family. Different in/out shape -> it is its **own family**.

Worked example: BM25, dense bi-encoder, hybrid/RRF, and ColBERT late-interaction
all share one contract, `query + top_k -> ranked passages with scores`. They are
**not four families**, they are four backends of a single `retrieve` family with
different pedigrees. The genuine contract boundaries are where the in/out
signature actually changes: rerank takes candidates in, generate returns prose
plus citations, structured returns typed rows, graph_rag traverses a subgraph,
orchestrator is opaque end-to-end.

## The landscape survey

~90 open-source RAG systems, grouped by the contract cluster they fall into.
Per system: the query contract (in -> out), the state it needs, the no-Docker
answer, the family it maps to, and a pedigree tag
(`real-lib-inmem` / `real-lib-server` / `fixture-stub` / `research-prototype`).

No-Docker legend: **in-mem** = runs in-process after at most a pip install
(possibly a model download, noted); **fixture** = exercisable only via a static
fixture / REST stub; **server** = genuinely needs a running server or Docker.

### Orchestration frameworks and RAG applications (-> `orchestrator`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| LlamaIndex | `index.as_query_engine().query(str)` -> answer + `source_nodes` | in-mem `VectorStoreIndex`; LLM + embed model | in-mem (basic) | orchestrator | real-lib-inmem |
| Haystack 2.x | `Pipeline.run({"query"})` -> answers + retrieved documents | DocumentStore (InMemory available) | in-mem (basic) | orchestrator | real-lib-inmem |
| txtai | `embeddings.search(q)` / `rag(q)` -> results / answer | in-mem embeddings index (SQLite + ANN) | in-mem | orchestrator | real-lib-inmem |
| LangChain (RAG chains) | `RetrievalQA.invoke({"query"})` -> result + source_documents | in-mem vectorstore (FAISS/Chroma); LLM | in-mem (basic) | orchestrator | real-lib-inmem |
| DSPy | `module(question=...)` -> `Prediction.answer` | retriever + LM clients | in-mem (basic) | orchestrator | real-lib-inmem |
| Embedchain (mem0) | `app.add(src)`; `app.query(q)` -> answer | in-mem Chroma default; LLM API | in-mem (basic) | orchestrator | real-lib-inmem |
| LLMWare | `Query(library).query(text)` -> results; `Prompt` -> answer | library store (SQLite/Mongo); small model | in-mem (basic) | orchestrator | real-lib-inmem |
| LocalGPT | `run_localGPT.py` query -> answer + sources | local Chroma from ingest; local model | in-mem (basic) | orchestrator | real-lib-inmem |
| FlashRAG | `pipeline.run(dataset)` -> answers + eval | corpus + prebuilt index; model downloads | in-mem (basic) | orchestrator | research-prototype |
| AutoRAG | evaluator over QA data -> best pipeline; `query` -> answer | corpus + QA dataset + YAML; model APIs | in-mem (basic) | orchestrator | research-prototype |
| FLARE | `query` -> answer via iterative look-ahead retrieval | retriever + LM | in-mem (basic) | orchestrator | research-prototype |
| Self-RAG | `query` -> answer with retrieval/critique reflection tokens | fine-tuned Llama checkpoint; retriever | in-mem (basic) | generate / orchestrator | research-prototype |
| Canopy (Pinecone) | `ChatEngine.chat` / `query` -> answer + context | Pinecone index (external); OpenAI key | server | orchestrator | real-lib-server |
| Cognita (TrueFoundry) | REST `/query` -> answer + sources | backend server, vector DB, metadata store | server | orchestrator | real-lib-server |
| R2R (SciPhi) | REST `/rag` query -> answer + citations | Postgres + pgvector; Docker compose | server | orchestrator | real-lib-server |
| RAGFlow (InfiniFlow) | REST/UI query -> answer + grounded chunks | Docker stack (deep parse, ES, MySQL, MinIO) | server | orchestrator | real-lib-server |
| Verba (Weaviate) | UI/API query -> answer + context windows | Weaviate instance | server | orchestrator | real-lib-server |
| Quivr | API query -> answer over uploaded docs | Supabase/Postgres backend | server | orchestrator | real-lib-server |
| Danswer / Onyx | API query -> answer + cited sources | Docker stack (Postgres, Vespa, connectors) | server | orchestrator | real-lib-server |
| Khoj | API/chat query -> answer over personal corpus | Django server, embeddings DB | server | orchestrator | real-lib-server |
| PrivateGPT | API/UI query -> answer + source chunks | local LLM + embed; Qdrant/Chroma; FastAPI | server (local) | orchestrator | real-lib-server |
| AnythingLLM | API query (workspace) -> answer + citations | Node server, LanceDB default, LLM provider | server | orchestrator | real-lib-server |
| Open WebUI (RAG) | query + uploaded docs -> answer + refs | Open WebUI server + Ollama/OpenAI | server | orchestrator | real-lib-server |
| Dify | API app query -> answer + retrieved refs | Docker stack (Postgres, Redis, vector DB) | server | orchestrator | real-lib-server |
| Flowise | deployed flow API query -> answer | Node server; vector store; LLM keys | server | orchestrator | real-lib-server |
| kotaemon (Cinnamon) | UI/API query -> answer + citations (GraphRAG opt) | local/server install; vector + doc store | server | orchestrator | real-lib-server |
| Jina Reader | `GET r.jina.ai/<url>` -> clean markdown text | hosted REST API (no local index) | fixture (REST) | retrieve (ingest) | real-lib-server |

### Retrieval engines, vector stores, lexical / dense / late-interaction (-> `retrieve`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| FAISS | `query_emb + index -> top_k ids + distances` | prebuilt in-mem index | in-mem | retrieve | real-lib-inmem |
| Chroma | `query_text|emb + collection -> top_k docs + distances` | collection; optional embed fn | in-mem (embedded) / server | retrieve | real-lib-inmem |
| Qdrant | `query_vector(+filter) + collection -> scored points` | collection of vectors + payloads | in-mem (`:memory:`) / server | retrieve | real-lib-inmem |
| Milvus | `query_vectors + collection -> top_k ids + distances` | collection; index built | in-mem (Milvus-Lite) / server | retrieve | real-lib-inmem / server |
| LanceDB | `query_vector + table -> top_k rows + distances` | on-disk Lance table | in-mem (embedded) | retrieve | real-lib-inmem |
| Weaviate | `near_vector|near_text + class -> objects + scores` | schema + vectors; running node | server (embedded opt) | retrieve | real-lib-server |
| pgvector | `query_vector + table -> rows ORDER BY distance` | Postgres table + ANN index | server (Postgres) | retrieve | real-lib-server |
| Vespa | `YQL (ANN+BM25+rank) -> ranked hits` | deployed app package; running node | server | retrieve | real-lib-server |
| Elasticsearch (kNN + BM25) | `knn vector OR BM25 text + index -> ranked hits + _score` | index/mappings; running cluster | server | retrieve | real-lib-server |
| OpenSearch | `knn/neural OR BM25 + index -> ranked hits + _score` | index; running cluster (k-NN plugin) | server | retrieve | real-lib-server |
| Marqo | `query_text|image + index -> ranked docs + scores` | running server + embed models | server | retrieve | real-lib-server |
| Vald | `query_vector via gRPC -> nearest ids + distances` | K8s cluster (agent-NGT, gateway) | server (K8s) | retrieve | real-lib-server |
| Annoy | `query_vector + index -> top_k ids + distances` | prebuilt immutable index | in-mem | retrieve | real-lib-inmem |
| hnswlib | `query_vector + index -> top_k labels + distances` | in-mem HNSW graph | in-mem | retrieve | real-lib-inmem |
| ScaNN | `query_vector + searcher -> top_k ids + scores` | built partition/quantized index | in-mem | retrieve | real-lib-inmem |
| nmslib | `query_vector + index -> top_k ids + distances` | built in-mem index | in-mem | retrieve | real-lib-inmem |
| rank_bm25 | `tokenized_query + corpus -> per-doc score array` | in-mem tokenized corpus | in-mem | retrieve | real-lib-inmem |
| bm25s | `query_tokens + sparse index -> top_k ids + scores` | eagerly-scored sparse matrix (scipy) | in-mem | retrieve | real-lib-inmem |
| Pyserini (Anserini/Lucene) | `query_text + Lucene index -> ranked docids + scores` | prebuilt Lucene index; JVM (no server) | in-mem (needs Java) | retrieve | real-lib-inmem |
| Tantivy / tantivy-py | `query + index -> top docs + BM25 scores` | on-disk inverted index (embedded) | in-mem (embedded) | retrieve | real-lib-inmem |
| Whoosh | `query + index -> ranked hits + scores` | on-disk inverted index | in-mem | retrieve | real-lib-inmem |
| ColBERT | `query_tok_emb + token index -> MaxSim passages` | ColBERT checkpoint + PLAID index | in-mem (GPU pref.) | retrieve | real-lib-inmem |
| RAGatouille | `query_text + indexed corpus -> ranked passages` | ColBERT model + built index | in-mem | retrieve | real-lib-inmem |
| PLAID | `query_emb + compressed centroid index -> top_k` | quantized ColBERT index | in-mem | retrieve | research-prototype |
| SPLADE | `query_text -> sparse term weights -> ranked passages` | SPLADE model + sparse/inverted index | in-mem (index may be ES) | retrieve | research-prototype |
| DPR | `question_emb + FAISS passage index -> top_k passages` | Q/ctx encoders + FAISS index | in-mem | retrieve | research-prototype |
| sentence-transformers (bi-encoder) | `query_emb + corpus_emb -> top_k (semantic_search)` | downloaded model; corpus embeddings | in-mem (model dl) | retrieve | real-lib-inmem |
| Instructor embeddings | `(instruction, text) -> embedding` (feed to ANN) | downloaded INSTRUCTOR model | in-mem (model dl) | retrieve | real-lib-inmem |
| BGE / FlagEmbedding (retrieval) | `text -> embedding` (dense+sparse+colbert) | downloaded BGE model | in-mem (model dl) | retrieve | real-lib-inmem |
| ELSER (ES learned sparse) | `query_text -> expanded sparse tokens -> ranked hits` | ES cluster + deployed ELSER model | server | retrieve | real-lib-server |

### Rerankers (-> `rerank`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| FlashRank | `query + candidates -> reordered passages + scores` | ONNX cross-encoder download | in-mem (model dl) | rerank | real-lib-inmem |
| sentence-transformers CrossEncoder | `query + candidates -> reordered + relevance scores` | cross-encoder download | in-mem (model dl) | rerank | real-lib-inmem |
| BGE-reranker (FlagEmbedding) | `query + candidates -> reordered + scores` | BGE reranker download | in-mem (model dl) | rerank | real-lib-inmem |
| MixedBread mxbai-rerank | `query + candidates -> reordered + scores` | mxbai-rerank download | in-mem (model dl) | rerank | real-lib-inmem |
| monoT5 / castorini | `query + passage -> relevance score -> reordered` | T5 reranker download (pygaggle) | in-mem (model dl) | rerank | real-lib-inmem |
| ColBERT-as-reranker | `query + candidates -> MaxSim scores -> reordered` | ColBERT checkpoint download | in-mem (model dl) | rerank | real-lib-inmem |
| rerankers (AnswerDotAI) | `query + candidates -> reordered` (unified API) | backend model / API key | in-mem (model dl) | rerank | real-lib-inmem |
| RankGPT | `query + candidates -> LLM permutation -> reordered` | LLM API key | server (LLM API) | rerank | research-prototype |
| Cohere-rerank | `query + candidates -> reordered + scores` | Cohere API key (not OSS) | server (hosted) | rerank | real-lib-server |
| Lexical token-overlap reranker | `query + candidates -> reordered by overlap` | none | in-mem | rerank | fixture-stub |

### Answer generators (-> `generate`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| Template / extractive responder | `query + passages -> templated/extractive answer` | none | in-mem | generate | fixture-stub |
| HuggingFace QA pipeline (extractive) | `question + context -> answer span + score` | QA model download | in-mem (model dl) | generate | real-lib-inmem |
| Haystack readers | `query + docs -> answer span / generated + citations` | reader/LLM download or API key | in-mem (model dl) | generate | real-lib-inmem |
| LangChain generation | `query + passages -> LLM answer + citations` | LLM API key or local model | in-mem (model dl) | generate | real-lib-inmem |
| llama.cpp / Ollama | `query + passages prompt -> generated answer` | GGUF download / Ollama daemon | in-mem (model dl) | generate | real-lib-inmem |
| FiD (fusion-in-decoder) | `query + N passages -> fused generated answer` | trained FiD checkpoint | in-mem (model dl) | generate | research-prototype |

### Graph-RAG (-> `graph_rag`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| GraphRAG via networkx | `query -> in-mem graph traversal -> passages` | graph build (in-process) | in-mem | graph_rag | fixture-stub |
| Microsoft GraphRAG | `query -> community graph traversal -> answer` | graph build (parquet artifacts); LLM | in-mem (post-index) | graph_rag | real-lib-inmem |
| nano-graphrag | `query -> graph traversal -> context -> answer` | graph build; LLM API key | in-mem (file artifacts) | graph_rag | real-lib-inmem |
| LlamaIndex KnowledgeGraph / PropertyGraphIndex | `query -> KG traversal -> passages -> answer` | graph build; LLM API key | in-mem | graph_rag | real-lib-inmem |
| LightRAG | `query -> dual-level graph + vector -> answer` | graph build; embed/LLM API key | in-mem (file artifacts) | graph_rag | real-lib-inmem |
| HippoRAG | `query -> personalized PageRank over KG -> passages` | graph build; model download; LLM | in-mem (model dl) | graph_rag | research-prototype |
| Neo4j GraphRAG | `query -> Cypher/vector graph retrieval -> passages` | running Neo4j DB; LLM API key | server | graph_rag | real-lib-server |

### Text-to-SQL / structured retrieval (-> `structured`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| Rule-based text-to-SQL (in-mem SQLite) | `question + table -> SQL -> typed rows` | in-mem SQLite copy of the table | in-mem | structured | fixture-stub |
| LlamaIndex NLSQLTableQueryEngine | `NL question + schema -> SQL -> rows -> answer` | SQL DB (SQLite ok); LLM API key | in-mem (SQLite) | structured | real-lib-inmem |
| LangChain SQLDatabaseChain | `NL question + schema -> SQL -> rows -> answer` | SQL DB (SQLite ok); LLM API key | in-mem (SQLite) | structured | real-lib-inmem |
| Vanna.AI | `NL question + trained schema -> SQL -> rows` | vector store of schema; DB; LLM API key | in-mem (embeddable) | structured | real-lib-inmem |
| sqlcoder (defog) | `NL question + schema prompt -> SQL` | sqlcoder model download | in-mem (model dl) | structured | real-lib-inmem |
| DAIL-SQL / DIN-SQL | `NL question + schema (few-shot) -> SQL` | LLM API key (GPT-4); benchmark data | server (LLM API) | structured | research-prototype |
| PICARD | `NL question + schema -> constrained decode -> SQL` | T5 model + PICARD parsing server | server | structured | research-prototype |

### Evaluation harnesses (cross-cutting / out-of-scope for the connector layer)

These do not fit a retrieval family: they consume `(query, answer, contexts,
ground_truth)` and emit metric scores. They belong on top of the existing
`evaluation/` module, not as a connector family. Recorded for completeness.

| System | Query contract (in -> out) | No-Docker? | Disposition |
|---|---|---|---|
| RAGAS | `(query, answer, contexts, ground_truth) -> faithfulness/relevance scores` | in-mem (LLM API) | out-of-scope (eval) |
| TruLens | `(query, answer, contexts) + feedback fns -> scores (logged)` | in-mem (sqlite) | out-of-scope (eval) |
| DeepEval | `(query, answer, contexts, ground_truth) -> scores (pytest-style)` | in-mem (LLM API) | out-of-scope (eval) |
| ARES | `(query, answer, contexts) -> trained-judge scores` | in-mem (model dl) | out-of-scope (eval) |
| Phoenix (Arize) | `(query, answer, contexts, gt) -> scores + traces` | in-mem (local app) | out-of-scope (eval) |
| Giskard RAG (RAGET) | `(query, answer, contexts, gt) -> component scores + tests` | in-mem (LLM API) | out-of-scope (eval) |
| continuous-eval (relari) | `(query, answer, contexts, gt) -> modular metric scores` | in-mem (LLM API) | out-of-scope (eval) |

## The labeled family map

Six families, each a thin `Base<Family>Connector` FeatureGroup plus one or more
concrete backends gated by a `<family>_backend` selector, with an inheritable
contract-test suite. Every family has at least one no-Docker concrete (the
hard constraint from step 4). Contracts below are copied verbatim from the
shipped base classes in PR #31.

| Family | Reader contract (in -> out) | No-Docker concrete (shipped) | Other backends | Pedigree of the anchor |
|---|---|---|---|---|
| `retrieve` | `query_text + corpus + top_k -> ranked passages w/ scores` (`retrieved_passages: [{doc_id, text, score, rank}]`) | `Bm25sRetriever` (`bm25s`, zero-download lexical) | `TfidfRetriever` (vector-space lexical); dense/FAISS to come (#36) | real-lib-inmem |
| `rerank` | `query_text + candidates + top_k -> reordered passages w/ scores` (`reranked_passages`) | `LexicalReranker` (pure-Python token overlap, zero-download) | `FlashRankReranker` (ONNX cross-encoder, `rerank` extra, CI-skip on model download) | fixture-stub anchor + real-lib |
| `generate` | `query_text + passages -> answer + citations` (`generated_answer: {answer, citations}`), grounded by construction | `ExtractiveResponder` (stdlib sentence extraction) | `TemplateResponder` (multi-citation template); LLM generators are pedigree backends for later | fixture-stub anchor |
| `graph_rag` | `query_text + nodes + edges + top_k -> ranked passages` (`graph_passages`); query-overlap + one-hop neighbour bonus | `AdjacencyGraphRag` (stdlib adjacency map, zero-download) | `NetworkxGraphRag` (`networkx`, `graph` extra); parity test pins identical ranking | fixture-stub anchor + real-lib |
| `structured` | `question + table -> SQL -> typed rows` (`structured_rows: {sql, rows}`); in-mem SQLite, single-SELECT sqlglot guard | `RuleBasedSql` (deterministic NL->SQL over in-mem SQLite) | `AggregateSql` (aggregation queries) | fixture-stub anchor |
| `orchestrator` | `query_text + corpus + top_k -> answer + documents` (internals opaque) (`orchestrated_answer: {answer, documents}`) | `HaystackOrchestrator` (Haystack 2.x BM25 pipeline, offline, telemetry off) | `R2RFixtureOrchestrator` (file-fixture REST stub with `SUPPORTED_VALUES` + stripped params) | real-lib-inmem + fixture-stub |

### Why these six, from the survey

- **`retrieve`** absorbs the entire vector-store / lexical / late-interaction
  column (FAISS, Chroma, bm25s, ColBERT, ...): all share
  `query + top_k -> ranked passages`. Paradigm and vendor are backend and
  pedigree distinctions, not family boundaries.
- **`rerank`** is a genuine contract boundary: it takes *candidates* in, not a
  corpus (FlashRank, cross-encoders, RankGPT).
- **`generate`** returns prose plus citations, a different out-shape from a
  ranked list (extractive QA, Haystack readers, local LLMs).
- **`graph_rag`** traverses a node/edge graph; the distinguishing value is
  connected context (GraphRAG, LightRAG, HippoRAG, networkx prototypes).
- **`structured`** returns typed rows via generated SQL, an in/out shape none of
  the others express (Vanna, NLSQLTableQueryEngine).
- **`orchestrator`** is the opaque end-to-end surface for whole frameworks and
  apps (LlamaIndex, Haystack, txtai, and the server-shaped R2R/RAGFlow/Verba/...
  reached through a fixture stub).

## Reconciliation with the #25 v0 hypothesis

#25 proposed exactly these six as a hypothesis, to be confirmed or revised by the
survey. The survey **confirms all six** and resolves the open questions #25
flagged:

| #25 open question | Resolution from the survey |
|---|---|
| Is `agentic` (iterative retrieve + plan loop) a distinct family? | **No.** FLARE, Self-RAG, AutoRAG, DSPy all return `query -> answer` with the loop *internal*: same opaque contract as `orchestrator`. Agentic is a backend behaviour, not a contract shape. If a future tool exposes the plan/step loop as its query surface, revisit. |
| Does `structured` (text-to-SQL) belong here? | **Yes, kept.** `question + table -> SQL -> rows` is a contract shape no other family expresses, and it has a clean no-Docker anchor (in-mem SQLite). It is *not* a backend of `retrieve` (rows, not ranked passages). |
| Split `orchestrator` into in-memory vs server? | **No, kept unified.** The contract is identical; in-memory vs server is a *pedigree*, handled by two backends under one family (`HaystackOrchestrator` real-lib-inmem, `R2rFixtureOrchestrator` fixture-stub), exactly the open-kgo `rest_public` pattern. |
| How much of `graph_rag` should defer to open-kgo? | Kept as its own RAG family with a self-contained node/edge contract. The synergy (consume an open-kgo KG connector as the corpus) stays an open question, not a blocker. |
| Where do eval harnesses (RAGAS, TruLens, ...) sit? | **Out of scope** for the connector layer. They are cross-cutting, consuming `(query, answer, contexts, gt)`; they belong on top of the existing `evaluation/` module. |

Net: the v0 hypothesis held. No family was merged, split, or added. The survey's
job here was confirmation plus pinning the no-Docker anchor for each family,
which it did: every shipped family has at least one zero-download or fixture
concrete.

## Cross-cutting concerns (mixins, not families)

Mirroring open-kgo's `EntityFilter` / `Pagination` / `Traversal` / `Inference`
mixins, these recur across families and belong in a shared `mixins.py`, not on
the family axis:

- **TopK / score-threshold** (retrieve, rerank, graph_rag, orchestrator)
- **Metadata filter** (corpus subset selection)
- **Corpus / index handle** (the locator: which prebuilt index or fixture)
- **Embedding-model selection** (retrieve dense backend, graph_rag)
- **Citation / provenance** (generate, orchestrator)

As shipped in PR #31 these are implemented inline in each family's `base.py`
rather than extracted into a shared `mixins.py` / `errors.py`. Extracting them is
tracked in
[#35](https://github.com/TomKaltofen/rag_integration/issues/35), so this doc
records the intended axis, not the current file layout.

## Migration seam vs the existing stage pipeline

This is where "easy migration" lives or dies. The existing stage pipeline
(`feature_groups/rag_pipeline/`) already has a FAISS-backed `retrieval` stage
and an `llm_response` stage, which overlap the `retrieve` and `generate`
connectors head-on. The intended resolution:

- **Same output row shape.** A connector and the corresponding stage emit the
  same passage / answer row shape, so a downstream feature is agnostic to which
  produced it.
- **Stages = build-your-own; connectors = bring an existing tool.** A user
  either chains the native stages or drops in one connector that subsumes
  embed + index + retrieve. The existing FAISS retrieval stage should fold in as
  the canonical *dense backend* of the `retrieve` family rather than remaining a
  separate concept.
- **Migration = swap the connector id / options**, same `Feature -> run_all`
  shape, no pipeline rewrite. This is the open-kgo "swap `rdflib_sparql` for any
  of nine connectors, same shape" promise applied to RAG.

As shipped, `retrieve` has lexical backends (`bm25s`, `tfidf`) but **no dense /
FAISS backend yet**, and the seam to `rag_pipeline` is not wired. Both are
tracked in
[#36](https://github.com/TomKaltofen/rag_integration/issues/36). Stating the
seam here, before more families land, keeps the connector layer from forking
into a parallel, incompatible world next to the stage pipeline.

## Canonical package path (naming-variance decision)

#25 proposed a top-level `rag/` package (mirroring open-kgo's `kg/`); PR #31
shipped under `rag_integration/feature_groups/connectors/`. **Decision: keep
`rag_integration/feature_groups/connectors/`** as the canonical path.

Rationale:

- It matches this repo's existing convention: every plugin group already lives
  under `rag_integration/feature_groups/` (`rag_pipeline`, `image_pipeline`,
  `evaluation`, `datasets`). A top-level `rag/` would be the only exception and
  would sit oddly next to a repo already named `rag_integration`.
- The code, tests (`tests/connectors/`), and the README section shipped in #31
  already use it; renaming now would churn 60+ files for no functional gain.
- open-kgo's `kg/` lives under `open_kgo/feature_groups/kg/`, so
  `rag_integration/feature_groups/connectors/` is the faithful analogue, not a
  divergence. The only cosmetic difference is the leaf name (`connectors` vs
  `kg`/`rag`), which reads correctly here because the families *are* connectors
  to external tools.

Wherever #25's prose refers to `rag/base.py`, `rag/mixins.py`, or
`rag/tests/rag_contract.py`, read those as
`feature_groups/connectors/{base,mixins}.py` and
`tests/connectors/<family>/<family>_contract.py`. This supersedes the `rag/`
paths in #25.

## Open questions (carried forward)

- **graph_rag x open-kgo synergy.** Should `graph_rag` retrieval *consume* an
  open-kgo KG connector as its corpus rather than carry its own graph substrate?
  Candidate, not yet a commitment.
- **Hybrid as a backend vs a mixin.** Hybrid/RRF fusion sits inside `retrieve`
  as a backend today; if fusion needs to combine *across* families it may want a
  mixin instead.
- **Dense `retrieve` anchor.** The no-Docker dense anchor (numpy brute-force or
  the existing FAISS path) is identified but not yet shipped (#36).

## References

- The template: open-kgo, `code/mvp/open-kgo`, esp.
  `open_kgo/feature_groups/kg/README.md`, `.../base.py`, `.../mixins.py`,
  `.../errors.py`, `.../tests/kg_contract.py`.
- The shipped families:
  [PR #31](https://github.com/TomKaltofen/rag_integration/pull/31),
  `rag_integration/feature_groups/connectors/`.
- The strategy issue:
  [#25](https://github.com/TomKaltofen/rag_integration/issues/25).
- Follow-ups this doc references:
  [#33](https://github.com/TomKaltofen/rag_integration/issues/33) (family-map
  README), [#35](https://github.com/TomKaltofen/rag_integration/issues/35)
  (mixins/errors), [#36](https://github.com/TomKaltofen/rag_integration/issues/36)
  (FAISS dense backend + migration seam).
