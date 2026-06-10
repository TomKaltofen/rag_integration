"""Connector families: wrap external open-source RAG tools under one mloda surface.

Each family is a thin ``Base<Family>Connector`` FeatureGroup plus one or more
concrete backends, paired with an inheritable contract-test suite. Unlike the
stage pipeline under ``rag_pipeline/`` (build-your-own RAG from chained
FeatureGroups), a connector exposes a whole external retrieval/rerank/generate
tool through a single feature. See issue #25 for the family taxonomy and the
selection rationale.

Family axis is the query-contract shape; the canonical concrete per family is
the zero-download, deterministic backend that anchors the CI contract suite.
"""
