#!/usr/bin/env python3
"""Swap-backends demo (issue #34).

The open-kgo promise applied to RAG: swapping one connector backend for another
is an edit to the options dict, never a pipeline rewrite. Every family is driven
through one invariant call shape (:func:`run_connector`): build a Feature on the
family's root name, hand ``mlodaAPI.run_all`` the matching options plus the
backend's FeatureGroup, read the single result row. The calling code never
changes. A swap changes only the data:

  * within a family: the ``<family>_backend`` selector value (e.g.
    ``retrieve_backend="bm25s"`` -> ``"tfidf"``);
  * across families:  the selector key and the root feature name, with the same
    ``query_text`` / ``corpus`` / ``top_k`` inputs (e.g. a ``retrieve`` connector
    returning ranked passages vs an ``orchestrator`` connector returning an
    answer, over identical inputs).

Run with::

    python -m cli.swap_demo

The pure-Python backends (``tfidf``, ``extractive``, ``template``) always run.
The ``bm25s`` retrieve backend needs ``uv sync --extra connectors``; the
``haystack`` orchestrator backend needs ``uv sync --extra orchestrator``. A
missing extra is reported and skipped, never fatal, so the swap story is visible
on any install.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Set, Tuple, Type

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

# A tiny corpus where the query shares terms with the pet docs and nothing else,
# so every lexical backend ranks the same two documents above the distractors.
CORPUS: List[Dict[str, str]] = [
    {"doc_id": "d0", "text": "the mat lay flat on the floor by the window"},
    {"doc_id": "d1", "text": "a dog can be a loyal and energetic pet"},
    {"doc_id": "d2", "text": "a cat is an independent and curious pet"},
    {"doc_id": "d3", "text": "cars need regular engine oil and maintenance"},
]
QUERY = "cat pet"
TOP_K = 3

# The inputs that the retrieve and orchestrator families share verbatim. The
# across-family swap adds only a selector key on top of these.
SHARED_INPUTS: Dict[str, Any] = {"query_text": QUERY, "corpus": CORPUS, "top_k": TOP_K}


def run_connector(
    root_feature: str,
    options: Dict[str, Any],
    providers: Set[Type[FeatureGroup]],
) -> Any:
    """The invariant call shape every connector family shares.

    Build one Feature on the family's root name, run it through ``mlodaAPI`` with
    the backend's FeatureGroup, and return the single result row's value. Nothing
    here knows which family or backend it drives: that lives entirely in
    ``options`` and ``providers``, which is the whole point of the demo.
    """
    feature = Feature(root_feature, options=Options(context=dict(options)))
    result = mlodaAPI.run_all(
        [feature],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups(providers),
    )
    rows: List[Any] = result[0] if result and isinstance(result[0], list) else list(result)
    for row in rows:
        if isinstance(row, dict) and root_feature in row:
            return row[root_feature]
    raise AssertionError(f"run_all returned no '{root_feature}' row: {result!r}")


def _format_passages(passages: List[Dict[str, Any]]) -> str:
    """One line per ranked passage: ``rank. doc_id (score) text``."""
    if not passages:
        return "    (no passages)"
    lines = [f"    {p['rank']}. {p['doc_id']} ({p['score']:.4f}) {p['text']}" for p in passages]
    return "\n".join(lines)


def _format_answer(answer: Dict[str, Any]) -> str:
    """Render a ``{answer, citations}`` (generate) result."""
    if not answer.get("answer"):
        return "    (no answer)"
    return f"    answer:    {answer['answer']}\n    citations: {answer['citations']}"


def _run_or_skip(label: str, fn: Callable[[], Any], render: Callable[[Any], str]) -> None:
    """Run a backend, skipping cleanly if its optional extra is not installed."""
    try:
        value = fn()
    except ImportError as exc:
        print(f"  {label}\n    (skipped: optional extra not installed -> {exc})")
        return
    print(f"  {label}\n{render(value)}")


def demo_within_family_retrieve() -> None:
    """Swap the retrieve backend in place: only ``retrieve_backend`` changes."""
    from rag_integration.feature_groups.connectors.retrieve import Bm25sRetriever, TfidfRetriever

    print("\n[A] Within the retrieve family: swap retrieve_backend, identical call shape")

    def with_backend(backend: str, provider: Type[FeatureGroup]) -> List[Dict[str, Any]]:
        options = {"retrieve_backend": backend, **SHARED_INPUTS}
        result: List[Dict[str, Any]] = run_connector("retrieved_passages", options, {provider})
        return result

    _run_or_skip('retrieve_backend="bm25s"', lambda: with_backend("bm25s", Bm25sRetriever), _format_passages)
    _run_or_skip('retrieve_backend="tfidf"', lambda: with_backend("tfidf", TfidfRetriever), _format_passages)


def demo_within_family_generate() -> None:
    """Swap the generate backend in place: only ``generate_backend`` changes."""
    from rag_integration.feature_groups.connectors.generate import ExtractiveResponder, TemplateResponder

    print("\n[B] Within the generate family: swap generate_backend, identical call shape")
    passages = [{"doc_id": doc["doc_id"], "text": doc["text"]} for doc in CORPUS]

    def with_backend(backend: str, provider: Type[FeatureGroup]) -> Dict[str, Any]:
        options = {"generate_backend": backend, "query_text": QUERY, "passages": passages}
        result: Dict[str, Any] = run_connector("generated_answer", options, {provider})
        return result

    _run_or_skip(
        'generate_backend="extractive"', lambda: with_backend("extractive", ExtractiveResponder), _format_answer
    )
    _run_or_skip('generate_backend="template"', lambda: with_backend("template", TemplateResponder), _format_answer)


def demo_across_families() -> None:
    """Swap across families: same inputs, only the selector key and root name change."""
    from rag_integration.feature_groups.connectors.orchestrator import HaystackOrchestrator
    from rag_integration.feature_groups.connectors.retrieve import TfidfRetriever

    print("\n[C] Across families: same query/corpus/top_k, swap selector + root feature name")

    def as_retrieve() -> List[Dict[str, Any]]:
        options = {"retrieve_backend": "tfidf", **SHARED_INPUTS}
        result: List[Dict[str, Any]] = run_connector("retrieved_passages", options, {TfidfRetriever})
        return result

    def as_orchestrator() -> Dict[str, Any]:
        options = {"orchestrator_backend": "haystack", **SHARED_INPUTS}
        result: Dict[str, Any] = run_connector("orchestrated_answer", options, {HaystackOrchestrator})
        return result

    _run_or_skip("retrieve connector  -> retrieved_passages", as_retrieve, _format_passages)
    _run_or_skip(
        "orchestrator connector -> orchestrated_answer",
        as_orchestrator,
        lambda answer: (
            f"    answer:    {answer['answer']}\n    documents: {[d['doc_id'] for d in answer['documents']]}"
        ),
    )


DEMOS: Tuple[Callable[[], None], ...] = (
    demo_within_family_retrieve,
    demo_within_family_generate,
    demo_across_families,
)


def main() -> None:
    print(__doc__.splitlines()[0] if __doc__ else "Swap-backends demo")
    print(f"query={QUERY!r}  top_k={TOP_K}  corpus={len(CORPUS)} docs")
    for demo in DEMOS:
        demo()
    print("\nEvery result above came from the same run_connector(...) call. A swap is data, not code.")


if __name__ == "__main__":
    main()
