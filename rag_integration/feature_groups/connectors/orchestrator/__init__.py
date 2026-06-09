"""The ``orchestrator`` connector family: query + corpus -> answer (opaque pipeline)."""

from __future__ import annotations

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.connectors.orchestrator.haystack_orchestrator import HaystackOrchestrator
from rag_integration.feature_groups.connectors.orchestrator.r2r_fixture_orchestrator import R2RFixtureOrchestrator

__all__ = ["BaseOrchestratorConnector", "HaystackOrchestrator", "R2RFixtureOrchestrator"]
