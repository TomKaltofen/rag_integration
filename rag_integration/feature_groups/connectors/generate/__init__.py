"""The ``generate`` connector family: query + passages -> answer + citations."""

from __future__ import annotations

from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector
from rag_integration.feature_groups.connectors.generate.extractive_responder import ExtractiveResponder
from rag_integration.feature_groups.connectors.generate.template_responder import TemplateResponder

__all__ = ["BaseGenerateConnector", "ExtractiveResponder", "TemplateResponder"]
