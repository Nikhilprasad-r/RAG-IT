"""Built-in connectors register themselves with `ragkit.core.registry` on import."""

from ragkit.ingest.connectors.base import (
    ConnectorOptions,
    bounded_map,
    create_connector,
    glob_match,
    guess_mime_type,
)
from ragkit.ingest.connectors.files import FilesConnector
from ragkit.ingest.connectors.github import GitHubConnector
from ragkit.ingest.connectors.web import WebConnector

__all__ = [
    "ConnectorOptions",
    "FilesConnector",
    "GitHubConnector",
    "WebConnector",
    "bounded_map",
    "create_connector",
    "glob_match",
    "guess_mime_type",
]
