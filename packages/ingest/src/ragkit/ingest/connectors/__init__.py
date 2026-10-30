"""Built-in connectors register themselves with `ragkit.core.registry` on import."""

from ragkit.ingest.connectors.base import (
    ConnectorOptions,
    bounded_map,
    create_connector,
    glob_match,
    guess_mime_type,
)
from ragkit.ingest.connectors.confluence import ConfluenceConnector
from ragkit.ingest.connectors.files import FilesConnector
from ragkit.ingest.connectors.github import GitHubConnector
from ragkit.ingest.connectors.notion import NotionConnector
from ragkit.ingest.connectors.web import WebConnector

__all__ = [
    "ConfluenceConnector",
    "ConnectorOptions",
    "FilesConnector",
    "GitHubConnector",
    "NotionConnector",
    "WebConnector",
    "bounded_map",
    "create_connector",
    "glob_match",
    "guess_mime_type",
]
