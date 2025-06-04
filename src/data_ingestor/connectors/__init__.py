# Data Ingestor
# Package: data_ingestor.connectors
# File: __init__.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: Export available connectors and simple factory.
#

from .base import SourceConnector  # noqa: F401
from .csv import CSVConnector  # noqa: F401

__all__: list[str] = ["CSVConnector", "SourceConnector"]


class ConnectorFactory:
    """Create a SourceConnector instance from a URI string."""

    _SCHEME_MAP = {
        "file": CSVConnector,
        "": CSVConnector,  # bare paths
    }

    @classmethod
    def from_uri(cls, uri: str) -> "SourceConnector":
        scheme = uri.split(":", 1)[0] if "://" in uri else "file"
        connector_cls = cls._SCHEME_MAP.get(scheme)
        if connector_cls is None:
            raise ValueError(f"No connector for scheme {scheme!r}.")
        return connector_cls(uri)
