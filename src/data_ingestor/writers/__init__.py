# Data Ingestor
# Package: data_ingestor.writers
# File: __init__.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: Register writer implementations and factory helper.
#

from .blob import BlobWriter  # noqa: F401
from .stdout import StdoutWriter  # noqa: F401
from .sqlite import SQLiteWriter  # noqa: F401

TargetWriter = object  # placeholder type alias


class WriterFactory:
    """Return a writer instance based on *uri* string."""

    @staticmethod
    def from_uri(uri: str) -> TargetWriter:
        if uri == "blob":
            return BlobWriter()
        if uri == "stdout":
            return StdoutWriter()
        if uri.startswith("sqlite:///"):
            return SQLiteWriter(uri.replace("sqlite:///", ""))
        raise ValueError(f"Unknown writer: {uri!r}")
