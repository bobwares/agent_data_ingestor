# Data Ingestor
# Package: data_ingestor.connectors
# File: csv.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: Local‑file connector; tabular CSV or raw bytes.
#

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_ingestor.connectors.base import SourceConnector


class CSVConnector(SourceConnector):
    """Read a local file: CSV for tabular access or raw bytes for blob mode."""

    def __init__(self, uri: str) -> None:
        self.uri = uri

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self._path)

    def read_bytes(self) -> bytes:
        return self._path.read_bytes()

    @property
    def _path(self) -> Path:
        return Path(self.uri.replace("file://", "")).expanduser().resolve()
