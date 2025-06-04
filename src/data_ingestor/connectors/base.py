# Data Ingestor
# Package: data_ingestor.connectors
# File: base.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: Abstract connector protocol (DataFrame + bytes interfaces).
#

from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class SourceConnector(Protocol):
    """Interface for reading external data."""

    uri: str

    def read(self) -> pd.DataFrame: ...

    @abc.abstractmethod
    def read_bytes(self) -> bytes: ...
