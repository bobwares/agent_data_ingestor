# Data Ingestor
# Package: data_ingestor.writers
# File: base.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: TargetWriter protocol for DataFrame persistence.
#

from __future__ import annotations

import abc
from typing import Protocol

import pandas as pd


class TargetWriter(Protocol):
    @abc.abstractmethod
    def write(self, df: pd.DataFrame) -> None: ...
