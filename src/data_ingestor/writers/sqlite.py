# Data Ingestor
# Package: data_ingestor.writers
# File: sqlite.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: Persist DataFrame to SQLite table 'data'.
#

from __future__ import annotations

from pathlib import Path

import pandas as pd
import sqlalchemy as sa


class SQLiteWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.engine = sa.create_engine(f"sqlite:///{self.path}")

    def write(self, df: pd.DataFrame) -> None:
        df.to_sql("data", self.engine, if_exists="replace", index=False)
