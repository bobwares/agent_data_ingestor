# Data Ingestor
# Package: data_ingestor.writers
# File: stdout.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: Pretty‑print DataFrame head + shape.
#

from __future__ import annotations

import pandas as pd
from rich import print as rprint


def write(df: pd.DataFrame) -> None:
    rprint(df.head())
    rprint(f"[dim]Rows: {len(df):,} • Columns: {len(df.columns)}[/]")


class StdoutWriter:
    """Render a quick preview of the DataFrame."""
