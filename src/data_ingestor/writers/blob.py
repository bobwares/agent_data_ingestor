# Data Ingestor
# Package: data_ingestor.writers
# File: blob.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: Emit raw bytes (length) or Base‑64 for LLM input.
#

from __future__ import annotations

import base64
import sys


class BlobWriter:
    """Output bytes for downstream LLM processing."""

    def write_bytes(self, data: bytes, *, raw: bool = False) -> None:
        if raw:
            print(len(data))
            return

        b64 = base64.b64encode(data).decode()
        sys.stdout.write(b64)
        sys.stdout.flush()
