# Data Ingestor
# Package: data_ingestor
# File: api.py
# Version: 1.0.1
# Author: Bobwares
# Date: 2025-06-03 14:40
# Description: FastAPI surface exposing a placeholder /ingest endpoint.
#

from fastapi import FastAPI, File, UploadFile

app = FastAPI(title="Data Ingestor API", version="1.0.1")


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict[str, str]:
    """Upload an arbitrary file for ingestion.

    This endpoint is a placeholder. It simply returns basic metadata about
    the uploaded file until the full ingestion pipeline is implemented.
    """
    contents = await file.read()
    size = len(contents)
    return {"filename": file.filename, "size_bytes": str(size)}
