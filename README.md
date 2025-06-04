# Data Ingestor – Starter Project

This repository bootstraps the **Data Ingestor** application described in the Statement of Work (SOW) and Domain Design Document (DDD).

## Features

* **PEP 8‑compliant** Python 3.12 code under `./src`.
* CLI stub (Typer) and HTTP API stub (FastAPI).
* Makefile targets for environment setup, running, linting, and testing.
* Requirements pinned in `requirements.txt`; install with **pip**.

## Quick Start

```bash
#!/bin/bash
# 1. Create a virtual environment
python3 -m venv .venv
source venv/bin/activate   

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run CLI help
python -m data_ingestor --help

# 4. Launch the API (hot‑reload)
uvicorn data_ingestor.api:app --reload
```

## Makefile Shortcuts

| Target | Purpose |
|--------|---------|
| `make venv`    | Create `.venv` directory |
| `make install` | Install Python dependencies |
| `make run`     | Run CLI example |
| `make api`     | Start FastAPI server (localhost:8000) |
| `make lint`    | Lint source code with *ruff* |
| `make test`    | Run `pytest` |
| `make clean`   | Remove venv and build artifacts |

## Next Steps

* Flesh out `SchemaDetectionService` under `src/data_ingestor/schema_detection`.
* Implement connectors, mappers, validators, and writers as plugins.
* Add persistence layer (PostgreSQL) and job orchestration.

---

> Generated 3 June 2025
