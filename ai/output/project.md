## AI Prompt Context Instructions
    - This file includes the current of the application.
    - Always include metadata header section for project at the top of each source code file.
    - Definition of Metadata header section:

```markdown
# LangChain PoC - Minimal Chat Example
# Package: {{package}}
# File: {{file name}}
# Version: 2.0.29
# Author: Bobwares
# Date: {{current date/ time}}
# Description: document the function of the code.
#

```

- Update version each time new code is generated.   
- create file version.md with updated version number and list of changes.
- follow code formatting standards:   PEP 8: E303 too many blank lines (2)
## About This Project

## Project Overview
* 
* **Name**: `Data Ingestor`
* **Version**: `1.0.0`
* **Description**: Ingest different data types and transform.
* **Author**: Bobwares ([bobwares@outlook.com](mailto:bobwares@outlook.com))



# Version History

## 1.0.0
- Initial proof-of-concept release.

## Technology Stack

### Application Layer

#### Language

* **Python** `^3.12`

  * Modern version with improved performance, type hinting, and async support.

---

### Core Dependencies

| Package             | Version spec | Purpose                                |
| ------------------- | ------------ | -------------------------------------- |
| `fastapi`           | `>=0.111.0`  | Web API framework                      |
| `uvicorn[standard]` | `>=0.29.0`   | ASGI server for FastAPI                |
| `pydantic`          | `>=2.7.1`    | Data validation & settings management  |
| `typer[all]`        | `>=0.12.3`   | Developer-friendly CLI                 |
| `pandas`            | `>=2.2.2`    | Data manipulation & analysis           |
| `python-dotenv`     | `>=1.0.1`    | Load environment variables from `.env` |
| `sqlalchemy`        | `>=2.0.30`   | Database ORM / SQL toolkit             |
| `langchain`         | `>=0.1.16`   | LLM application framework              |
| `langgraph`         | `>=0.0.38`   | Graph orchestration for LangChain      |
| `openpyxl`          | `>=3.1.2`    | Excel file support                     |
| `ruff`              | `>=0.4.2`    | Fast Python linter                     |
| `pytest`            | `>=8.2.1`    | Testing framework                      |

# Configuration Files Compilation

## Miscellaneous

### File: ./.gitignore

```bash
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
.venv/

# Distribution / packaging
build/
dist/
*.egg-info/

# IDE settings
.idea/
.vscode/

# Logs
*.log

# OS files
.DS_Store
```


# Source Code

## File: ./src/data_ingestor/__init__.py
```python
# Data Ingestor
# Package: data_ingestor
# File: __init__.py
# Version: 1.0.3
# Author: Bobwares
# Date: 2025-06-03 15:35
# Description: Package initialisation, public exports, and version constant.
#

__all__: list[str] = ["cli"]

__version__: str = "1.0.3"
```

## File: ./src/data_ingestor/api.py
```python
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
```

## File: ./src/data_ingestor/connectors/__init__.py
```python
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
```

## File: ./src/data_ingestor/connectors/base.py
```python
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
```

## File: ./src/data_ingestor/connectors/csv.py
```python
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
```

## File: ./src/data_ingestor/llm_bridge.py
```python
# Data Ingestor
# Package: data_ingestor
# File: llm_bridge.py
# Version: 1.1.2
# Author: Bobwares
# Date: 2025-06-03
# Description: Bridge functions that forward a local file to an LLM via
#              three strategies: inline prompt, retrieval QA, or OpenAI
#              file-upload + prompt reference.
#

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import openai
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# ───────────────────────────────────────────────────────────────
#  Internal helper types
# ───────────────────────────────────────────────────────────────
class _LLM(Protocol):
    """Subset of Chat model interface used in this module."""

    def invoke(self, input) -> "ChatCompletion": ...


def _default_llm() -> ChatOpenAI:  # noqa: D401
    """Return a zero-temperature GPT-4o mini chat model instance."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ───────────────────────────────────────────────────────────────
#  Helper: upload file via OpenAI Files API
# ───────────────────────────────────────────────────────────────
def _upload_to_openai(path: str | Path) -> str:
    """Upload *path* once and return the resulting `file_id`."""
    client = openai.OpenAI()  # relies on OPENAI_API_KEY environment variable
    with Path(path).open("rb") as fh:
        resp = client.files.create(file=fh, purpose="assistants")
    return resp.id  # e.g. file-abc123


# ───────────────────────────────────────────────────────────────
#  Strategy 1: INLINE PROMPT
# ───────────────────────────────────────────────────────────────
def inline_prompt(path: str | Path, task: str, llm: _LLM | None = None) -> str:
    """Embed the entire file as text inside the prompt (≤ ~100 KB)."""
    data = Path(path).expanduser().read_bytes()
    text = data.decode("utf-8", errors="ignore")

    template = PromptTemplate.from_template(
        "File content:\n```text\n{blob}\n```\n\n{task}"
    )
    response = (template | (llm or _default_llm())).invoke(
        {"blob": text, "task": task}
    )
    return response.content


# ───────────────────────────────────────────────────────────────
#  Strategy 2: RETRIEVAL-BASED QA
# ───────────────────────────────────────────────────────────────
def retrieval_qa(path: str | Path, query: str, llm: _LLM | None = None) -> str:
    """Chunk large text/PDF/CSV, embed, and answer *query* via retrieval."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        docs = CSVLoader(file_path=str(path)).load()
    elif ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        docs = TextLoader(str(path)).load()

    store = FAISS.from_documents(docs, OpenAIEmbeddings())
    chain = RetrievalQA.from_chain_type(
        llm=llm or _default_llm(),
        retriever=store.as_retriever(),
        chain_type="stuff",
    )
    result = chain.invoke({"query": query})
    return result["result"]


def upload_and_prompt(path: str | Path, task: str) -> str:
    """
    Upload a local file once to the OpenAI Files API, then reference its file
    identifier in a chat completion request that enables the `file_search`
    tool.

    The model (default: ``gpt-4o-mini``) receives the user-supplied *task*
    followed by the attached ``file_id`` and returns its answer.

    Parameters
    ----------
    path :
        Local file path to upload.
    task :
        Instruction the language model should carry out on the uploaded file.

    Returns
    -------
    str
        Content of the model’s response message.
    """
    file_id = _upload_to_openai(path)
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {"type": "file_id", "file_id": file_id},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""

```

## File: ./src/data_ingestor/main.py
```python
# Data Ingestor
# Package: data_ingestor
# File: main.py
# Version: 1.1.0
# Author: Bobwares
# Date: 2025-06-03 16:10
# Description: CLI – now has 'ingest' and 'send' commands.
#

from __future__ import annotations

from pathlib import Path

import typer

from data_ingestor.connectors import ConnectorFactory
from data_ingestor.llm_bridge import inline_prompt, retrieval_qa, upload_and_prompt
from data_ingestor.writers import WriterFactory

app = typer.Typer(add_completion=False)

# ──────────────────────────────────────────────────────────────────────────────
#  Ingest (existing)
# ──────────────────────────────────────────────────────────────────────────────
@app.command()
def ingest(
        source: str = typer.Argument(...),
        target: str = typer.Option("blob", "--target", "-t"),
        raw: bool = typer.Option(False, "--raw"),
) -> None:
    blob = ConnectorFactory.from_uri(source).read_bytes()
    writer = WriterFactory.from_uri(target)

    if hasattr(writer, "write_bytes"):
        writer.write_bytes(blob, raw=raw)  # type: ignore[attr-defined]
    else:
        writer.write(blob)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
#  NEW: Send file to LLM
# ──────────────────────────────────────────────────────────────────────────────
@app.command()
def send(  # noqa: D401
        file_path: Path = typer.Argument(..., exists=True, readable=True),
        mode: str = typer.Option(
            "inline",
            "--mode",
            "-m",
            help="inline | retrieval | upload",
        ),
        task: str = typer.Option(
            None,
            "--task",
            "-t",
            help="Instruction for inline/upload modes.",
        ),
        query: str = typer.Option(
            None,
            "--query",
            "-q",
            help="Question for retrieval mode.",
        ),
) -> None:
    if mode == "inline":
        if not task:
            typer.echo("`--task` is required for inline mode", err=True)
            raise typer.Exit(1)
        result = inline_prompt(file_path, task)
    elif mode == "retrieval":
        if not query:
            typer.echo("`--query` is required for retrieval mode", err=True)
            raise typer.Exit(1)
        result = retrieval_qa(file_path, query)
    elif mode == "upload":
        if not task:
            typer.echo("`--task` is required for upload mode", err=True)
            raise typer.Exit(1)
        result = upload_and_prompt(file_path, task)
    else:
        typer.echo(f"Unknown mode: {mode}", err=True)
        raise typer.Exit(1)

    typer.echo(result)


def cli() -> None:  # noqa: D401
    app()


if __name__ == "__main__":
    cli()
```

## File: ./src/data_ingestor/writers/__init__.py
```python
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
```

## File: ./src/data_ingestor/writers/base.py
```python
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
```

## File: ./src/data_ingestor/writers/blob.py
```python
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
```

## File: ./src/data_ingestor/writers/sqlite.py
```python
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
```

## File: ./src/data_ingestor/writers/stdout.py
```python
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
```


# Infrastructure-as-Code (IaC) Report

## File: ./test.sh

```bash
curl -F "file=@./data/data_1.csv" http://127.0.0.1:8000/ingest```

## File: ./venv/lib/python3.13/site-packages/tqdm/completion.sh

```bash
#!/usr/bin/env bash
_tqdm(){
  local cur prv
  cur="${COMP_WORDS[COMP_CWORD]}"
  prv="${COMP_WORDS[COMP_CWORD - 1]}"

  case ${prv} in
  --bar_format|--buf_size|--colour|--comppath|--delay|--delim|--desc|--initial|--lock_args|--manpath|--maxinterval|--mininterval|--miniters|--ncols|--nrows|--position|--postfix|--smoothing|--total|--unit|--unit_divisor)
    # await user input
    ;;
  "--log")
    COMPREPLY=($(compgen -W       'CRITICAL FATAL ERROR WARN WARNING INFO DEBUG NOTSET' -- ${cur}))
    ;;
  *)
    COMPREPLY=($(compgen -W '--ascii --bar_format --buf_size --bytes --colour --comppath --delay --delim --desc --disable --dynamic_ncols --help --initial --leave --lock_args --log --manpath --maxinterval --mininterval --miniters --ncols --nrows --null --position --postfix --smoothing --tee --total --unit --unit_divisor --unit_scale --update --update_to --version --write_bytes -h -v' -- ${cur}))
    ;;
  esac
}
complete -F _tqdm tqdm
```

