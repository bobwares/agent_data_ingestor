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
from tempfile import NamedTemporaryFile
from typing import Protocol
from openai.types.chat import ChatCompletion

import openai
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

    def invoke(self, input) -> ChatCompletion: ...


def _default_llm() -> ChatOpenAI:  # noqa: D401
    """Return a zero-temperature GPT-4o chat model instance."""
    return ChatOpenAI(model="gpt-4o", temperature=0)


# ───────────────────────────────────────────────────────────────
#  Helper: upload file via OpenAI Files API
# ───────────────────────────────────────────────────────────────
def _upload_to_openai(path: str | Path) -> str:
    """Upload *path* once and return the resulting `file_id`.

    The OpenAI Files API currently rejects CSV files for retrieval. If a
    ``.csv`` path is given, its contents are first copied into a temporary text
    file which is then uploaded instead.
    """
    upload_path = Path(path)
    tmp: NamedTemporaryFile | None = None
    if upload_path.suffix.lower() == ".csv":
        tmp = NamedTemporaryFile("w+", suffix=".txt", delete=False)
        tmp.write(upload_path.read_text())
        tmp.flush()
        upload_path = Path(tmp.name)

    client = openai.OpenAI()  # relies on OPENAI_API_KEY environment variable
    with upload_path.open("rb") as fh:
        resp = client.files.create(file=fh, purpose="assistants")

    if tmp is not None:
        Path(tmp.name).unlink(missing_ok=True)

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

    The model (default: ``gpt-4o``) receives the user-supplied *task*
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
    # NOTE: `file_search` requires the Assistants v2 beta header and
    # must use the beta chat completions endpoint.
    response = client.beta.chat.completions.create(
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {"type": "file", "file": {"file_id": file_id}},
                ],
            }
        ],
        extra_headers={"OpenAI-Beta": "assistants=v2"},
    )
    return response.choices[0].message.content or ""

