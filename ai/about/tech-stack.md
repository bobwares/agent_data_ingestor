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
