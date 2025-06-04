PYTHON := python3
VENV_BIN := .venv/bin
PIP := $(VENV_BIN)/pip

.PHONY: venv install run api test lint clean

venv:
	$(PYTHON) -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(VENV_BIN)/python -m src.data_ingestor.main

api: install
	$(VENV_BIN)/uvicorn data_ingestor.api:app --reload

test: install
	$(VENV_BIN)/pytest -q

lint: install
	$(VENV_BIN)/ruff src

clean:
	rm -rf .venv build dist *.egg-info
