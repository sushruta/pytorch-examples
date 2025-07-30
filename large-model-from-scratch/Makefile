.PHONY: install install-dev lint format type-check run

VENV_DIR := .venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

install:
	uv venv $(VENV_DIR)
	uv pip install -e .

install-dev:
	uv venv $(VENV_DIR)
	uv pip install -e .[dev]

format: install-dev
	uv run black .
	uv run ruff format .

type-check: install-dev
	uv run mypy .

