BOOTSTRAP_PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
# Use the project venv when it exists, otherwise fall back to system python3.
PY ?= $(if $(wildcard $(BIN)/python),$(BIN)/python,python3)
MODEL ?= qwen2.5-1.5b

.PHONY: setup test test-slow eval-local clean format lint

setup:  ## Create a project-local venv and install deps (never system-wide)
	$(BOOTSTRAP_PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"

test:  ## Fast tests: imports + pure-Python logic, no model download
	$(PY) -m pytest -q

test-slow:  ## Includes tests that download weights / run inference
	$(PY) -m pytest -q -m slow

eval-local:  ## Run the toy cue eval on the local backend
	$(PY) scripts/run_eval_local.py --model $(MODEL) --out results/

format:
	$(PY) -m ruff format src scripts tests
	$(PY) -m ruff check --fix src scripts tests

lint:
	$(PY) -m ruff check src scripts tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
