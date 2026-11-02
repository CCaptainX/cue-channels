BOOTSTRAP_PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
# Use the project venv when it exists, otherwise fall back to system python3.
PY ?= $(if $(wildcard $(BIN)/python),$(BIN)/python,python3)
MODEL ?= qwen2.5-1.5b

.PHONY: setup setup-exact test test-slow reproduce-h2 figures clean format lint

setup:  ## Create a project-local venv and install pinned deps (never system-wide)
	$(BOOTSTRAP_PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e . --no-deps

setup-exact:  ## Same, but with the full transitive lock (requirements-lock.txt)
	$(BOOTSTRAP_PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -r requirements-lock.txt
	$(BIN)/pip install -e . --no-deps

test:  ## Fast tests: imports + pure-Python logic, no model download
	$(PY) -m pytest -q

test-slow:  ## Includes tests that download weights / run inference
	$(PY) -m pytest -q -m slow

reproduce-h2:  ## Re-run the headline result into results/h2_repro and compare (~25 min on M4 Pro)
	PY=$(PY) bash scripts/reproduce_h2.sh results/h2_repro

figures:  ## Rebuild the write-up figures from committed metrics (writeup/figures/)
	$(PY) writeup/make_figures.py

format:
	$(PY) -m ruff format src scripts tests
	$(PY) -m ruff check --fix src scripts tests

lint:
	$(PY) -m ruff check src scripts tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
