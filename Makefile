# Common workflows. Run `make help` for the list.
.DEFAULT_GOAL := help
PYTHON ?= python
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help install dev-install lint format test test-fast cov audit train demo benchmark serve ui docker docker-run clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(VENV): ## Create the virtual environment
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(VENV) ## Install the package and its runtime dependencies
	$(BIN)/pip install -e .

dev-install: $(VENV) ## Install everything, including dev and optional extras
	$(BIN)/pip install -e ".[all]"

lint: ## Check style and imports
	$(BIN)/ruff check .

format: ## Auto-fix what can be fixed
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test: ## Run the full test suite
	$(BIN)/pytest

test-fast: ## Run everything except the end-to-end training tests
	$(BIN)/pytest -m "not slow"

cov: ## Run tests with a coverage report
	$(BIN)/pytest --cov --cov-report=term-missing --cov-report=html

audit: ## Quantify duplicate rows and leakage risk in the dataset
	$(BIN)/dp audit

train: ## Train on the real dataset and write bundle + reports
	$(BIN)/dp train

demo: ## Train on generated data, so the pipeline runs with no dataset present
	$(BIN)/dp train --synthetic --model rules

benchmark: ## Compare every model under the leakage-aware protocol
	$(BIN)/dp benchmark

serve: ## Run the REST API on http://127.0.0.1:8000
	$(BIN)/dp serve

ui: ## Run the Streamlit interface
	$(BIN)/streamlit run app/streamlit_app.py

docker: ## Build the container image
	docker build -t disease-predictor:latest .

docker-run: ## Run the API in a container on port 8000
	docker run --rm -p 8000:8000 disease-predictor:latest

clean: ## Remove caches, build output and generated reports
	rm -rf build dist htmlcov .pytest_cache .ruff_cache .coverage coverage.xml
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf reports/* artifacts/*
