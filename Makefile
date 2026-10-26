.PHONY: setup repos neo4j lint test data index eval clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"
	bash scripts/clone_repos.sh

repos:
	bash scripts/clone_repos.sh

neo4j:
	docker compose up -d

lint:
	ruff check src tests
	mypy src

test:
	pytest

data:
	python scripts/build_corpus.py

index:
	python scripts/build_index.py

eval:
	python -m litagent.eval.report
