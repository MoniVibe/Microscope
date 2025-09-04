SHELL := /bin/bash

.PHONY: bootstrap verify format lint types test clean

bootstrap:
	./scripts/bootstrap.sh

verify:
	./scripts/verify.sh

format:
	ruff format .

lint:
	ruff check .

types:
	mypy .

test:
	pytest -q -m 'not gpu' --cov=optics_sim --cov-report=xml

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache reports htmlcov coverage.xml dist build


