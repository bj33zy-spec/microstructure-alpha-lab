.PHONY: install test lint demo analyze clean

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests

demo:
	lobalpha demo --output data/synthetic_quotes.csv --rows 12000 --seed 7

analyze:
	lobalpha analyze --input data/synthetic_quotes.csv --output-dir reports/demo --tick-size 0.25 --horizon 20 --min-train-size 2000 --test-block-size 1000

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info reports/demo data/synthetic_quotes.csv
