.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f -name "*.py[cod]" -exec rm -f {} +
	rm -rf .mypy_cache .pytest_cache data/output

lint:
	uv run flake8 .
	uv run python -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs