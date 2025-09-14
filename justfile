# notebookize development commands

# Default command - show available commands
default:
    @just --list

# Run all tests
test:
    uv run pytest tests/ -v

# Run basic extraction tests
test-basic:
    uv run pytest tests/test_basic_extraction.py -v

# Run file watching tests
test-watch:
    uv run pytest tests/test_file_watching.py -v

# Run manual test with JupyterLab (interactive)
test-manual:
    uv run python manual_test_jupyterlab.py

# Run type checking with mypy
lint:
    uv run mypy notebookize.py
    uv run ruff check --fix notebookize.py
    uv run ruff check --fix tests/*.py

# Clean up temporary files and caches
clean:
    rm -rf __pycache__ .pytest_cache .venv
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name ".coverage" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
