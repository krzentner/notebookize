# notebookize development commands

# Default command - show available commands
default:
    @just --list

# Run all tests
test:
    uv run pytest tests/ -v

# Run all tests including kernel connection test
test-all:
    uv run pytest tests/ -v
    @echo "All tests passed!"

# Run basic extraction tests
test-basic:
    uv run pytest tests/test_basic_extraction.py -v

# Run file watching tests
test-watch:
    uv run pytest tests/test_file_watching.py -v

# Run manual test with JupyterLab (interactive)
test-manual:
    echo "# THIS FILE IS A COPY AND GITIGNORED" >> manual_test_jupyterlab.py
    cat tests/manual_test_jupyterlab.py >> manual_test_jupyterlab.py
    uv run python manual_test_jupyterlab.py

# Run kernel demo
run-demo:
    echo "# THIS FILE IS A COPY AND GITIGNORED" >> demo.py
    cat tests/demo.py >> demo.py
    uv run python demo.py

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

# Clean up any leftover notebookize kernels
clean-kernels:
    @echo "Cleaning up notebookize kernels..."
    @uv run jupyter kernelspec list 2>/dev/null | grep notebookize | awk '{print $$1}' | while read kernel; do \
        echo "Removing kernel: $$kernel"; \
        uv run jupyter kernelspec remove "$$kernel" -f 2>/dev/null || true; \
    done
    @echo "Done cleaning kernels"
