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

# Run tests with coverage
test-cov:
    uv run pytest tests/ -v --cov=notebookize --cov-report=term-missing --cov-report=html

# Generate coverage report and show in terminal
coverage:
    uv run coverage run -m pytest tests/ -v
    uv run coverage report -m
    uv run coverage html
    @echo "HTML coverage report generated in htmlcov/index.html"

# Generate HTML coverage report
coverage-html:
    uv run coverage run -m pytest tests/ -v
    uv run coverage html
    @echo "Coverage report generated in htmlcov/"

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
    echo "# THIS FILE IS A COPY AND GITIGNORED" > demo.py
    cat tests/demo.py >> demo.py
    uv run python demo.py

# Clean up stale Jupyter kernels and sessions
clean-jupyter:
    uv run python scripts/cleanup_jupyter.py


# Run type checking with mypy
lint:
    uv run ruff format .
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
    # Clean up generated notebook files (with timestamp pattern)
    find . -type f -name "*_????????_??????_????????.jupytext.py" -delete
    find . -type f -name "*_????????_??????_????????.py" -delete

build:
    uv run flit build

publish: build
    uv run flit publish

# Clean up any leftover notebookize kernels and connection files
clean-kernels:
    #!/usr/bin/env bash
    echo "Cleaning up notebookize kernels and connection files..."

    # Remove notebookize kernelspecs
    uv run jupyter kernelspec list 2>/dev/null | grep "notebookize-" | sed 's/^[[:space:]]*//' | awk '{print $1}' | while read kernel; do
        echo "Removing kernelspec: $kernel"
        uv run jupyter kernelspec remove "$kernel" -f 2>/dev/null || true
    done

    # Remove external kernel connection directories
    for dir in /tmp/notebookize_kernels_*; do
        if [ -d "$dir" ]; then
            echo "Removing external kernel dir: $dir"
            rm -rf "$dir"
        fi
    done

    # Remove kernel connection files
    for file in /tmp/kernel-*.json; do
        if [ -f "$file" ]; then
            echo "Removing connection file: $file"
            rm -f "$file"
        fi
    done

    # Remove runtime connection files
    runtime_dir="$HOME/.local/share/jupyter/runtime"
    if [ -d "$runtime_dir" ]; then
        for file in "$runtime_dir"/kernel-*.json; do
            if [ -f "$file" ]; then
                echo "Removing runtime connection file: $file"
                rm -f "$file"
            fi
        done
    fi

    echo "Cleanup complete"
