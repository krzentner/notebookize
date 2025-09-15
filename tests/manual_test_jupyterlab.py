#!/usr/bin/env python3
"""
Manual test for the notebookize decorator with JupyterLab opening.

This test will:
1. Create a notebook from the decorated function
2. Open it in JupyterLab
3. Watch for changes and sync them back to the source

To run this test:
    python manual_test_jupyterlab.py

Requirements:
    pip install jupyterlab jupytext
"""

import os
import tempfile
from notebookize import notebookize

# Set up a temporary directory for notebooks
temp_dir = tempfile.mkdtemp(prefix="notebookize_jupyterlab_test_")
os.environ["NOTEBOOKIZE_PATH"] = temp_dir

print(f"Notebooks will be saved to: {temp_dir}")
print("=" * 60)


@notebookize(open_jupyterlab=True)
def fibonacci_explorer():
    """Explore Fibonacci numbers interactively."""
    # Initialize the Fibonacci sequence
    fib = [0, 1]

    # Generate the first 20 Fibonacci numbers
    for i in range(18):
        next_num = fib[-1] + fib[-2]
        fib.append(next_num)

    # Display the sequence
    print("First 20 Fibonacci numbers:")
    for i, num in enumerate(fib):
        print(f"F({i}) = {num}")

    # Calculate the golden ratio approximation
    # The ratio of consecutive Fibonacci numbers approaches phi
    # Yoooooooo
    ratios = []
    for i in range(2, len(fib)):
        ratio = fib[i] / fib[i - 1]
        ratios.append(ratio)
        print(f"F({i})/F({i - 1}) = {ratio:.10f}")

    # The golden ratio
    import math

    phi = (1 + math.sqrt(5)) / 2
    print(f"\nGolden ratio (phi) = {phi:.10f}")

    # Check convergence
    last_ratio = ratios[-1]
    error = abs(last_ratio - phi)
    print(f"Final ratio: {last_ratio:.10f}")
    print(f"Error from phi: {error:.10e}")

    return fib, ratios


if __name__ == "__main__":
    print("Starting notebookize with JupyterLab...")
    print("This will:")
    print("1. Generate a .py notebook (jupytext percent format) from the function")
    print("2. Create a paired .ipynb file that stays in sync with the .py file")
    print("3. Open the .ipynb in JupyterLab (opens as notebook automatically)")
    print("4. Watch for changes in both .py and .ipynb files")
    print("5. Sync changes back to this source file")
    print("\nPress Ctrl+C to stop watching for changes")
    print("=" * 60)

    # This will open JupyterLab and start watching
    fibonacci_explorer()
