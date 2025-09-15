#!/usr/bin/env python
"""Demonstration of the kernel functionality."""

import sys
import argparse
from notebookize import notebookize

# Global variable that the kernel can access
GLOBAL_CONFIG = {"version": "1.0", "debug": True}

def process_data(data_list, multiplier=2):
    """Process data with access to function arguments in the kernel.

    The kernel will have access to:
    - data_list: the input list
    - multiplier: the multiplication factor
    - GLOBAL_CONFIG: the global configuration
    - All other globals from this module
    """
    # Process each item
    results = []
    for item in data_list:
        result = item * multiplier
        results.append(result)

    # The kernel can inspect and modify these variables
    total = sum(results)
    average = total / len(results) if results else 0

    return {
        "results": results,
        "total": total,
        "average": average
    }

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Notebookize demo with kernel support")
    parser.add_argument('--use-jupyterlab', action='store_true', 
                        help='Open JupyterLab with the notebook (default)')
    parser.add_argument('--use-console', action='store_true',
                        help='Open Jupyter console instead of JupyterLab')
    args = parser.parse_args()
    
    # Determine which interface to use
    use_jupyterlab = True  # Default
    use_console = False
    
    if args.use_console:
        use_jupyterlab = False
        use_console = True
    elif args.use_jupyterlab:
        use_jupyterlab = True
        use_console = False
    
    # Apply the decorator with the appropriate settings
    process_data_wrapped = notebookize(
        process_data,
        open_jupyterlab=use_jupyterlab,
        open_console=use_console,
        kernel=True
    )
    
    # Call the wrapped function with some test data
    test_data = [1, 2, 3, 4, 5]
    output = process_data_wrapped(test_data, multiplier=3)
    print(f"Output: {output}")
