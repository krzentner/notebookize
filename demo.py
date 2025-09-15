# THIS FILE IS A COPY AND GITIGNORED
#!/usr/bin/env python
"""Demonstration of the kernel functionality."""

from notebookize import notebookize

# Global variable that the kernel can access
GLOBAL_CONFIG = {"version": "1.0", "debug": True}

@notebookize(open_jupyterlab=True, 
             open_console=False,
             kernel=True)
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
    # Call the function with some test data
    test_data = [1, 2, 3, 4, 5]
    output = process_data(test_data, multiplier=3)
    print(f"Output: {output}")
