"""A decorator that prints the source code of functions when called."""

__version__ = "0.1.0"

import ast
import inspect
import functools
import textwrap
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime


def get_logger():
    """Get or create the notebookize logger."""
    logger = logging.getLogger("notebookize")
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def get_notebook_dir():
    """Get the directory for saving notebooks from environment variable or default."""
    notebook_dir = os.environ.get("NOTEBOOKIZE_PATH", "~/notebooks/")
    notebook_dir = os.path.expanduser(notebook_dir)
    return Path(notebook_dir)


def get_function_source_and_def_index(func):
    """
    Get the source lines of a function and find where the actual 
    function definition starts (skipping decorators).
    Returns (source_lines, func_def_index).
    """
    source_lines, _ = inspect.getsourcelines(func)
    
    # Find the index of the actual function definition line (skipping decorators)
    func_def_index = 0
    for i, line in enumerate(source_lines):
        if line.strip().startswith('def '):
            func_def_index = i
            break
    
    return source_lines, func_def_index


def parse_function_ast(source_lines, func_def_index, func_name):
    """Parse the function source and return the AST node for the function."""
    # Get source starting from the function definition
    source_from_def = ''.join(source_lines[func_def_index:])
    
    # Dedent the source to remove leading indentation for parsing
    dedented_source = textwrap.dedent(source_from_def)
    
    # Parse the source to get the AST
    tree = ast.parse(dedented_source)
    
    # Find the function definition node
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    
    return None


def get_function_body_bounds(func_node, source_lines):
    """
    Get the line bounds of the function body.
    Returns (first_body_line, last_body_line) relative to the parsed source.
    """
    if not func_node or not func_node.body:
        return None, None
    
    # The function signature ends with a colon, body starts on next line
    func_def_line = func_node.lineno
    first_body_line = func_def_line + 1
    last_body_line = func_node.body[-1].end_lineno
    
    return first_body_line, last_body_line


def extract_body_lines(source_lines, func_def_index, first_body_line, last_body_line):
    """Extract the actual body lines from the source."""
    # Adjust indices: offset by func_def_index since we parsed from there
    # and line numbers in AST are 1-indexed relative to the parsed source
    actual_start = func_def_index + first_body_line - 1
    actual_end = func_def_index + last_body_line
    return source_lines[actual_start:actual_end]


def dedent_body_lines(body_lines):
    """Remove common indentation from body lines while preserving relative indentation."""
    if not body_lines:
        return ""
    
    # Join lines and use textwrap.dedent to remove common leading whitespace
    body_text = ''.join(body_lines)
    return textwrap.dedent(body_text)


def convert_to_percent_format(body_source):
    """
    Convert function body source to jupytext percent format.
    Splits on blank lines and top-level comments to create cells.
    """
    lines = body_source.split('\n')
    cells = []
    current_cell = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a blank line
        if not line.strip():
            # If we have content in current cell, save it and start a new one
            if current_cell:
                cells.append('\n'.join(current_cell))
                current_cell = []
            i += 1
            continue
        
        # Check if this is a top-level comment block
        if line.strip().startswith('#'):
            # Save current cell if it has content
            if current_cell:
                cells.append('\n'.join(current_cell))
                current_cell = []
            
            # Collect all consecutive comment lines
            comment_lines = []
            while i < len(lines) and lines[i].strip().startswith('#'):
                comment_lines.append(lines[i])
                i += 1
            
            # Add comment block as a markdown cell
            if comment_lines:
                cells.append(('markdown', '\n'.join(comment_lines)))
            continue
        
        # Regular code line
        current_cell.append(line)
        i += 1
    
    # Add any remaining content
    if current_cell:
        cells.append('\n'.join(current_cell))
    
    return cells


def generate_jupytext_notebook(func_name, body_source):
    """
    Generate a jupytext .py percent format notebook from function source code.
    Returns the path to the generated notebook.
    """
    # Create notebook directory if it doesn't exist
    notebook_dir = get_notebook_dir()
    notebook_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{func_name}_{timestamp}_{unique_id}.py"
    notebook_path = notebook_dir / filename
    
    # Convert body source to cells
    cells = convert_to_percent_format(body_source)
    
    # Create the jupytext percent format content
    content_parts = []
    
    # Add header
    content_parts.append(f"""# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Function: {func_name}
#
# Generated from function `{func_name}` at {datetime.now().isoformat()}

# %% [markdown]
# ## Function Body
""")
    
    # Add cells
    for cell in cells:
        if isinstance(cell, tuple) and cell[0] == 'markdown':
            # Markdown cell (comments)
            content_parts.append("\n# %% [markdown]")
            content_parts.append(cell[1])
        else:
            # Code cell
            content_parts.append("\n# %%")
            content_parts.append(cell)
    
    # Add footer with execution cell
    content_parts.append("""
# %% [markdown]
# ## Execution
#
# You can add cells below to test and explore this function.

# %%
# Add your code here
""")
    
    content = '\n'.join(content_parts)
    
    # Write the notebook file
    notebook_path.write_text(content)
    
    return notebook_path


def extract_function_body(func):
    """
    Extract the body source code of a function.
    Returns the body source as a string or None if extraction fails.
    """
    # Get source lines and find where the function definition starts
    source_lines, func_def_index = get_function_source_and_def_index(func)
    
    # Parse AST to find function boundaries
    func_node = parse_function_ast(source_lines, func_def_index, func.__name__)
    
    if not func_node:
        return None
    
    # Get body line boundaries
    first_body_line, last_body_line = get_function_body_bounds(func_node, source_lines)
    
    if first_body_line is None or last_body_line is None:
        return None
    
    # Extract and dedent body lines
    body_lines = extract_body_lines(source_lines, func_def_index, first_body_line, last_body_line)
    return dedent_body_lines(body_lines)


def notebookize(func):
    """
    Decorator that prints the inner source code of a function when called
    and generates a jupytext markdown notebook.
    Uses AST to find function boundaries while preserving original formatting.
    """
    logger = get_logger()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract the function body
        body_source = extract_function_body(func)
        
        if body_source:
            logger.info("# Function body:")
            logger.info(body_source)
            
            # Generate jupytext notebook
            try:
                notebook_path = generate_jupytext_notebook(func.__name__, body_source)
                logger.info(f"# Notebook saved to: {notebook_path}")
            except Exception as e:
                logger.error(f"# Failed to generate notebook: {e}")
        else:
            logger.error(f"# Unable to extract function body for {func.__name__}")
        
        # Execute the original function
        return func(*args, **kwargs)
    
    return wrapper
