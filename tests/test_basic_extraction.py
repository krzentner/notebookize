"""Tests for basic function extraction without file watching."""

import sys
import os
import pytest
from pathlib import Path

# Add parent directory to path to import notebookize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notebookize import (
    extract_function_body,
    get_function_source_and_def_index,
    convert_to_percent_format,
    generate_jupytext_notebook
)


def test_extract_simple_function():
    """Test extracting a simple function body."""
    def simple_func():
        return 42
    
    body = extract_function_body(simple_func)
    assert body is not None
    assert "return 42" in body


def test_extract_function_with_comments():
    """Test extracting a function with comments."""
    def func_with_comments():
        # This is a comment
        x = 10  # inline comment
        # Another comment
        return x * 2
    
    body = extract_function_body(func_with_comments)
    assert body is not None
    assert "# This is a comment" in body
    assert "x = 10  # inline comment" in body
    assert "# Another comment" in body
    assert "return x * 2" in body


def test_extract_multiline_statement():
    """Test extracting a function with multi-line statements."""
    def multiline_func():
        return (
            1 + 2 + 3 +
            4 + 5 + 6
        )
    
    body = extract_function_body(multiline_func)
    assert body is not None
    assert "return (" in body
    assert "1 + 2 + 3 +" in body
    assert "4 + 5 + 6" in body
    assert ")" in body


def test_extract_function_with_docstring():
    """Test extracting a function with a docstring."""
    def func_with_docstring():
        """This is a docstring."""
        x = 100
        return x
    
    body = extract_function_body(func_with_docstring)
    assert body is not None
    assert '"""This is a docstring."""' in body
    assert "x = 100" in body
    assert "return x" in body


def test_convert_to_percent_format():
    """Test converting function body to percent format cells."""
    body_source = """# This is a comment
# Another line

x = 10
y = 20

# A second comment block

result = x + y
return result"""
    
    cells = convert_to_percent_format(body_source)
    
    # Should have code cells separated by blank lines
    assert len(cells) > 0
    
    # All cells should be strings (code cells) now - no markdown cells
    assert all(isinstance(c, str) for c in cells)
    
    # Should have at least 3 cells due to blank line separations
    assert len(cells) >= 3
    
    # Check that comments are preserved in code cells
    assert any("# This is a comment" in cell for cell in cells)
    assert any("# A second comment block" in cell for cell in cells)
    assert any("x = 10" in cell for cell in cells)
    assert any("result = x + y" in cell for cell in cells)


def test_notebook_generation(tmp_path, monkeypatch):
    """Test generating a jupytext notebook."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    
    body_source = """x = 42
y = x * 2
return y"""
    
    notebook_path = generate_jupytext_notebook("test_func", body_source)
    
    # Check that notebook was created
    assert notebook_path.exists()
    assert notebook_path.suffix == ".py"
    assert "test_func" in notebook_path.name
    
    # Check notebook content
    content = notebook_path.read_text()
    assert "jupytext" in content
    assert "format_name: percent" in content
    assert "# %%" in content
    assert "x = 42" in content
    assert "y = x * 2" in content
    assert "return y" in content