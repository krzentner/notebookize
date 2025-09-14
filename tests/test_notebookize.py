import sys
import os
import pytest
import tempfile
from pathlib import Path

# Add parent directory to path to import notebookize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notebookize import notebookize


def test_simple_function(caplog, tmp_path, monkeypatch):
    """Test a simple function with single-line body."""
    # Use temp directory for notebooks
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    
    @notebookize
    def simple_func():
        return 42
    
    # Capture logging output
    with caplog.at_level("INFO"):
        result = simple_func()
    
    output = caplog.text
    assert "return 42" in output
    assert result == 42


def test_function_with_comments(caplog, tmp_path, monkeypatch):
    """Test that comments inside function are preserved."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    @notebookize
    def func_with_comments():
        # This is a comment
        x = 10  # inline comment
        # Another comment
        return x * 2
    
    with caplog.at_level("INFO"):
        result = func_with_comments()
    
    output = caplog.text
    assert "# This is a comment" in output
    assert "x = 10  # inline comment" in output
    assert "# Another comment" in output
    assert "return x * 2" in output
    assert result == 20


def test_multiline_statement(caplog, tmp_path, monkeypatch):
    """Test function with multi-line statement at the end."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    @notebookize
    def multiline_func():
        return (
            1 + 2 + 3 +
            4 + 5 + 6
        )
    
    with caplog.at_level("INFO"):
        result = multiline_func()
    
    output = caplog.text
    assert "return (" in output
    assert "1 + 2 + 3 +" in output
    assert "4 + 5 + 6" in output
    assert ")" in output
    assert result == 21


def test_multiple_decorators(caplog, tmp_path, monkeypatch):
    """Test function with multiple decorators."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    def another_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    
    @another_decorator
    @notebookize
    def decorated_func(x):
        y = x * 2
        return y + 1
    
    with caplog.at_level("INFO"):
        result = decorated_func(5)
    
    output = caplog.text
    assert "y = x * 2" in output
    assert "return y + 1" in output
    assert result == 11


def test_function_with_docstring(caplog, tmp_path, monkeypatch):
    """Test that docstrings are included in the body."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    @notebookize
    def func_with_docstring():
        """This is a docstring."""
        x = 100
        return x
    
    with caplog.at_level("INFO"):
        result = func_with_docstring()
    
    output = caplog.text
    assert '"""This is a docstring."""' in output
    assert "x = 100" in output
    assert "return x" in output
    assert result == 100


def test_nested_function(caplog, tmp_path, monkeypatch):
    """Test function with nested function definition."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    @notebookize
    def outer_func():
        def inner_func(x):
            return x * 2
        
        result = inner_func(5)
        return result + 10
    
    with caplog.at_level("INFO"):
        result = outer_func()
    
    output = caplog.text
    assert "def inner_func(x):" in output
    assert "return x * 2" in output
    assert "result = inner_func(5)" in output
    assert "return result + 10" in output
    assert result == 20


def test_function_with_blank_lines(caplog, tmp_path, monkeypatch):
    """Test that blank lines in function body are preserved."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    @notebookize
    def func_with_blanks():
        x = 1
        
        # Space above and below
        
        y = 2
        return x + y
    
    with caplog.at_level("INFO"):
        result = func_with_blanks()
    
    output = caplog.text
    lines = output.split('\n')
    # Check that blank lines exist (they'll be empty strings in the split)
    assert any(line == "" for line in lines)
    assert "x = 1" in output
    assert "y = 2" in output
    assert result == 3


def test_function_with_complex_indentation(caplog, tmp_path, monkeypatch):
    """Test function with complex control flow and indentation."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    @notebookize
    def complex_func(n):
        if n > 0:
            for i in range(n):
                if i % 2 == 0:
                    print(f"Even: {i}")
                else:
                    print(f"Odd: {i}")
        else:
            return "Invalid"
        return "Done"
    
    with caplog.at_level("INFO"):
        result = complex_func(3)
    
    output = caplog.text
    assert "if n > 0:" in output
    assert "for i in range(n):" in output
    assert "if i % 2 == 0:" in output
    assert 'print(f"Even: {i}")' in output
    assert 'print(f"Odd: {i}")' in output
    assert 'return "Done"' in output


def test_class_method(caplog, tmp_path, monkeypatch):
    """Test decorator on a class method."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    class TestClass:
        @notebookize
        def method(self, x):
            # Process the input
            result = x * 2 + 1
            return result
    
    obj = TestClass()
    with caplog.at_level("INFO"):
        result = obj.method(10)
    
    output = caplog.text
    assert "# Process the input" in output
    assert "result = x * 2 + 1" in output
    assert "return result" in output
    assert result == 21


def test_function_with_args_kwargs(caplog, tmp_path, monkeypatch):
    """Test function with *args and **kwargs."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    @notebookize
    def func_with_args(*args, **kwargs):
        total = sum(args)
        for key, value in kwargs.items():
            total += value
        return total
    
    with caplog.at_level("INFO"):
        result = func_with_args(1, 2, 3, a=4, b=5)
    
    output = caplog.text
    assert "total = sum(args)" in output
    assert "for key, value in kwargs.items():" in output
    assert "total += value" in output
    assert "return total" in output
    assert result == 15


def test_notebook_generation(caplog, tmp_path, monkeypatch):
    """Test that a jupytext .py notebook is generated."""
    # Set the notebook path to a temporary directory
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    
    @notebookize
    def test_func():
        x = 42
        y = x * 2
        return y
    
    with caplog.at_level("INFO"):
        result = test_func()
    
    # Check that function executed correctly
    assert result == 84
    
    # Check that a notebook was created
    notebooks = list(tmp_path.glob("*.py"))
    assert len(notebooks) == 1
    
    # Check notebook content
    notebook_content = notebooks[0].read_text()
    assert "jupytext" in notebook_content
    assert "format_name: percent" in notebook_content
    assert "test_func" in notebook_content
    assert "# %%" in notebook_content  # Check for percent format cells
    assert "x = 42" in notebook_content
    assert "y = x * 2" in notebook_content
    assert "return y" in notebook_content
    
    # Check log messages
    output = caplog.text
    assert "Notebook saved to:" in output
    assert str(notebooks[0]) in output


def test_notebook_with_cells(caplog, tmp_path, monkeypatch):
    """Test that blank lines and comments create separate cells."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    
    @notebookize
    def multi_cell_func():
        # This is a comment block
        # It should become a markdown cell
        
        x = 10
        y = 20
        
        # Another comment
        # With multiple lines
        
        result = x + y
        return result
    
    with caplog.at_level("INFO"):
        result = multi_cell_func()
    
    assert result == 30
    
    # Check notebook was created
    notebooks = list(tmp_path.glob("*.py"))
    assert len(notebooks) == 1
    
    # Check that cells are properly separated
    notebook_content = notebooks[0].read_text()
    
    # Should have markdown cells for comments
    assert "# %% [markdown]" in notebook_content
    assert "# This is a comment block" in notebook_content
    assert "# Another comment" in notebook_content
    
    # Should have code cells separated by # %%
    assert notebook_content.count("# %%") >= 3  # At least 3 code cells


