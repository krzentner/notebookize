"""Tests for basic function extraction without file watching."""

import sys
import os

# Add parent directory to path to import notebookize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notebookize import (
    _extract_function_body,
    _convert_to_percent_format,
    _generate_jupytext_notebook,
)


def test_extract_simple_function():
    """Test extracting a simple function body."""

    def simple_func():
        return 42

    body = _extract_function_body(simple_func)
    assert body is not None
    assert "return 42" in body


def test_extract_function_with_comments():
    """Test extracting a function with comments."""

    def func_with_comments():
        # This is a comment
        x = 10  # inline comment
        # Another comment
        return x * 2

    body = _extract_function_body(func_with_comments)
    assert body is not None
    assert "# This is a comment" in body
    assert "x = 10  # inline comment" in body
    assert "# Another comment" in body
    assert "return x * 2" in body


def test_extract_multiline_statement():
    """Test extracting a function with multi-line statements."""

    def multiline_func():
        # fmt: off
        return (
            1 + 2 + 3 +
            4 + 5 + 6
        )
        # fmt: on

    body = _extract_function_body(multiline_func)
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

    body = _extract_function_body(func_with_docstring)
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


# A second comment block after 2 blanks

result = x + y
return result"""

    cells = _convert_to_percent_format(body_source)

    # Should have code cells
    assert len(cells) > 0

    # All cells should be strings
    assert all(isinstance(c, str) for c in cells)

    # Most importantly: round-trip should be exact
    reconstructed = "\n".join(cells)
    assert reconstructed == body_source, (
        f"Round-trip failed:\nOriginal:\n{repr(body_source)}\nReconstructed:\n{repr(reconstructed)}"
    )


def test_cell_markers_not_in_docstrings():
    """Test that cell markers are not inserted inside multi-line docstrings."""
    body_source = '''"""This is a multi-line
docstring that spans

multiple lines with blank lines.

It should remain as a single cell."""

x = 10
y = 20

def nested_func():
    """Another docstring."""
    return 42

result = nested_func()'''

    cells = _convert_to_percent_format(body_source)

    # Most importantly: round-trip should be exact
    reconstructed = "\n".join(cells)
    assert reconstructed == body_source, (
        f"Round-trip failed:\nOriginal:\n{repr(body_source)}\nReconstructed:\n{repr(reconstructed)}"
    )

    # Find the cell containing the multi-line docstring
    docstring_cell = None
    for cell in cells:
        if "This is a multi-line" in cell:
            docstring_cell = cell
            break

    assert docstring_cell is not None, "Docstring cell not found"

    # The entire docstring should be in one cell (not split despite blank lines)
    assert "multiple lines with blank lines" in docstring_cell
    assert "It should remain as a single cell" in docstring_cell
    assert '"""This is a multi-line' in docstring_cell
    assert 'It should remain as a single cell."""' in docstring_cell


def test_notebook_generation(tmp_path, monkeypatch):
    """Test generating a jupytext notebook."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))

    body_source = """x = 42
y = x * 2
return y"""

    # Use a test source file path
    test_source_file = "/tmp/test_source.py"
    notebook_path = _generate_jupytext_notebook(
        "test_func", body_source, test_source_file
    )

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
    # Check that return statement was transformed to _return assignment
    assert "_return = y" in content
