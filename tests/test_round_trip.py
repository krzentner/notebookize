"""Test round-trip preservation of code formatting."""

import tempfile
from pathlib import Path
from notebookize import _extract_function_body, _convert_to_percent_format


def test_round_trip_preserves_blank_lines():
    """Test that converting to cells and back preserves formatting exactly."""
    # Original body with specific formatting
    original_body = """# Process data
for item in data:
    result = item * multiplier
    results.append(result)


# The kernel can inspect and modify these variables
total = sum(results)

average = total / len(results) if results else 0


return {"results": results, "total": total, "average": average}"""

    # Convert to cells
    cells = _convert_to_percent_format(original_body)
    
    # Reconstruct body from cells - should be exact
    reconstructed_body = "\n".join(cells)
    
    # Should be perfectly preserved
    assert reconstructed_body == original_body


def test_round_trip_preserves_single_blank_line():
    """Test that single blank lines between statements are preserved."""
    original_body = """x = 1

y = 2

z = 3"""
    
    cells = _convert_to_percent_format(original_body)
    reconstructed = "\n".join(cells)
    
    assert reconstructed == original_body


def test_round_trip_preserves_multiple_blank_lines():
    """Test that multiple blank lines are preserved exactly."""
    original_body = """x = 1


# Comment after two blank lines
y = 2


z = 3"""
    
    cells = _convert_to_percent_format(original_body)
    reconstructed = "\n".join(cells)
    
    # Should preserve exact formatting including multiple blank lines
    assert reconstructed == original_body


def test_round_trip_with_docstring():
    """Test that docstrings with internal blank lines work correctly."""
    original_body = '''"""This is a docstring.

It has a blank line inside.
"""


x = 10'''
    
    cells = _convert_to_percent_format(original_body)
    reconstructed = "\n".join(cells)
    
    # Should preserve everything exactly, including blank lines in docstring
    assert reconstructed == original_body
    
    # Docstring should remain intact in one cell (not split despite internal blank)
    assert any('"""This is a docstring.\n\nIt has a blank line inside.\n"""' in cell for cell in cells)