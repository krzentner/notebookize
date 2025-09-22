"""Test that exact spacing is preserved through notebook round-trip."""

from notebookize import _convert_to_percent_format, _extract_code_from_notebook
from pathlib import Path
import tempfile


def notebook_round_trip(body_source: str) -> str:
    """Simulate converting to notebook and back."""
    # Convert to cells
    cells = _convert_to_percent_format(body_source)
    
    # Create a temporary notebook with these cells
    notebook_content = ["# ---", "# jupyter:", "# ---"]
    for cell in cells:
        notebook_content.append("\n# %%")
        notebook_content.append(cell)
    
    # Write to temp file and extract back
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('\n'.join(notebook_content))
        temp_path = Path(f.name)
    
    try:
        # Extract code back from notebook
        extracted = _extract_code_from_notebook(temp_path)
        return extracted
    finally:
        temp_path.unlink()


def test_preserve_single_blank_line_in_loop():
    """Test that single blank line after loop is preserved."""
    original = """for item in data:
    result = item * 2
    results.append(result)

# Comment after single blank"""
    
    result = notebook_round_trip(original)
    assert result == original, f"Lost blank line after loop\nOriginal:\n{repr(original)}\nResult:\n{repr(result)}"


def test_preserve_no_blank_between_comment_and_code():
    """Test that comment directly followed by code stays together."""
    original = """# Process data
for item in data:
    process(item)"""
    
    result = notebook_round_trip(original)
    assert result == original, f"Added unwanted blank\nOriginal:\n{repr(original)}\nResult:\n{repr(result)}"


def test_preserve_double_blank_lines():
    """Test that double blank lines are preserved."""
    original = """x = 1


# After two blanks
y = 2"""
    
    result = notebook_round_trip(original)
    assert result == original, f"Lost double blank\nOriginal:\n{repr(original)}\nResult:\n{repr(result)}"


def test_preserve_blank_after_assignment():
    """Test blank line after assignment is preserved."""
    original = """total = sum(results)

average = total / len(results)

return average"""
    
    result = notebook_round_trip(original)
    assert result == original, f"Lost blank lines\nOriginal:\n{repr(original)}\nResult:\n{repr(result)}"


def test_demo_py_exact_case():
    """Test the exact case from demo.py that's failing."""
    # This is the ACTUAL spacing from demo.py - it has DOUBLE blank lines!
    original = """# Process each item
results = []


for item in data_list:
    result = item * multiplier
    results.append(result)


# The kernel can inspect and modify these variables
total = sum(results)


average = total / len(results) if results else 0


return {"results": results, "total": total, "average": average}"""
    
    result = notebook_round_trip(original)
    
    # Double blanks are cell boundaries and are preserved as double blanks
    # This is correct behavior - we preserve the structure
    assert result == original, f"Spacing not preserved correctly\nOriginal:\n{repr(original)}\nResult:\n{repr(result)}"