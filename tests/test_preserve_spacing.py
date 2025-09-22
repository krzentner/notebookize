"""Test that exact spacing is preserved through notebook round-trip."""

from notebookize import _convert_to_percent_format, _extract_code_from_notebook
from pathlib import Path
import tempfile


def notebook_round_trip(body_source: str) -> str:
    """Simulate converting to notebook and back."""
    # Convert to cells
    cells = _convert_to_percent_format(body_source)
    
    # Create notebook content exactly like _generate_jupytext_notebook does
    content_parts = []
    
    # Add minimal header
    content_parts.append("# ---")
    content_parts.append("# jupyter:")
    content_parts.append("# ---")
    
    # Add cells - exactly like the real code does (no newline before # %%)
    for cell in cells:
        content_parts.append("# %%")
        content_parts.append(cell)
    
    # Join with newlines like the real code
    notebook_content = "\n".join(content_parts)
    
    # Write to temp file and extract back
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(notebook_content)
        temp_path = Path(f.name)
    
    try:
        # Extract code back from notebook
        extracted = _extract_code_from_notebook(temp_path)
        # Remove the trailing newline that _extract_code_from_notebook adds
        # to match the original input (which typically doesn't have one)
        if extracted.endswith('\n') and not body_source.endswith('\n'):
            extracted = extracted[:-1]
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