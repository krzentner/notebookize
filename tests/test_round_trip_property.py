"""Property-based tests for the round-trip invariant."""

from notebookize import _convert_to_percent_format


def test_round_trip_invariant_empty():
    """Test round-trip with empty string."""
    body = ""
    cells = _convert_to_percent_format(body)
    assert "\n".join(cells) == body


def test_round_trip_invariant_single_line():
    """Test round-trip with single line."""
    body = "x = 42"
    cells = _convert_to_percent_format(body)
    assert "\n".join(cells) == body


def test_round_trip_invariant_only_whitespace():
    """Test round-trip with only whitespace."""
    test_cases = [
        "\n",
        "\n\n",
        "\n\n\n",
        "   \n",
        "\n   \n",
    ]
    for body in test_cases:
        cells = _convert_to_percent_format(body)
        reconstructed = "\n".join(cells)
        assert reconstructed == body, f"Failed for {repr(body)}"


def test_round_trip_invariant_complex_docstrings():
    """Test round-trip with various docstring formats."""
    test_cases = [
        '"""Simple docstring."""',
        '"""Multi-line\ndocstring."""',
        '"""Docstring with\n\nblank line."""',
        '"""Docstring with\n\n\nmultiple blanks."""',
        '"""\nDocstring starting on new line.\n"""',
        '"""Docstring with\n    indentation\n        preserved."""',
        "'''Single quotes\n\ntoo.'''",
        '"""Mixed \'quotes\' and "quotes"."""',
    ]

    for body in test_cases:
        cells = _convert_to_percent_format(body)
        reconstructed = "\n".join(cells)
        assert reconstructed == body, f"Failed for docstring: {repr(body)}"


def test_round_trip_invariant_comments():
    """Test round-trip with various comment patterns."""
    test_cases = [
        "# Single comment",
        "# Comment 1\n# Comment 2",
        "# Comment\n\n# After blank",
        "# Comment\n\n\n# After double blank",
        "x = 1  # Inline comment",
        "# Comment\nx = 1  # Inline",
    ]

    for body in test_cases:
        cells = _convert_to_percent_format(body)
        reconstructed = "\n".join(cells)
        assert reconstructed == body, f"Failed for: {repr(body)}"


def test_round_trip_invariant_control_flow():
    """Test round-trip with control flow structures."""
    test_cases = [
        "if x:\n    y = 1",
        "if x:\n    y = 1\nelse:\n    y = 2",
        "for i in range(10):\n    print(i)",
        "while True:\n    break",
        "try:\n    x = 1\nexcept:\n    pass",
        "with open('file') as f:\n    data = f.read()",
    ]

    for body in test_cases:
        cells = _convert_to_percent_format(body)
        reconstructed = "\n".join(cells)
        assert reconstructed == body, f"Failed for: {repr(body)}"


def test_round_trip_invariant_nested_structures():
    """Test round-trip with nested functions and classes."""
    body = '''def outer():
    """Outer docstring."""
    
    def inner():
        """Inner docstring."""
        pass
    
    class NestedClass:
        """Class docstring."""
        
        def method(self):
            """Method docstring."""
            return 42
    
    return inner, NestedClass'''

    cells = _convert_to_percent_format(body)
    reconstructed = "\n".join(cells)
    assert reconstructed == body


def test_round_trip_invariant_mixed_content():
    """Test round-trip with realistic mixed content."""
    body = '''"""Main docstring with
multiple lines and

blank lines inside."""

import math
from typing import List

# Configuration
CONFIG = {"debug": True}


def process(data: List[float]) -> float:
    """Process data.
    
    Args:
        data: Input data
    
    Returns:
        Processed result
    """
    # Initialize
    result = 0.0
    
    # Process each item
    for item in data:
        if item > 0:
            result += math.sqrt(item)
        else:
            # Handle negative
            result -= abs(item)
    
    
    # Final calculation
    return result / len(data) if data else 0'''

    cells = _convert_to_percent_format(body)
    reconstructed = "\n".join(cells)
    assert reconstructed == body


def test_round_trip_invariant_edge_cases():
    """Test round-trip with edge cases."""
    test_cases = [
        # Trailing newline
        "x = 1\n",
        "x = 1\n\n",
        # Leading newline
        "\nx = 1",
        "\n\nx = 1",
        # Both
        "\nx = 1\n",
        # String literals that look like comments
        'x = "# Not a comment"',
        "x = '# Also not a comment'",
        'x = """# Not a\n# comment"""',
        # Escaped quotes in strings
        'x = "String with \\"quotes\\""',
        "x = 'String with \\'quotes\\''",
        # Raw strings
        'x = r"\\n is literal"',
        'x = r"""Raw\n\nmultiline"""',
        # F-strings
        'x = f"Value: {y}"',
        'x = f"""Multi\n{y}\nline"""',
    ]

    for body in test_cases:
        cells = _convert_to_percent_format(body)
        reconstructed = "\n".join(cells)
        assert reconstructed == body, f"Failed for edge case: {repr(body)}"


def test_round_trip_invariant_comprehensive():
    """Test that the invariant holds for all test cases."""
    # This is the key property: for ANY valid Python function body,
    # joining the cells with "\n" should exactly recreate the original
    import inspect

    # Test with this very function's body
    def sample_function():
        """A sample function."""
        x = 1

        y = 2

        z = 3
        return x + y + z

    import textwrap

    body = textwrap.dedent(inspect.getsource(sample_function).split(":", 1)[1]).lstrip(
        "\n"
    )
    cells = _convert_to_percent_format(body)
    reconstructed = "\n".join(cells)

    # The invariant must hold
    assert reconstructed == body, (
        f"Invariant violated:\nOriginal:\n{repr(body)}\nReconstructed:\n{repr(reconstructed)}"
    )
