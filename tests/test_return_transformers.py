"""Test the return statement transformers."""

import pytest
from notebookize import (
    ReturnToAssignmentTransformer,
    AssignmentToReturnTransformer,
    _generate_jupytext_notebook,
    _extract_code_from_notebook,
)
import libcst as cst
from pathlib import Path


class TestReturnToAssignmentTransformer:
    """Test the transformer that converts return statements to _return assignments."""

    def test_simple_return_value(self):
        """Test return with a simple value."""
        code = "return 42"
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "_return = 42"

    def test_return_expression(self):
        """Test return with an expression."""
        code = "return x + y * 2"
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "_return = x + y * 2"

    def test_return_none_explicit(self):
        """Test return with explicit None."""
        code = "return None"
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "_return = None"

    def test_return_none_implicit(self):
        """Test bare return statement."""
        code = "return"
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "_return = None"

    def test_return_dict(self):
        """Test return with dictionary literal."""
        code = "return {'key': 'value', 'count': 123}"
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "_return = {'key': 'value', 'count': 123}"

    def test_return_list(self):
        """Test return with list literal."""
        code = "return [1, 2, 3, 4]"
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "_return = [1, 2, 3, 4]"

    def test_return_function_call(self):
        """Test return with function call."""
        code = "return calculate_result(x, y, z=10)"
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "_return = calculate_result(x, y, z=10)"

    def test_multiple_returns(self):
        """Test multiple return statements in conditional."""
        code = """if condition:
    return True
else:
    return False"""
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        expected = """if condition:
    _return = True
else:
    _return = False"""
        assert modified.code.strip() == expected

    def test_nested_return(self):
        """Test return in nested structure."""
        code = """for i in range(10):
    if i == 5:
        return i"""
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        expected = """for i in range(10):
    if i == 5:
        _return = i"""
        assert modified.code.strip() == expected

    def test_return_in_try_except(self):
        """Test return statements in try/except blocks."""
        code = """try:
    return process()
except Exception:
    return None"""
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        expected = """try:
    _return = process()
except Exception:
    _return = None"""
        assert modified.code.strip() == expected

    def test_preserves_comments(self):
        """Test that comments are preserved."""
        code = """# Important comment
return result  # inline comment"""
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert "# Important comment" in modified.code
        assert "# inline comment" in modified.code
        assert "_return = result" in modified.code

    def test_string_with_return_keyword(self):
        """Test that 'return' in strings is not affected."""
        code = """message = "Please return the item"
return message"""
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert '"Please return the item"' in modified.code
        assert "_return = message" in modified.code


class TestAssignmentToReturnTransformer:
    """Test the transformer that converts _return assignments back to return statements."""

    def test_simple_assignment_to_return(self):
        """Test _return = value becomes return value."""
        code = "_return = 42"
        tree = cst.parse_module(code)
        transformer = AssignmentToReturnTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "return 42"

    def test_assignment_expression_to_return(self):
        """Test _return = expression becomes return expression."""
        code = "_return = x + y * 2"
        tree = cst.parse_module(code)
        transformer = AssignmentToReturnTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "return x + y * 2"

    def test_assignment_none_to_return(self):
        """Test _return = None becomes bare return."""
        code = "_return = None"
        tree = cst.parse_module(code)
        transformer = AssignmentToReturnTransformer()
        modified = tree.visit(transformer)
        assert modified.code.strip() == "return"

    def test_preserves_other_assignments(self):
        """Test that other assignments are not affected."""
        code = """x = 10
y = 20
_result = x + y
_return = _result"""
        tree = cst.parse_module(code)
        transformer = AssignmentToReturnTransformer()
        modified = tree.visit(transformer)
        result = modified.code.strip()
        assert "x = 10" in result
        assert "y = 20" in result
        assert "_result = x + y" in result
        assert "return _result" in result

    def test_multiple_return_assignments(self):
        """Test multiple _return assignments in conditional."""
        code = """if condition:
    _return = True
else:
    _return = False"""
        tree = cst.parse_module(code)
        transformer = AssignmentToReturnTransformer()
        modified = tree.visit(transformer)
        expected = """if condition:
    return True
else:
    return False"""
        assert modified.code.strip() == expected

    def test_ignores_similar_names(self):
        """Test that similar variable names are not affected."""
        code = """_return_value = compute()
my_return = 100
_return = _return_value"""
        tree = cst.parse_module(code)
        transformer = AssignmentToReturnTransformer()
        modified = tree.visit(transformer)
        result = modified.code.strip()
        assert "_return_value = compute()" in result
        assert "my_return = 100" in result
        assert "return _return_value" in result


class TestRoundTripTransformation:
    """Test the complete round-trip transformation."""

    def test_simple_round_trip(self):
        """Test simple value round-trip."""
        original = "return 42"

        # Forward transform
        tree = cst.parse_module(original)
        forward = ReturnToAssignmentTransformer()
        intermediate = tree.visit(forward).code
        assert "_return = 42" in intermediate

        # Reverse transform
        tree2 = cst.parse_module(intermediate)
        reverse = AssignmentToReturnTransformer()
        final = tree2.visit(reverse).code.strip()
        assert final == original

    def test_complex_round_trip(self):
        """Test complex code round-trip."""
        original = """# Calculate result
x = 10
y = 20

if x > y:
    return x
else:
    return y"""

        # Forward transform
        tree = cst.parse_module(original)
        forward = ReturnToAssignmentTransformer()
        intermediate = tree.visit(forward).code
        assert "_return = x" in intermediate
        assert "_return = y" in intermediate

        # Reverse transform
        tree2 = cst.parse_module(intermediate)
        reverse = AssignmentToReturnTransformer()
        final = tree2.visit(reverse).code.strip()
        assert "return x" in final
        assert "return y" in final

    def test_notebook_generation_with_return(self, tmp_path, monkeypatch):
        """Test that notebook generation transforms return statements."""
        monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))

        body_source = """x = 10
y = 20
return x + y"""

        notebook_path = _generate_jupytext_notebook(
            "test_func", body_source, "/tmp/test.py"
        )

        content = notebook_path.read_text()
        # Should have transformed return to _return
        assert "_return = x + y" in content
        assert "return x + y" not in content

    def test_notebook_extraction_restores_return(self, tmp_path):
        """Test that extraction restores return statements."""
        # Create a notebook with _return assignment
        notebook_content = """# ---
# jupyter:
# ---
# %%
x = 10
y = 20
_return = x + y"""

        notebook_path = tmp_path / "test.py"
        notebook_path.write_text(notebook_content)

        extracted = _extract_code_from_notebook(notebook_path)
        # Should have restored return statement
        assert "return x + y" in extracted
        assert "_return =" not in extracted

    def test_full_pipeline_with_multiple_returns(self, tmp_path, monkeypatch):
        """Test the full pipeline with multiple return statements."""
        monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))

        body_source = """# Process data
if x > 0:
    result = x * 2
else:
    result = -x

# Return final result
return result"""

        # Generate notebook
        notebook_path = _generate_jupytext_notebook(
            "test_func", body_source, "/tmp/test.py"
        )

        content = notebook_path.read_text()
        # The top-level return should be transformed
        assert "_return = result" in content
        # Check that it's not a raw return statement (outside of comments)
        lines = content.split("\n")
        for line in lines:
            if not line.strip().startswith("#") and "return result" in line:
                assert False, f"Found untransformed return in line: {line}"

        # Extract and verify restoration
        extracted = _extract_code_from_notebook(notebook_path)
        assert "return result" in extracted
        assert "_return =" not in extracted


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_return_multiline_expression(self):
        """Test return with multiline expression."""
        code = """return (
    x + y +
    z
)"""
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert "_return = (" in modified.code
        assert "x + y +" in modified.code
        assert "z" in modified.code

    def test_return_with_await(self):
        """Test return with await expression."""
        code = "return await async_function()"
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert "_return = await async_function()" in modified.code

    def test_return_with_yield(self):
        """Test return in generator context."""
        code = """yield value
return final"""
        tree = cst.parse_module(code)
        transformer = ReturnToAssignmentTransformer()
        modified = tree.visit(transformer)
        assert "yield value" in modified.code
        assert "_return = final" in modified.code

    def test_empty_file(self):
        """Test transformers on empty file."""
        code = ""
        tree = cst.parse_module(code)

        forward = ReturnToAssignmentTransformer()
        modified1 = tree.visit(forward)
        assert modified1.code == ""

        reverse = AssignmentToReturnTransformer()
        modified2 = tree.visit(reverse)
        assert modified2.code == ""

    def test_only_comments(self):
        """Test transformers on file with only comments."""
        code = """# This is a comment
# Another comment"""
        tree = cst.parse_module(code)

        forward = ReturnToAssignmentTransformer()
        modified1 = tree.visit(forward)
        assert "# This is a comment" in modified1.code

        reverse = AssignmentToReturnTransformer()
        modified2 = tree.visit(reverse)
        assert "# This is a comment" in modified2.code
