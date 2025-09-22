"""Test that notebookize works with class methods."""

from notebookize import (
    _split_file_at_function,
    _rewrite_function_in_file,
)


def test_find_method_in_class(tmp_path):
    """Test that we can find and split at a method inside a class."""
    source_content = '''class MyClass:
    def __init__(self):
        self.value = 0
    
    def evaluate(self, x):
        """Evaluate the function."""
        result = x * 2
        self.value = result
        return result
    
    def another_method(self):
        return self.value
'''

    source_file = tmp_path / "test_class.py"
    source_file.write_text(source_content)

    # Try to find and split at the evaluate method
    before, body, after, indent = _split_file_at_function(str(source_file), "evaluate")

    # Verify the body was extracted correctly
    assert "Evaluate the function" in body
    assert "result = x * 2" in body
    assert "self.value = result" in body
    assert "return result" in body


def test_rewrite_method_in_class(tmp_path):
    """Test that we can rewrite a method inside a class."""
    source_content = '''class MyClass:
    def __init__(self):
        self.value = 0
    
    def evaluate(self, x):
        """Evaluate the function."""
        result = x * 2
        self.value = result
        return result
    
    def another_method(self):
        return self.value
'''

    source_file = tmp_path / "test_class.py"
    source_file.write_text(source_content)

    new_body = '''"""Updated evaluation."""
result = x * 3  # Changed multiplier
self.value = result
return result'''

    # Rewrite the evaluate method
    success = _rewrite_function_in_file(str(source_file), "evaluate", new_body)
    assert success

    # Verify the method was rewritten
    modified_content = source_file.read_text()
    assert "result = x * 3" in modified_content
    assert "Updated evaluation" in modified_content
    assert "result = x * 2" not in modified_content


def test_nested_class_method(tmp_path):
    """Test finding methods in nested classes."""
    source_content = """class OuterClass:
    class InnerClass:
        def nested_method(self):
            return 42
    
    def outer_method(self):
        return self.InnerClass()
"""

    source_file = tmp_path / "test_nested.py"
    source_file.write_text(source_content)

    # Try to find the nested method - it should work now
    before, body, after, indent = _split_file_at_function(
        str(source_file), "nested_method"
    )

    # Verify we found it
    assert "return 42" in body


def test_static_and_class_methods(tmp_path):
    """Test finding static methods and class methods."""
    source_content = """class MyClass:
    @staticmethod
    def static_method(x):
        return x * 2
    
    @classmethod
    def class_method(cls, x):
        return x * 3
    
    def instance_method(self, x):
        return x * 4
"""

    source_file = tmp_path / "test_decorators.py"
    source_file.write_text(source_content)

    # All of these should work now
    for method_name in ["static_method", "class_method", "instance_method"]:
        before, body, after, indent = _split_file_at_function(
            str(source_file), method_name
        )

        # Verify we found the method
        if method_name == "static_method":
            assert "return x * 2" in body
        elif method_name == "class_method":
            assert "return x * 3" in body
        else:  # instance_method
            assert "return x * 4" in body
