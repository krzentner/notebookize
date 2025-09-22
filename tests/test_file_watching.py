import sys
import os
import time
import subprocess

# Add parent directory to path to import notebookize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notebookize import (
    _extract_code_from_notebook,
    _rewrite_function_in_file,
    _split_file_at_function,
)


def test_extract_code_from_notebook(tmp_path):
    """Test extracting code from a jupytext notebook."""
    notebook_content = """# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Test Notebook

# %%
x = 10
y = 20

# %% [markdown]
# Some comment

# %%
result = x + y
print(result)
"""

    notebook_path = tmp_path / "test.py"
    notebook_path.write_text(notebook_content)

    extracted = _extract_code_from_notebook(notebook_path)

    # Check that code was extracted properly
    assert "x = 10" in extracted
    assert "y = 20" in extracted
    assert "result = x + y" in extracted
    assert "print(result)" in extracted


def test_split_file_at_function(tmp_path):
    """Test splitting a file at a function boundary."""
    source_content = """def hello():
    print("Hello")

def test_func():
    x = 1
    y = 2
    return x + y

def goodbye():
    print("Goodbye")
"""

    source_file = tmp_path / "test_source.py"
    source_file.write_text(source_content)

    # Split at test_func
    before, body, after, indent = _split_file_at_function(str(source_file), "test_func")

    # Check the parts
    assert "def hello():" in before
    assert 'print("Hello")' in before
    assert "def test_func():" in before

    assert "x = 1" in body
    assert "y = 2" in body
    assert "return x + y" in body

    assert "def goodbye():" in after
    assert 'print("Goodbye")' in after

    assert indent == "    "

    # Verify that concatenating gives back the original
    reconstructed = before + "\n" + body + "\n" + after
    assert reconstructed == source_content


def test_rewrite_function_in_file(tmp_path):
    """Test rewriting a function in a Python file."""
    source_content = """def hello():
    print("Hello")

def test_func():
    x = 1
    y = 2
    return x + y

def goodbye():
    print("Goodbye")
"""

    source_file = tmp_path / "test_source.py"
    source_file.write_text(source_content)

    new_body = """z = 100
return z * 2"""

    # Rewrite the test_func
    success = _rewrite_function_in_file(str(source_file), "test_func", new_body)
    assert success

    # Check the result
    modified_content = source_file.read_text()

    # Original functions should still be there
    assert "def hello():" in modified_content
    assert "def goodbye():" in modified_content
    assert 'print("Hello")' in modified_content
    assert 'print("Goodbye")' in modified_content

    # New body should be there
    assert "z = 100" in modified_content
    assert "return z * 2" in modified_content

    # Old body should be gone
    assert "x = 1" not in modified_content
    assert "y = 2" not in modified_content
    assert "return x + y" not in modified_content


def test_file_watching_integration(tmp_path, monkeypatch, caplog):
    """Test the complete file watching and rewriting flow."""
    # Set environment variables
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    monkeypatch.setenv(
        "NOTEBOOKIZE_CHECK_INTERVAL", "0.05"
    )  # Very fast checking for tests

    # Create a test Python file
    source_content = """from notebookize import notebookize

@notebookize(open_jupyterlab=False)
def my_function():
    # Original implementation
    x = 42
    return x
    
if __name__ == "__main__":
    my_function()
"""

    source_file = tmp_path / "test_script.py"
    source_file.write_text(source_content)

    # Run the script in a subprocess that we can kill
    proc = subprocess.Popen(
        [sys.executable, str(source_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Poll for notebook creation instead of fixed sleep
    notebook_path = None
    for _ in range(20):  # Try for up to 1 second (20 * 0.05s)
        notebooks = list(tmp_path.glob("my_function_*.py"))
        if notebooks:
            notebook_path = notebooks[0]
            break
        time.sleep(0.05)

    assert notebook_path is not None, "Notebook was not created in time"

    # Read the original notebook content
    original_notebook = notebook_path.read_text()
    assert "x = 42" in original_notebook

    # Modify the notebook
    modified_notebook = original_notebook.replace("x = 42", "x = 100\ny = 200")
    notebook_path.write_text(modified_notebook)

    # Poll for source file update instead of fixed sleep
    source_updated = False
    for _ in range(20):  # Try for up to 1 second (20 * 0.05s)
        current_source = source_file.read_text()
        if "x = 100" in current_source:
            source_updated = True
            break
        time.sleep(0.05)

    # Kill the subprocess
    proc.terminate()
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Check that the source file was updated
    assert source_updated, "Source file was not updated in time"
    updated_source = source_file.read_text()
    assert "x = 100" in updated_source
    assert "y = 200" in updated_source
    assert "x = 42" not in updated_source  # Old code should be gone


def test_notebook_cell_separation(tmp_path, monkeypatch):
    """Test that blank lines properly create cell separations."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))

    from notebookize import _generate_jupytext_notebook

    body_source = """x = 10


y = 20
z = x + y


return z"""

    # Generate notebook
    test_source_file = "/tmp/test_source.py"
    notebook_path = _generate_jupytext_notebook(
        "test_func", body_source, test_source_file
    )

    content = notebook_path.read_text()

    # With smart splitting, the return statement creates a separate cell
    assert (
        content.count("# %%") == 3
    )  # Should have 3 code cells (smart splitting creates more cells)

    # Check that no markdown cells or comments were added
    assert "# %% [markdown]" not in content
    assert "# Add your code here" not in content

    # Extract code back
    from notebookize import _extract_code_from_notebook

    extracted = _extract_code_from_notebook(notebook_path)

    # Should have all the code parts
    assert "x = 10" in extracted
    assert "y = 20" in extracted
    assert "z = x + y" in extracted
    assert "return z" in extracted


def test_notebook_modifications_with_comments(tmp_path, monkeypatch):
    """Test that notebook modifications including comments are written back correctly."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    monkeypatch.setenv("NOTEBOOKIZE_CHECK_INTERVAL", "0.05")  # Fast checking for tests

    # Create a test Python file with a function containing comments
    source_content = '''from notebookize import notebookize

@notebookize(open_jupyterlab=False)
def process_data():
    """Process data with comments."""
    # Initialize variables
    x = 10
    y = 20
    
    # Calculate the sum
    result = x + y
    
    # Print the result
    print(f"Result: {result}")
    
    return result
    
if __name__ == "__main__":
    process_data()
'''

    source_file = tmp_path / "test_comments.py"
    source_file.write_text(source_content)

    # Run the script in a subprocess to generate the notebook
    proc = subprocess.Popen(
        [sys.executable, str(source_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Poll for notebook creation instead of fixed sleep
    notebook_path = None
    for _ in range(20):  # Try for up to 1 second (20 * 0.05s)
        notebooks = list(tmp_path.glob("process_data_*.py"))
        if notebooks:
            notebook_path = notebooks[0]
            break
        time.sleep(0.05)

    assert notebook_path is not None, "Notebook was not created in time"

    # Read the original notebook content
    original_notebook = notebook_path.read_text()

    # Verify that original comments are present in the notebook
    assert "# Initialize variables" in original_notebook
    assert "# Calculate the sum" in original_notebook
    assert "# Print the result" in original_notebook

    # Modify the notebook: add new comments, modify existing ones, and change code
    modified_lines = []
    for line in original_notebook.split("\n"):
        if "# Initialize variables" in line:
            modified_lines.append("# Initialize variables with new values")
        elif "x = 10" in line:
            modified_lines.append("# Added comment: starting value")
            modified_lines.append("x = 100")
        elif "y = 20" in line:
            modified_lines.append("y = 200  # inline comment")
        elif "# Calculate the sum" in line:
            modified_lines.append("# Calculate the sum of x and y")
        elif "result = x + y" in line:
            modified_lines.append("# Perform addition")
            modified_lines.append("result = x + y")
            modified_lines.append("# Store in result variable")
        else:
            modified_lines.append(line)

    modified_notebook = "\n".join(modified_lines)
    notebook_path.write_text(modified_notebook)

    # Poll for source file update instead of fixed sleep
    source_updated = False
    for _ in range(20):  # Try for up to 1 second (20 * 0.05s)
        current_source = source_file.read_text()
        if "x = 100" in current_source:
            source_updated = True
            break
        time.sleep(0.05)

    # Kill the subprocess
    proc.terminate()
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Check that the source file was updated with all modifications
    assert source_updated, "Source file was not updated in time"
    updated_source = source_file.read_text()

    # Verify code changes
    assert "x = 100" in updated_source
    assert "y = 200" in updated_source
    # Check that old values are replaced (note: "x = 10" is substring of "x = 100")
    assert "x = 10\n" not in updated_source and "x = 10 " not in updated_source
    assert "y = 20\n" not in updated_source and "y = 20 " not in updated_source

    # Verify original comments were preserved/modified
    assert "# Initialize variables with new values" in updated_source
    assert "# Calculate the sum of x and y" in updated_source

    # Verify new comments were added
    assert "# Added comment: starting value" in updated_source
    assert "y = 200  # inline comment" in updated_source
    assert "# Perform addition" in updated_source
    assert "# Store in result variable" in updated_source

    # Verify the structure is maintained
    assert "def process_data():" in updated_source
    assert '"""Process data with comments."""' in updated_source
    assert "return result" in updated_source


def test_write_back_disabled(tmp_path, monkeypatch):
    """Test that write_back=False prevents source file updates."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    monkeypatch.setenv("NOTEBOOKIZE_CHECK_INTERVAL", "0.05")  # Fast checking for tests

    # Create a test Python file
    source_content = '''from notebookize import notebookize

@notebookize(open_jupyterlab=False, write_back=False)
def test_func():
    """Test function."""
    x = 42
    return x
    
if __name__ == "__main__":
    test_func()
'''

    source_file = tmp_path / "test_no_writeback.py"
    source_file.write_text(source_content)

    # Run the script in a subprocess
    proc = subprocess.Popen(
        [sys.executable, str(source_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Poll for notebook creation instead of fixed sleep
    notebook_path = None
    for _ in range(20):  # Try for up to 1 second (20 * 0.05s)
        notebooks = list(tmp_path.glob("test_func_*.py"))
        if notebooks:
            notebook_path = notebooks[0]
            break
        time.sleep(0.05)

    assert notebook_path is not None, "Notebook was not created in time"

    # Modify the notebook
    original_notebook = notebook_path.read_text()
    modified_notebook = original_notebook.replace("x = 42", "x = 100")
    notebook_path.write_text(modified_notebook)

    # Wait a bit to ensure watcher would have detected if write_back was enabled
    time.sleep(0.2)

    # Kill the subprocess
    proc.terminate()
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Check that the source file was NOT updated
    final_source = source_file.read_text()
    assert "x = 42" in final_source  # Original value should still be there
    assert "x = 100" not in final_source  # New value should NOT be there
