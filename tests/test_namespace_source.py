#!/usr/bin/env python
"""Test that kernel gets namespace from function's module, not caller's module."""

import tempfile
import sys
from pathlib import Path


def test_namespace_from_function_module():
    """Test that the kernel namespace comes from the function's module."""

    # Create a module with a decorated function
    module_code = """
from notebookize import notebookize

# Module global that SHOULD be in kernel
MODULE_VAR = "correct_value"

@notebookize(open_jupyterlab=False)
def test_func(x=1):
    return x * 2
"""

    # Create a caller with different globals
    caller_code = """
import sys
sys.path.insert(0, "{temp_dir}")

# Caller global that should NOT be in kernel
CALLER_VAR = "wrong_value"

# Import and mock the kernel start
from unittest.mock import patch, MagicMock
with patch("notebookize._start_kernel_directly") as mock_start:
    mock_start.return_value = ("/tmp/test.json", MagicMock())
    
    # Import the module
    import test_module
    
    # Call the function
    result = test_module.test_func(5)
    
    # Check what was passed to _start_kernel_directly
    call_args = mock_start.call_args
    if call_args:
        func_name, logger, user_ns, user_module = call_args[0]
        
        # Check that MODULE_VAR is in the namespace
        if "MODULE_VAR" in user_ns:
            print("✓ MODULE_VAR found in namespace")
            print(f"  Value: {{user_ns['MODULE_VAR']}}")
        else:
            print("✗ MODULE_VAR not found in namespace")
        
        # Check that CALLER_VAR is NOT in the namespace
        if "CALLER_VAR" not in user_ns:
            print("✓ CALLER_VAR correctly NOT in namespace")
        else:
            print("✗ CALLER_VAR incorrectly in namespace")
            
        # Check that function args are in namespace
        if "x" in user_ns:
            print(f"✓ Function argument 'x' in namespace: {{user_ns['x']}}")
        else:
            print("✗ Function argument 'x' not in namespace")
            
        # Check the module
        if user_module and hasattr(user_module, "__name__"):
            print(f"✓ Module: {{user_module.__name__}}")
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Write module file
        module_path = Path(temp_dir) / "test_module.py"
        module_path.write_text(module_code)

        # Write and run caller
        caller_path = Path(temp_dir) / "test_caller.py"
        caller_path.write_text(caller_code.format(temp_dir=temp_dir))

        import subprocess

        result = subprocess.run(
            [sys.executable, str(caller_path)],
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)

        # Check the output
        assert "✓ MODULE_VAR found in namespace" in result.stdout
        assert "✓ CALLER_VAR correctly NOT in namespace" in result.stdout
        assert "✓ Function argument 'x' in namespace: 5" in result.stdout
        assert "✓ Module: test_module" in result.stdout

        print("\n✅ Test passed: Kernel gets namespace from function's module")


def test_demo_module_globals():
    """Test that demo.py provides GLOBAL_CONFIG to kernel."""

    test_script = """
import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

# Mock the kernel start to inspect namespace
from unittest.mock import patch, MagicMock
with patch("notebookize._start_kernel_directly") as mock_start:
    mock_start.return_value = ("/tmp/test.json", MagicMock())
    
    # Import demo
    from tests.demo import process_data, GLOBAL_CONFIG
    
    # Call the function
    try:
        result = process_data([1, 2, 3], multiplier=2)
    except:
        pass  # Expected since kernel is mocked
    
    # Check what namespace was passed
    if mock_start.called:
        call_args = mock_start.call_args[0]
        user_ns = call_args[2] if len(call_args) > 2 else None
        
        if user_ns:
            # Check for GLOBAL_CONFIG
            if "GLOBAL_CONFIG" in user_ns:
                print(f"✓ GLOBAL_CONFIG in kernel namespace: {user_ns['GLOBAL_CONFIG']}")
            else:
                print("✗ GLOBAL_CONFIG not in kernel namespace")
                
            # Check for function arguments
            if "data_list" in user_ns:
                print(f"✓ data_list in kernel namespace: {user_ns['data_list']}")
            if "multiplier" in user_ns:
                print(f"✓ multiplier in kernel namespace: {user_ns['multiplier']}")
"""

    # Run from repo root
    repo_root = Path(__file__).parent.parent

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        test_file = f.name

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )

        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)

        # Verify the output
        assert "✓ GLOBAL_CONFIG in kernel namespace:" in result.stdout
        assert (
            "'version': '1.0'" in result.stdout or '"version": "1.0"' in result.stdout
        )
        assert "✓ data_list in kernel namespace: [1, 2, 3]" in result.stdout
        assert "✓ multiplier in kernel namespace: 2" in result.stdout

        print("\n✅ Demo test passed: GLOBAL_CONFIG available in kernel")

    finally:
        import os

        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    print("Testing kernel namespace source...")
    print("=" * 60)

    test_namespace_from_function_module()
    print()
    test_demo_module_globals()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nVerified: Kernel namespace includes:")
    print("  • Globals from the decorated function's module")
    print("  • Function arguments with their values")
    print("  • NOT globals from the calling code")
