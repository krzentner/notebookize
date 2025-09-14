"""Tests for kernel setup functionality."""

import sys
import os
import time
import subprocess
import json
import tempfile
from pathlib import Path

# Add parent directory to path to import notebookize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notebookize import notebookize


def test_kernel_registration(tmp_path, monkeypatch):
    """Test that kernel registration works."""
    monkeypatch.setenv("NOTEBOOKIZE_PATH", str(tmp_path))
    
    # Create a test Python file with kernel enabled
    source_content = '''from notebookize import notebookize

@notebookize(open_jupyterlab=False, kernel=True)
def test_kernel_func():
    """Test function with kernel."""
    x = 42
    return x
    
if __name__ == "__main__":
    test_kernel_func()
'''
    
    source_file = tmp_path / "test_kernel.py"
    source_file.write_text(source_content)
    
    # Run the script in a subprocess
    proc = subprocess.Popen(
        [sys.executable, str(source_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give it time to register the kernel
    time.sleep(2)
    
    # Check if kernel was registered
    result = subprocess.run(
        ["jupyter", "kernelspec", "list", "--json"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        kernels = json.loads(result.stdout)
        kernel_specs = kernels.get("kernelspecs", {})
        
        # Look for our kernel
        notebookize_kernels = [k for k in kernel_specs.keys() if k.startswith("notebookize-")]
        assert len(notebookize_kernels) > 0, "No notebookize kernel found"
        
        # Remember kernel name for cleanup
        kernel_name = notebookize_kernels[0]
        
        # Clean up - kill the subprocess
        proc.terminate()
        proc.wait(timeout=2)
        
        # Manually clean up the kernel since the process was terminated
        subprocess.run(
            ["jupyter", "kernelspec", "remove", kernel_name, "-f"],
            capture_output=True,
            text=True
        )
        
        # Wait a bit for cleanup
        time.sleep(0.5)
        
        # Verify kernel was unregistered
        result = subprocess.run(
            ["jupyter", "kernelspec", "list", "--json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            kernels = json.loads(result.stdout)
            kernel_specs = kernels.get("kernelspecs", {})
            remaining_kernels = [k for k in kernel_specs.keys() if k.startswith("notebookize-")]
            print(f"Original kernel: {kernel_name}")
            print(f"Remaining kernels: {remaining_kernels}")
            assert len(remaining_kernels) == 0, f"Kernel was not cleaned up: {remaining_kernels}"
    else:
        # If jupyter command fails, skip the test
        proc.terminate()
        proc.wait(timeout=2)
        import pytest
        pytest.skip("jupyter command not available")


def test_kernel_command_handler():
    """Test the start-kernel command handler."""
    # Create a temporary connection file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        connection_data = {
            "shell_port": 12345,
            "iopub_port": 12346,
            "stdin_port": 12347,
            "control_port": 12348,
            "hb_port": 12349,
            "ip": "127.0.0.1",
            "key": "test-key",
            "transport": "tcp",
            "signature_scheme": "hmac-sha256",
            "kernel_name": "test"
        }
        json.dump(connection_data, f)
        connection_file = f.name
    
    try:
        # Test that the command can be invoked (it will timeout, but that's expected)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "notebookize", "start-kernel", "99999", connection_file],
                capture_output=True,
                text=True,
                timeout=1
            )
            # If it doesn't timeout, check for error
            assert result.returncode != 0
            assert "Error" in result.stderr or "Timeout" in result.stderr
        except subprocess.TimeoutExpired:
            # Expected - the command tries to connect to a non-existent port
            pass
    finally:
        # Clean up
        os.unlink(connection_file)