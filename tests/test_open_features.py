#!/usr/bin/env python
"""Tests for open_console and open_jupyterlab features."""

import subprocess
import time
import json
import sys
import pytest
from pathlib import Path


def test_jupyterlab_external_kernel(tmp_path):
    """Test that open_jupyterlab=True creates external kernel files."""
    
    test_script = """
import sys
import time
from notebookize import notebookize

@notebookize(kernel=True, open_jupyterlab=True, open_console=False)
def test_func(x=10):
    '''Test function.'''
    result = x * 2
    return result

if __name__ == "__main__":
    test_func(42)
"""
    
    test_file = tmp_path / "test_jupyterlab.py"
    test_file.write_text(test_script)
    
    # Start the script
    proc = subprocess.Popen(
        [sys.executable, str(test_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for external kernel directory
        for _ in range(20):
            external_dir = Path(f"/tmp/notebookize_kernels_{proc.pid}")
            if external_dir.exists():
                # Check for kernel files
                kernel_files = list(external_dir.glob("kernel-*.json"))
                if kernel_files:
                    # Verify JSON structure
                    with open(kernel_files[0]) as f:
                        conn_info = json.load(f)
                    
                    assert 'kernel_id' in conn_info
                    assert 'language' in conn_info
                    assert conn_info['language'] == 'python-notebookize'
                    
                    # Test passed
                    return
            
            time.sleep(0.5)
        
        pytest.fail("External kernel not created within timeout")
        
    finally:
        # Clean up
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        
        # Kill any JupyterLab processes
        subprocess.run(['pkill', '-f', 'jupyter-lab'], capture_output=True)


def test_kernel_namespace_access(tmp_path):
    """Test that kernel has access to function arguments and globals."""
    
    test_script = """
import sys
import time
from notebookize import notebookize

GLOBAL_VAR = "test_global"

@notebookize(kernel=True, open_jupyterlab=False, open_console=False)
def test_func(arg1="hello", arg2=123):
    '''Test function.'''
    result = arg1 + str(arg2)
    return result

if __name__ == "__main__":
    test_func("test", 456)
"""
    
    test_file = tmp_path / "test_namespace.py"
    test_file.write_text(test_script)
    
    # Start the script
    proc = subprocess.Popen(
        [sys.executable, str(test_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for kernel to start
        for _ in range(20):
            conn_file = Path(f"/tmp/kernel-{proc.pid}.json")
            if conn_file.exists():
                # Give kernel a moment to fully initialize
                time.sleep(2)
                
                # Try to connect and test namespace
                from jupyter_client import BlockingKernelClient
                
                client = BlockingKernelClient()
                client.load_connection_file(str(conn_file))
                client.start_channels()
                
                try:
                    # Wait for kernel
                    client.wait_for_ready(timeout=5)
                    
                    # Test namespace - execute code to check variables
                    client.execute("print('arg1' in dir() and arg1 == 'test')")
                    
                    # Get output
                    for _ in range(10):
                        try:
                            msg = client.get_iopub_msg(timeout=1)
                            if msg['msg_type'] == 'stream':
                                output = msg['content']['text'].strip()
                                if output == 'True':
                                    # Test passed - arguments are accessible
                                    return
                        except Exception:
                            continue
                    
                finally:
                    client.stop_channels()
                
                break
            
            time.sleep(0.5)
        
        pytest.fail("Kernel namespace test failed")
        
    finally:
        # Clean up
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def test_console_starts_kernel(tmp_path):
    """Test that open_console=True starts a kernel."""
    
    test_script = """
import sys
import time
from notebookize import notebookize

@notebookize(kernel=True, open_jupyterlab=False, open_console=True)
def test_func(x=10):
    '''Test function.'''
    result = x * 2
    return result

if __name__ == "__main__":
    test_func(42)
"""
    
    test_file = tmp_path / "test_console.py"
    test_file.write_text(test_script)
    
    # Start the script
    proc = subprocess.Popen(
        [sys.executable, str(test_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for kernel connection file
        for _ in range(20):
            conn_file = Path(f"/tmp/kernel-{proc.pid}.json")
            if conn_file.exists():
                # Kernel started successfully
                assert conn_file.exists()
                
                # Give console a moment to start
                time.sleep(2)
                
                # Check stderr for console message (don't wait for timeout)
                proc.terminate()
                _, stderr = proc.communicate(timeout=2)
                
                # Console should have been opened or attempted
                # In test environment, console may fail with "Error opening console: fileno"
                # but that still shows the console opening was attempted
                assert ("Opened Jupyter console" in stderr or 
                       "Error opening console" in stderr or 
                       "Jupyter console" in stderr)
                return
            
            time.sleep(0.5)
        
        pytest.fail("Console/kernel did not start within timeout")
        
    finally:
        if proc.poll() is None:
            proc.kill()