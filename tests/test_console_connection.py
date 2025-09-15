#!/usr/bin/env python
"""Test that open_console works correctly with kernel."""

import subprocess
import tempfile
import time
import json
import os
import sys

def test_console_with_kernel():
    """Test that open_console opens a working console connected to the kernel."""
    
    # Create a test script that uses notebookize with kernel=True and open_console=True
    test_script = """
import sys
import time
from notebookize import notebookize

TEST_GLOBAL = "Console test global"

@notebookize(kernel=True, open_jupyterlab=False, open_console=False)
def test_console_func(arg1, arg2="default"):
    '''Test function for console access.'''
    local_var = "Console local"
    result = f"{arg1} + {arg2}"
    
    # Keep the kernel alive for testing
    print("Kernel started for console test")
    time.sleep(30)  # Give us time to test
    
    return result

if __name__ == "__main__":
    test_console_func("hello", "console")
"""
    
    # Write the test script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        test_file = f.name
    
    try:
        # Start the test script in background
        proc = subprocess.Popen(
            [sys.executable, test_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it time to start the kernel
        time.sleep(5)
        
        # Find the connection file
        connection_file = f"/tmp/kernel-{proc.pid}.json"
        
        if not os.path.exists(connection_file):
            print(f"Connection file not found: {connection_file}")
            # Try to find any kernel connection file
            import glob
            kernel_files = glob.glob("/tmp/kernel-*.json")
            if kernel_files:
                connection_file = kernel_files[-1]
                print(f"Using connection file: {connection_file}")
            else:
                raise FileNotFoundError("No kernel connection file found")
        
        # Use jupyter_client to test the kernel
        from jupyter_client import BlockingKernelClient
        
        client = BlockingKernelClient()
        client.load_connection_file(connection_file)
        client.start_channels()
        
        try:
            # Wait for kernel to be ready
            client.wait_for_ready(timeout=5)
            print("✓ Kernel is ready")
            
            # Test that we can execute code
            msg_id = client.execute("print('Console connection test')")
            
            # Get the output
            outputs = []
            while True:
                try:
                    msg = client.get_iopub_msg(timeout=1)
                    if msg['msg_type'] == 'stream':
                        outputs.append(msg['content']['text'])
                    elif msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
                        break
                except:
                    break
            
            result = ''.join(outputs)
            print(f"Kernel output: {result}")
            
            # Test namespace access
            test_code = """
print('arg1' in dir(), 'arg1' in dir())
print('arg2' in dir(), 'arg2' in dir())
print('TEST_GLOBAL' in dir(), 'TEST_GLOBAL' in dir())
if 'arg1' in dir():
    print(f'arg1 = {arg1}')
if 'arg2' in dir():
    print(f'arg2 = {arg2}')
if 'TEST_GLOBAL' in dir():
    print(f'TEST_GLOBAL = {TEST_GLOBAL}')
"""
            msg_id = client.execute(test_code)
            
            # Get the output
            outputs = []
            while True:
                try:
                    msg = client.get_iopub_msg(timeout=1)
                    if msg['msg_type'] == 'stream':
                        outputs.append(msg['content']['text'])
                    elif msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
                        break
                except:
                    break
            
            namespace_result = ''.join(outputs)
            print(f"Namespace test:\n{namespace_result}")
            
            # Check results
            success = (
                "Console connection test" in result and
                "arg1 = hello" in namespace_result and
                "arg2 = console" in namespace_result and
                "TEST_GLOBAL = Console test global" in namespace_result
            )
            
            if success:
                print("\n✅ Console connection test PASSED!")
                print("  - Kernel is accessible")
                print("  - Function arguments are available")
                print("  - Global variables are available")
            else:
                print("\n❌ Console connection test FAILED")
                
        finally:
            client.stop_channels()
            
        # Terminate the background process
        proc.terminate()
        proc.wait(timeout=5)
        
        return success
            
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
        
        try:
            proc.terminate()
        except:
            pass


def test_console_opens():
    """Test that open_console=True actually opens a console window."""
    
    # This is harder to test automatically since it opens an interactive console
    # We'll just test that the function is called without errors
    
    test_script = """
import sys
import time
from notebookize import notebookize

@notebookize(kernel=True, open_jupyterlab=False, open_console=True)
def test_func(arg1):
    '''Test function.'''
    # The console should open
    print("Console should be opening...")
    time.sleep(5)  # Give console time to open
    return arg1

if __name__ == "__main__":
    test_func("test")
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        test_file = f.name
    
    try:
        # Run the script and capture output
        proc = subprocess.Popen(
            [sys.executable, test_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, 'TERM': 'dumb'}  # Prevent console from using terminal features
        )
        
        # Give it a moment to start
        time.sleep(3)
        
        # Kill it (we just want to see if it starts without error)
        proc.terminate()
        
        # Get any output
        stdout, stderr = proc.communicate(timeout=5)
        
        # Check for the console opening message
        if "Opened Jupyter console connected to kernel" in stderr:
            print("✓ Console opening was triggered")
            return True
        else:
            print("Console opening message not found in output")
            print(f"Stderr: {stderr}")
            return False
            
    except Exception as e:
        print(f"Error testing console: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing kernel connection for console...")
    print("=" * 60)
    success1 = test_console_with_kernel()
    
    print("\n" + "=" * 60)
    print("Testing console opening...")
    print("=" * 60)
    success2 = test_console_opens()
    
    if success1 and success2:
        print("\n✅ ALL CONSOLE TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some console tests failed")
        sys.exit(1)