#!/usr/bin/env python
"""Test that kernel connection with --existing actually works."""

import subprocess
import tempfile
import time
import json
import os
import sys

def test_kernel_with_existing():
    """Test that we can connect to the kernel with --existing and access the namespace."""
    
    # Create a test script that uses notebookize with kernel=True
    test_script = """
import sys
import time
from notebookize import notebookize

TEST_GLOBAL = "I am a global variable"

@notebookize(kernel=True, open_jupyterlab=False)
def test_func(arg1, arg2="default"):
    '''Test function with kernel support.'''
    local_var = "I am local"
    result = arg1 + " " + arg2
    
    # Keep the kernel alive for testing
    print("Kernel started, sleeping...")
    time.sleep(30)  # Give us time to connect and test
    
    return result

if __name__ == "__main__":
    test_func("hello", "world")
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
        
        # Find the connection file - it should be /tmp/kernel-{pid}.json
        connection_file = f"/tmp/kernel-{proc.pid}.json"
        
        if not os.path.exists(connection_file):
            # Try to find any kernel connection file
            import glob
            kernel_files = glob.glob("/tmp/kernel-*.json")
            if kernel_files:
                connection_file = kernel_files[-1]  # Use the most recent
                print(f"Using connection file: {connection_file}")
            else:
                raise FileNotFoundError("No kernel connection file found")
        
        # Read connection file to verify it exists and is valid
        with open(connection_file, 'r') as f:
            conn_info = json.load(f)
            print(f"Connection info: {conn_info}")
        
        # First, test if we can connect at all using jupyter_client
        print("\nTesting direct connection with jupyter_client...")
        from jupyter_client import BlockingKernelClient
        
        client = BlockingKernelClient()
        client.load_connection_file(connection_file)
        client.start_channels()
        
        # Wait for kernel to be ready
        client.wait_for_ready(timeout=5)
        print("✓ Kernel is ready and responding")
        
        # Execute a simple command to test
        msg_id = client.execute("print('KERNEL_IS_ALIVE')")
        
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
        
        print(f"Kernel output: {''.join(outputs)}")
        
        # Now test namespace
        print("\nTesting namespace access...")
        test_code = """
import sys
print('sys.argv:', sys.argv)
print('arg1' in dir(), 'arg1' in dir())
print('arg2' in dir(), 'arg2' in dir())
print('TEST_GLOBAL' in dir(), 'TEST_GLOBAL' in dir())
if 'arg1' in dir():
    print('arg1 =', arg1)
if 'arg2' in dir():
    print('arg2 =', arg2)
if 'TEST_GLOBAL' in dir():
    print('TEST_GLOBAL =', TEST_GLOBAL)
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
        
        result = ''.join(outputs)
        print(f"Namespace test output:\n{result}")
        
        client.stop_channels()
        
        # Create a test script to run in the kernel
        test_commands = """
import sys
import json

# Check sys.argv
print("sys.argv:", sys.argv)
print("sys.argv[0] ends with test file:", sys.argv[0].endswith('.py'))

# Check for function arguments
print("arg1 exists:", 'arg1' in dir())
print("arg2 exists:", 'arg2' in dir())
if 'arg1' in dir():
    print("arg1 value:", arg1)
if 'arg2' in dir():
    print("arg2 value:", arg2)

# Check for global variable
print("TEST_GLOBAL exists:", 'TEST_GLOBAL' in dir())
if 'TEST_GLOBAL' in dir():
    print("TEST_GLOBAL value:", TEST_GLOBAL)

# Check for local variable
print("local_var exists:", 'local_var' in dir())
if 'local_var' in dir():
    print("local_var value:", local_var)

# Check for function name
print("test_func exists:", 'test_func' in dir())

# Create a marker to show we ran
test_marker = "KERNEL_TEST_SUCCESSFUL"
print(test_marker)
"""
        
        # Use jupyter console to connect and run commands
        console_proc = subprocess.Popen(
            ["jupyter", "console", "--existing", connection_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send our test commands
        output, errors = console_proc.communicate(input=test_commands, timeout=10)
        
        print("Console output:")
        print(output)
        
        if errors:
            print("Console errors:")
            print(errors)
        
        # Check the results
        success = False
        has_args = False
        has_globals = False
        
        if "KERNEL_TEST_SUCCESSFUL" in output:
            success = True
            print("✓ Successfully connected to kernel")
        
        if "arg1 value: hello" in output and "arg2 value: world" in output:
            has_args = True
            print("✓ Function arguments are accessible")
        
        if "TEST_GLOBAL value: I am a global variable" in output:
            has_globals = True
            print("✓ Global variables are accessible")
        
        if "local_var value: I am local" in output:
            print("✓ Local variables are accessible")
        
        # Terminate the background process
        proc.terminate()
        proc.wait(timeout=5)
        
        # Report results
        if success and has_args and has_globals:
            print("\n✅ TEST PASSED: Kernel connection works correctly!")
            return True
        else:
            print("\n❌ TEST FAILED: Kernel connection issues detected")
            if not success:
                print("  - Could not connect to kernel")
            if not has_args:
                print("  - Function arguments not accessible")
            if not has_globals:
                print("  - Global variables not accessible")
            return False
            
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # Kill any remaining processes
        try:
            proc.terminate()
        except:
            pass

if __name__ == "__main__":
    success = test_kernel_with_existing()
    sys.exit(0 if success else 1)