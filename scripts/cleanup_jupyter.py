#!/usr/bin/env python
"""Clean up stale Jupyter kernels and sessions."""

import subprocess
import json
import os
from pathlib import Path


def cleanup_stale_kernels():
    """Remove references to non-existent kernels from running Jupyter servers."""
    runtime_dir = Path.home() / ".local/share/jupyter/runtime"

    # Find all server info files
    server_files = list(runtime_dir.glob("jpserver-*.json"))

    for server_file in server_files:
        try:
            with open(server_file) as f:
                server_info = json.load(f)

            port = server_info.get("port", 8888)
            token = server_info.get("token", "")

            # Try to list kernels using the Jupyter API
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/api/kernels?token={token}"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                try:
                    kernels = json.loads(result.stdout)
                    print(f"Server on port {port} has {len(kernels)} kernels")

                    # Could add logic here to clean up specific kernels
                    for kernel in kernels:
                        kernel_id = kernel.get("id")
                        kernel_name = kernel.get("name", "unknown")
                        print(f"  - Kernel {kernel_id} ({kernel_name})")

                        # If it's a notebookize kernel with no connection file, delete it
                        if "notebookize" in kernel_name:
                            connection_file = (
                                f"/tmp/kernel-{kernel['execution_state']}.json"
                                if "execution_state" in kernel
                                else None
                            )
                            if connection_file and not Path(connection_file).exists():
                                print(f"    Removing stale kernel {kernel_id}")
                                subprocess.run(
                                    [
                                        "curl",
                                        "-X",
                                        "DELETE",
                                        "-s",
                                        f"http://localhost:{port}/api/kernels/{kernel_id}?token={token}",
                                    ],
                                    capture_output=True,
                                )
                except json.JSONDecodeError:
                    print(f"Could not parse kernel list from server on port {port}")

        except Exception as e:
            print(f"Error processing {server_file}: {e}")


if __name__ == "__main__":
    cleanup_stale_kernels()
