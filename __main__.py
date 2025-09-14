"""Entry point for python -m notebookize."""

from notebookize import _start_kernel_handler
import sys

if __name__ == "__main__":
    # Handle command-line invocation for kernel startup
    if len(sys.argv) >= 4 and sys.argv[1] == "start-kernel":
        port = int(sys.argv[2])
        connection_file = sys.argv[3]
        _start_kernel_handler(port, connection_file)
    else:
        print("Usage: python -m notebookize start-kernel <port> <connection_file>", file=sys.stderr)
        sys.exit(1)