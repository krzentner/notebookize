"""A decorator that prints the source code of functions when called."""

__version__ = "0.1.0"

import ast
import inspect
import functools
import textwrap
import logging
import os
import uuid
import time
import subprocess
import threading
import json
import tempfile
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable, Any, Optional, Tuple, List, Union, TypeVar, Dict

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore

try:
    from ipykernel.kernelapp import IPKernelApp
    from IPython.utils.frame import extract_module_locals
except ImportError:
    IPKernelApp = None  # type: ignore
    extract_module_locals = None  # type: ignore


def _get_logger() -> logging.Logger:
    """Get or create the notebookize logger."""
    logger = logging.getLogger("notebookize")

    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def _get_notebook_dir() -> Path:
    """Get the directory for saving notebooks from environment variable or default."""
    notebook_dir = os.environ.get("NOTEBOOKIZE_PATH", "~/notebooks/")
    notebook_dir = os.path.expanduser(notebook_dir)
    return Path(notebook_dir)


def _get_function_source_and_def_index(
    func: Callable[..., Any],
) -> Tuple[List[str], int]:
    """
    Get the source lines of a function and find where the actual
    function definition starts (skipping decorators).
    Returns (source_lines, func_def_index).
    """
    source_lines, _ = inspect.getsourcelines(func)

    # Find the index of the actual function definition line (skipping decorators)
    func_def_index = 0
    for i, line in enumerate(source_lines):
        if line.strip().startswith("def "):
            func_def_index = i
            break

    return source_lines, func_def_index


def _parse_function_ast(
    source_lines: List[str], func_def_index: int, func_name: str
) -> Optional[ast.FunctionDef]:
    """Parse the function source and return the AST node for the function."""
    # Get source starting from the function definition
    source_from_def = "".join(source_lines[func_def_index:])

    # Dedent the source to remove leading indentation for parsing
    dedented_source = textwrap.dedent(source_from_def)

    # Parse the source to get the AST
    tree = ast.parse(dedented_source)

    # Find the function definition node
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node

    return None


def _get_function_body_bounds(
    func_node: Optional[ast.FunctionDef], source_lines: List[str]
) -> Tuple[Optional[int], Optional[int]]:
    """
    Get the line bounds of the function body.
    Returns (first_body_line, last_body_line) relative to the parsed source.
    """
    if not func_node or not func_node.body:
        return None, None

    # The function signature ends with a colon, body starts on next line
    func_def_line = func_node.lineno
    first_body_line = func_def_line + 1
    last_body_line = func_node.body[-1].end_lineno

    return first_body_line, last_body_line


def _extract_body_lines(
    source_lines: List[str],
    func_def_index: int,
    first_body_line: int,
    last_body_line: int,
) -> List[str]:
    """Extract the actual body lines from the source."""
    # Adjust indices: offset by func_def_index since we parsed from there
    # and line numbers in AST are 1-indexed relative to the parsed source
    actual_start = func_def_index + first_body_line - 1
    actual_end = func_def_index + last_body_line
    return source_lines[actual_start:actual_end]


def _dedent_body_lines(body_lines: List[str]) -> str:
    """Remove common indentation from body lines while preserving relative indentation."""
    if not body_lines:
        return ""

    # Join lines and use textwrap.dedent to remove common leading whitespace
    body_text = "".join(body_lines)
    return textwrap.dedent(body_text)


def _convert_to_percent_format(body_source: str) -> List[str]:
    """
    Convert function body source to jupytext percent format.
    Splits on blank lines to create cells. Comments are preserved in code cells.
    """
    lines = body_source.split("\n")
    cells: List[str] = []
    current_cell: List[str] = []

    for line in lines:
        # Check if this is a blank line
        if not line.strip():
            # If we have content in current cell, save it and start a new one
            if current_cell:
                cells.append("\n".join(current_cell))
                current_cell = []
            continue

        # Regular code line (including comments)
        current_cell.append(line)

    # Add any remaining content
    if current_cell:
        cells.append("\n".join(current_cell))

    return cells


def _generate_jupytext_notebook(func_name: str, body_source: str, kernel_name: Optional[str] = None) -> Path:
    """
    Generate a jupytext .py percent format notebook from function source code.
    Returns the path to the generated notebook.
    """
    import yaml
    
    # Create notebook directory if it doesn't exist
    notebook_dir = _get_notebook_dir()
    notebook_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{func_name}_{timestamp}_{unique_id}.py"
    notebook_path = notebook_dir / filename

    # Convert body source to cells
    cells = _convert_to_percent_format(body_source)

    # Create the jupytext percent format content
    content_parts: List[str] = []

    # Build the header metadata as a dictionary
    metadata = {
        "jupyter": {
            "jupytext": {
                "text_representation": {
                    "extension": ".py",
                    "format_name": "percent",
                    "format_version": "1.3",
                    "jupytext_version": "1.16.0"
                }
            }
        }
    }
    
    # Add kernel spec if kernel is enabled
    if kernel_name:
        pid = os.getpid()
        kernel_id = f"notebookize-{func_name.lower()}-{pid}"
        display_name = f"Notebookize: {func_name} (PID {pid})"
        metadata["jupyter"]["kernelspec"] = {
            "display_name": display_name,
            "language": "python",
            "name": kernel_id
        }
    else:
        metadata["jupyter"]["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    
    # Generate YAML header with proper formatting
    yaml_str = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    
    # Format as jupytext header comment
    header_lines = ["# ---"]
    for line in yaml_str.strip().split('\n'):
        header_lines.append(f"# {line}")
    header_lines.append("# ---")
    header = "\n".join(header_lines)
    
    # Log the header for debugging
    logger = _get_logger()
    logger.info(f"Jupytext header:\n{header}")
    
    content_parts.append(header)

    # Add cells - all are code cells now (including comments)
    for cell in cells:
        content_parts.append("\n# %%")
        content_parts.append(cell)

    content = "\n".join(content_parts)

    # Write the notebook file
    notebook_path.write_text(content)

    return notebook_path


def _extract_code_from_notebook(notebook_path: Path) -> str:
    """
    Extract code cells from a jupytext percent format notebook.
    Returns the combined code as a string.
    """
    content = notebook_path.read_text()
    lines = content.split("\n")

    code_parts: List[str] = []
    in_code_cell = False
    current_cell: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for cell markers
        if line.strip() == "# %%":
            # Start of a code cell
            if current_cell and in_code_cell:
                # Save previous code cell
                code_parts.append("\n".join(current_cell))
                current_cell = []
            in_code_cell = True
            i += 1
            continue
        elif line.strip() == "# %% [markdown]":
            # Start of a markdown cell
            if current_cell and in_code_cell:
                # Save previous code cell
                code_parts.append("\n".join(current_cell))
                current_cell = []
            in_code_cell = False
            i += 1
            continue

        # Collect lines if in code cell
        if in_code_cell:
            current_cell.append(line)

        i += 1

    # Add any remaining code cell
    if current_cell and in_code_cell:
        code_parts.append("\n".join(current_cell))

    # Combine all code parts, separated by blank lines
    filtered_parts = []
    for part in code_parts:
        if part.strip():
            filtered_parts.append(part.strip())

    return "\n\n".join(filtered_parts)


def _rewrite_function_in_file(file_path: str, func_name: str, new_body: str) -> bool:
    """
    Rewrite a function's body in a Python file while preserving everything else.
    Uses split_file_at_function for a much simpler implementation.
    """
    # Split the file into three parts
    before, old_body, after, indent = _split_file_at_function(file_path, func_name)

    # Prepare the new body with proper indentation
    new_body_lines = []
    for line in new_body.split("\n"):
        if line.strip():
            new_body_lines.append(indent + line)
        else:
            new_body_lines.append("")

    # Reconstruct the file
    new_body_indented = "\n".join(new_body_lines)
    new_content = before + "\n" + new_body_indented
    if after:
        new_content += "\n" + after

    # Write back to the file
    with open(file_path, "w") as f:
        f.write(new_content)

    return True


def _find_function_end_by_dedent(lines: List[str], body_start_line: int) -> int:
    """Find the end of a function by looking for dedentation."""
    if body_start_line >= len(lines):
        return body_start_line

    # Get the indentation of the first body line
    first_body_line = lines[body_start_line] if body_start_line < len(lines) else ""
    base_indent = (
        len(first_body_line) - len(first_body_line.lstrip())
        if first_body_line.strip()
        else 4
    )

    # Look for the next line with less indentation
    for i in range(body_start_line + 1, len(lines)):
        line = lines[i]
        if not line.strip():  # Skip empty lines
            continue

        line_indent = len(line) - len(line.lstrip())
        if line_indent < base_indent:
            return i - 1

    return len(lines) - 1


def _split_file_at_function(file_path: str, func_name: str) -> Tuple[str, str, str, str]:
    """
    Split a file into three parts: before function body, function body, and after function body.
    Returns (before, body, after, indent) where concatenating them gives the original file.
    """
    with open(file_path, "r") as f:
        original_content = f.read()

    # Parse the file to find the function
    tree = ast.parse(original_content)

    # Find the function node
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_node = node
            break

    if not func_node:
        raise ValueError(f"Function {func_name} not found in {file_path}")

    # Get the lines of the original file
    lines = original_content.split("\n")

    # Find where the function signature ends (look for the colon)
    func_start_line = func_node.lineno - 1  # Convert to 0-indexed
    func_def_end_line = func_start_line
    while func_def_end_line < len(lines) and not lines[
        func_def_end_line
    ].rstrip().endswith(":"):
        func_def_end_line += 1

    # The body starts on the next line
    body_start_line = func_def_end_line + 1

    # Find the end of the function body
    if not func_node.body:
        body_end_line = body_start_line
    else:
        # Use AST to find the last line
        ast_end_lineno = func_node.body[-1].end_lineno
        if ast_end_lineno is not None:
            body_end_line = ast_end_lineno - 1
        else:
            body_end_line = -1

        # Handle case where end_lineno is None
        if body_end_line == -1:
            body_end_line = _find_function_end_by_dedent(lines, body_start_line)

    # Get the indentation of the function body
    indent = ""
    for i in range(body_start_line, min(body_end_line + 1, len(lines))):
        if lines[i].strip():
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            break
    if not indent:
        indent = "    "

    # Split into three parts
    before_lines = lines[:body_start_line]
    body_lines = lines[body_start_line : body_end_line + 1]
    after_lines = lines[body_end_line + 1 :]

    before = "\n".join(before_lines)
    body = "\n".join(body_lines)
    after = "\n".join(after_lines) if after_lines else ""

    return before, body, after, indent


def _extract_function_body(func: Callable[..., Any]) -> Optional[str]:
    """
    Extract the body source code of a function.
    Returns the body source as a string or None if extraction fails.
    """
    # Get source lines and find where the function definition starts
    source_lines, func_def_index = _get_function_source_and_def_index(func)

    # Parse AST to find function boundaries
    func_node = _parse_function_ast(source_lines, func_def_index, func.__name__)

    if not func_node:
        return None

    # Get body line boundaries
    first_body_line, last_body_line = _get_function_body_bounds(func_node, source_lines)

    if first_body_line is None or last_body_line is None:
        return None

    # Extract and dedent body lines
    body_lines = _extract_body_lines(
        source_lines, func_def_index, first_body_line, last_body_line
    )
    return _dedent_body_lines(body_lines)


def _start_kernel_directly(func_name: str, logger: logging.Logger, 
                          user_ns: Optional[Dict[str, Any]] = None,
                          user_module: Optional[Any] = None) -> Tuple[Optional[str], Optional[Any]]:
    """Start an IPython kernel directly and return the connection file path and app instance.
    
    Returns:
        Tuple of (connection_file_path, kernel_app) or (None, None) if startup failed
    """
    try:
        from ipykernel.kernelapp import IPKernelApp
        import tempfile
        import os
        
        # Generate a unique connection file path (but don't create the file)
        temp_dir = tempfile.gettempdir()
        connection_file = os.path.join(temp_dir, f'kernel-{os.getpid()}.json')
        
        # Remove the file if it exists from a previous run
        if os.path.exists(connection_file):
            os.remove(connection_file)
        
        # Initialize the kernel app with the specific connection file
        app = IPKernelApp.instance()
        
        # Store the user namespace and module for later injection
        app._user_ns_to_inject = user_ns
        app._user_module_to_inject = user_module
        
        # We need to set up a custom initialization to inject our namespace
        original_init_kernel = app.init_kernel
        
        def custom_init_kernel():
            # Call the original initialization
            original_init_kernel()
            
            # Now inject our namespace after kernel is initialized
            if hasattr(app, '_user_ns_to_inject') and app._user_ns_to_inject:
                app.kernel.shell.user_ns.update(app._user_ns_to_inject)
                logger.info(f"Setting user namespace with {len(app._user_ns_to_inject)} variables")
            
            if hasattr(app, '_user_module_to_inject') and app._user_module_to_inject:
                app.kernel.shell.user_module = app._user_module_to_inject
                logger.info(f"Setting user module: {app._user_module_to_inject}")
        
        # Replace the init method
        app.init_kernel = custom_init_kernel
        
        # Pass the connection file and kernel name as command line arguments
        # Use PID in kernel name for uniqueness
        kernel_name = f"notebookize-{func_name}-{os.getpid()}"
        app.initialize([
            '--IPKernelApp.connection_file=' + connection_file,
            '--IPKernelApp.kernel_name=' + kernel_name
        ])
        logger.info(f"Kernel name: {kernel_name}")
        
        logger.info(f"Kernel initialized with connection file: {connection_file}")
        
        # Return both the connection file and the app
        # The app will be started in the main thread later
        return connection_file, app
        
    except Exception as e:
        logger.error(f"Failed to initialize kernel: {e}")
        return None, None


def _open_notebook_in_jupyterlab(notebook_path: Path, logger: logging.Logger, connection_file: Optional[str] = None) -> None:
    """Open the generated .py notebook directly in JupyterLab.
    
    Args:
        notebook_path: Path to the .py notebook file
        logger: Logger instance
        connection_file: Optional connection file to use with --existing
    """
    try:
        if connection_file:
            # Create external kernels directory in /tmp
            external_kernels_dir = Path("/tmp") / f"notebookize_kernels_{os.getpid()}"
            external_kernels_dir.mkdir(exist_ok=True)
            
            # Read and modify connection file to add kernel metadata
            import json
            
            with open(connection_file, 'r') as f:
                connection_info = json.load(f)
            
            # Add kernel metadata for better identification in JupyterLab
            # Extract function name from notebook path for context
            func_name = notebook_path.stem.split('_')[0] if '_' in notebook_path.stem else notebook_path.stem
            
            connection_info['kernel_name'] = f"Notebookize: {func_name}"
            connection_info['metadata'] = {
                'kernel_name': f"Notebookize: {func_name}",
                'display_name': f"Notebookize: {func_name} (PID {os.getpid()})"
            }
            
            # Write the enhanced connection file to external kernels directory
            external_connection_file = external_kernels_dir / os.path.basename(connection_file)
            with open(external_connection_file, 'w') as f:
                json.dump(connection_info, f, indent=2)
            
            logger.info(f"Created external kernel connection at: {external_connection_file}")
            
            # Open JupyterLab with external kernel support
            subprocess.Popen(
                [
                    "jupyter", "lab", str(notebook_path),
                    f"--ServerApp.external_connection_dir={external_kernels_dir}",
                    "--ServerApp.allow_external_kernels=True"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Opened JupyterLab with notebook: {notebook_path}")
            logger.info(f"Kernel '{func_name}' should be available in: Kernel > Change Kernel")
        else:
            # Open normally without kernel
            subprocess.Popen(
                ["jupyter", "lab", str(notebook_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Opened JupyterLab with notebook: {notebook_path}")
            
    except FileNotFoundError:
        logger.error(
            "JupyterLab not found. Please install with: pip install jupyterlab"
        )
    except Exception as e:
        logger.error(f"Error opening JupyterLab: {e}")


def _open_jupyter_console(connection_file: str, logger: logging.Logger) -> None:
    """Open Jupyter console connected to the running kernel.
    
    Args:
        connection_file: Path to the kernel connection file
        logger: Logger instance
    """
    try:
        import sys
        import os
        
        # On Unix systems, use script command to allocate a PTY
        # This avoids double-echo issues while keeping the console non-blocking
        if sys.platform != "win32":
            # Use 'script' command to allocate a PTY for the console
            # -q: quiet mode, -c: command to run
            # /dev/null: don't save typescript
            subprocess.Popen(
                ["script", "-q", "-c", f"jupyter console --existing {connection_file}", "/dev/null"],
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
        else:
            # On Windows, just run normally (no PTY issues)
            subprocess.Popen(
                ["jupyter", "console", "--existing", connection_file]
            )
        
        logger.info(f"Opened Jupyter console connected to kernel: {connection_file}")
    except FileNotFoundError as e:
        logger.error(
            f"Command not found: {e}. Please install jupyter-console with: pip install jupyter-console"
        )
    except Exception as e:
        logger.error(f"Error opening console: {e}")


def _extract_function_body_from_source(source_content: str, func_name: str) -> str:
    """Extract function body from source code content."""
    try:
        tree = ast.parse(source_content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                # Get the original source lines
                source_lines = source_content.splitlines()
                
                # Find function start and end
                func_start_line = node.lineno - 1  # Convert to 0-based
                
                # Find the last line of the function
                func_end_line = func_start_line
                for child in ast.walk(node):
                    if hasattr(child, 'lineno') and child.lineno:
                        func_end_line = max(func_end_line, child.lineno - 1)
                
                # Extract function lines (skip the def line)
                if func_start_line + 1 <= func_end_line:
                    func_lines = source_lines[func_start_line + 1:func_end_line + 1]
                    return '\n'.join(func_lines)
                
        return ""
    except Exception as e:
        return ""


def _handle_notebook_change(
    notebook_path: Path, source_file: str, func_name: str, logger: logging.Logger
) -> bool:
    """Handle a detected change in the notebook file."""
    logger.info(f"Notebook changed, updating {func_name} in {source_file}")

    # Extract code from the modified notebook
    new_body = _extract_code_from_notebook(notebook_path)

    if not new_body:
        logger.warning("No code found in notebook")
        return False

    # Get the current function body for diff
    try:
        with open(source_file, 'r') as f:
            source_content = f.read()
        old_body = _extract_function_body_from_source(source_content, func_name)
    except Exception as e:
        logger.warning(f"Could not read current function body for diff: {e}")
        old_body = ""

    # Show diff if we have both old and new content
    if old_body and old_body.strip() != new_body.strip():
        import difflib
        diff_lines = list(difflib.unified_diff(
            old_body.splitlines(keepends=True),
            new_body.splitlines(keepends=True),
            fromfile=f"{source_file}:{func_name} (before)",
            tofile=f"{source_file}:{func_name} (after)",
            lineterm=""
        ))
        
        if diff_lines:
            logger.info("Changes detected:")
            for line in diff_lines:
                logger.info(line.rstrip())
    elif old_body.strip() == new_body.strip():
        logger.info("No actual changes detected (content is identical)")

    # Rewrite the function in the source file
    success = _rewrite_function_in_file(source_file, func_name, new_body)

    if success:
        logger.info(f"Successfully updated {func_name}")
    else:
        logger.error(f"Failed to update {func_name}")

    return success


def _watch_notebook_for_changes(
    notebook_path: Path, source_file: str, func_name: str, logger: logging.Logger,
    write_back: bool = True
) -> None:
    """Watch notebook file for changes and optionally update the source file when detected."""
    logger.info(f"Watching {notebook_path} for changes...")
    
    if not write_back:
        logger.info("Note: write_back is disabled - changes will NOT be written to source file")

    logger.info("Press Ctrl+C to stop watching")

    last_mtime = notebook_path.stat().st_mtime
    check_interval = float(os.environ.get("NOTEBOOKIZE_CHECK_INTERVAL", "1.0"))

    try:
        while True:
            time.sleep(check_interval)

            try:
                current_mtime = notebook_path.stat().st_mtime
                
                if current_mtime > last_mtime:
                    last_mtime = current_mtime
                    logger.info("Change detected in notebook")
                    
                    if write_back:
                        _handle_notebook_change(
                            notebook_path, source_file, func_name, logger
                        )
                    else:
                        logger.info(f"Notebook {notebook_path.name} changed (write_back disabled)")

            except FileNotFoundError:
                logger.error(f"Notebook {notebook_path} was deleted")
                break
            except Exception as e:
                logger.error(f"Error checking notebook: {e}")

    except KeyboardInterrupt:
        logger.info("Stopped watching for changes")




def _unregister_kernel(kernel_name: str) -> None:
    """Unregister the kernel from Jupyter."""
    logger = _get_logger()
    
    try:
        subprocess.run(
            ["jupyter", "kernelspec", "remove", kernel_name, "-f"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Unregistered kernel: {kernel_name}")
    except subprocess.CalledProcessError:
        pass  # Kernel might not exist
    except FileNotFoundError:
        pass  # jupyter command not found


def _setup_kernel_if_requested(func_name: str, kernel_enabled: bool, logger: logging.Logger, 
                               user_ns: Optional[Dict[str, Any]] = None,
                               user_module: Optional[Any] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Set up a kernel for the notebook if requested.
    
    Args:
        func_name: Name of the decorated function
        kernel_enabled: Whether to enable kernel
        logger: Logger instance
        user_ns: User namespace (locals) to provide to the kernel
        user_module: Module to provide to the kernel
    
    Returns:
        Tuple of (kernel_name, kernel_info) where kernel_info contains connection details,
        or (None, None) if kernel not set up.
    """
    if not kernel_enabled:
        return None, None
    
    if IPKernelApp is None or extract_module_locals is None:
        logger.warning("ipykernel not installed, kernel mode disabled")
        return None, None
    
    # Find a free port for ZMQ communication
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    
    # Generate a unique kernel name
    kernel_id = str(uuid.uuid4())[:8]
    kernel_name = f"notebookize-{func_name}-{kernel_id}"
    
    # Create a temporary connection file with initial data
    connection_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    connection_file_path = connection_file.name
    
    # Write initial connection data (ports will be filled in by kernel)
    initial_connection_info = {
        "shell_port": 0,
        "iopub_port": 0,
        "stdin_port": 0,
        "control_port": 0,
        "hb_port": 0,
        "ip": "127.0.0.1",
        "key": str(uuid.uuid4()),
        "transport": "tcp",
        "signature_scheme": "hmac-sha256",
        "kernel_name": kernel_name
    }
    json.dump(initial_connection_info, connection_file)
    connection_file.close()
    
    # Create kernel.json for registration
    kernel_spec = {
        "argv": [
            sys.executable, "-m", "notebookize", 
            "start-kernel", str(port), "{connection_file}"
        ],
        "display_name": f"Notebookize - {func_name}",
        "language": "python",
        "metadata": {"notebookize": True}
    }
    
    # Register the kernel
    kernel_dir = Path(tempfile.mkdtemp(prefix="notebookize-kernel-"))
    kernel_json_path = kernel_dir / "kernel.json"
    with open(kernel_json_path, 'w') as f:
        json.dump(kernel_spec, f, indent=2)
    
    try:
        subprocess.run(
            ["jupyter", "kernelspec", "install", str(kernel_dir), "--name", kernel_name, "--user"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Registered kernel: {kernel_name}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to register kernel: {e}")
        return None, None
    
    # Return kernel info to be started in main thread
    kernel_info = {
        "name": kernel_name,
        "port": port,
        "connection_file": connection_file_path,
        "user_ns": user_ns,
        "user_module": user_module
    }
    
    return kernel_name, kernel_info


def _wait_for_connection_file(port: int, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """Wait for a connection file to be sent over ZMQ socket."""
    if zmq is None:
        return None
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{port}")
    
    logger = _get_logger()
    logger.info(f"Waiting for kernel connection on port {port}...")
    
    # Set timeout
    socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    
    try:
        # Wait for connection file data
        message = socket.recv_json()
        logger.info("Received kernel connection file")
        
        # Send acknowledgment
        socket.send_json({"status": "ok"})
        
        if isinstance(message, dict):
            return message
        else:
            logger.error(f"Invalid connection file format: {type(message)}")
            return None
    except zmq.error.Again:
        logger.error(f"Timeout waiting for kernel connection after {timeout} seconds")
        return None
    except Exception as e:
        logger.error(f"Error receiving connection file: {e}")
        return None
    finally:
        socket.close()
        context.term()


F = TypeVar("F", bound=Callable[..., Any])


def notebookize(
    func: Optional[F] = None, *, 
    open_jupyterlab: bool = True,
    open_console: bool = False,
    write_back: bool = True,
    kernel: bool = False
) -> Union[F, Callable[[F], F]]:
    """
    Decorator that generates a jupytext notebook and watches for changes,
    optionally updating the original function source when the notebook is modified.
    
    Args:
        func: The function to decorate
        open_jupyterlab: Whether to open the notebook in JupyterLab
        open_console: Whether to open Jupyter console connected to the kernel
        write_back: Whether to write changes back to the source file (default: True)
        kernel: Whether to set up a custom kernel for the notebook (default: False)
    """
    if func is None:
        return functools.partial(  # type: ignore[return-value]
            notebookize, open_jupyterlab=open_jupyterlab, open_console=open_console, 
            write_back=write_back, kernel=kernel
        )

    logger = _get_logger()

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
        # Get the source file path of the function
        try:
            source_file = inspect.getsourcefile(func)
            if not source_file:
                logger.error(f"Cannot determine source file for {func.__name__}")
                return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error getting source file: {e}")
            return func(*args, **kwargs)

        # Extract the function body
        body_source = _extract_function_body(func)

        if not body_source:
            logger.error(f"Unable to extract function body for {func.__name__}")
            return func(*args, **kwargs)

        logger.info(f"Original function body for {func.__name__}:")
        logger.info(body_source)

        # Set up kernel if requested 
        connection_file = None
        kernel_app = None
        if kernel:
            # Capture the calling frame's namespace
            user_ns = None
            user_module = None
            if extract_module_locals is not None:
                # Get the calling frame's module and locals
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_frame = frame.f_back
                    user_module = inspect.getmodule(caller_frame)
                    # Combine function arguments with caller's locals
                    user_ns = dict(caller_frame.f_locals)
                    # Add the function arguments
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    user_ns.update(bound_args.arguments)
            
            # Initialize kernel and get connection file
            connection_file, kernel_app = _start_kernel_directly(func.__name__, logger, user_ns, user_module)

        # Generate jupytext notebook with kernel name if kernel is enabled
        try:
            # Pass a kernel name if we have a kernel running
            kernel_name = "kernel" if kernel and connection_file else None
            notebook_path = _generate_jupytext_notebook(func.__name__, body_source, kernel_name)
            logger.info(f"Notebook saved to: {notebook_path}")
        except Exception as e:
            logger.error(f"Failed to generate notebook: {e}")
            return func(*args, **kwargs)
        
        # Open in JupyterLab if requested
        if open_jupyterlab:
            _open_notebook_in_jupyterlab(notebook_path, logger, connection_file)
        
        # Open console if requested (requires kernel)
        if open_console and connection_file:
            _open_jupyter_console(connection_file, logger)
        elif open_console and not connection_file:
            logger.warning("open_console requires kernel=True to be set")

        # If we have a kernel, we need to run it in the main thread
        if kernel_app and connection_file:
            # Start file watching in a background thread
            watch_thread = threading.Thread(
                target=_watch_notebook_for_changes,
                args=(notebook_path, source_file, func.__name__, logger, write_back),
                daemon=True
            )
            watch_thread.start()
            
            # Run the kernel in the main thread
            try:
                logger.info("Starting IPython kernel in main thread...")
                logger.info(f"Connection file: {connection_file}")
                logger.info("Kernel is ready for connections")
                kernel_app.start()  # This blocks until kernel is terminated
            except KeyboardInterrupt:
                logger.info("Kernel interrupted by user")
            except Exception as e:
                logger.error(f"Kernel error: {e}")
        else:
            # No kernel, just watch for changes normally
            try:
                _watch_notebook_for_changes(notebook_path, source_file, func.__name__, logger, write_back)
            except KeyboardInterrupt:
                logger.info("Watching interrupted by user")

        # Never actually call the original function
        logger.info(f"Watching stopped. Function {func.__name__} was not executed.")
        return None

    return wrapper  # type: ignore[return-value]


def _read_connection_file(connection_file: str) -> Dict[str, Any]:
    """Read and parse a kernel connection file."""
    try:
        with open(connection_file, 'r') as f:
            data = json.load(f)
            return data  # type: ignore[no-any-return]
    except Exception as e:
        print(f"Error reading connection file: {e}", file=sys.stderr)
        sys.exit(1)


def _send_connection_to_decorator(port: int, connection_data: Dict[str, Any]) -> None:
    """Send connection data to the decorator's ZMQ socket."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{port}")
    
    try:
        # Send connection file data
        socket.send_json(connection_data)
        
        # Wait for acknowledgment
        socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        reply = socket.recv_json()
        
        if not isinstance(reply, dict) or reply.get("status") != "ok":
            print(f"Unexpected reply: {reply}", file=sys.stderr)
            sys.exit(1)
            
        print(f"Successfully sent connection file to port {port}")
        
    except zmq.error.Again:
        print(f"Timeout waiting for acknowledgment from port {port}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error sending connection file: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        socket.close()
        context.term()


def _start_kernel_handler(port: int, connection_file: str) -> None:
    """Handler for the start-kernel command that sends connection file to the decorator."""
    if zmq is None:
        print("Error: pyzmq is not installed", file=sys.stderr)
        sys.exit(1)
    
    connection_data = _read_connection_file(connection_file)
    _send_connection_to_decorator(port, connection_data)


if __name__ == "__main__":
    # Handle command-line invocation for kernel startup
    if len(sys.argv) >= 4 and sys.argv[1] == "start-kernel":
        port = int(sys.argv[2])
        connection_file = sys.argv[3]
        _start_kernel_handler(port, connection_file)
    else:
        print("Usage: python -m notebookize start-kernel <port> <connection_file>", file=sys.stderr)
        sys.exit(1)


