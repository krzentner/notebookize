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


def _generate_jupytext_notebook(func_name: str, body_source: str) -> Path:
    """
    Generate a jupytext .py percent format notebook from function source code.
    Returns the path to the generated notebook.
    """
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

    # Add header (minimal, just the jupytext metadata)
    content_parts.append("""# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---""")

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


def _open_notebook_in_jupyterlab(notebook_path: Path, logger: logging.Logger, kernel_name: Optional[str] = None) -> None:
    """Open the generated notebook in JupyterLab using paired .ipynb file.
    
    Args:
        notebook_path: Path to the .py notebook file
        logger: Logger instance
        kernel_name: Optional kernel name to use for the notebook
    """
    # Ensure Python files open as notebooks by default in JupyterLab
    try:
        result = subprocess.run(
            ["jupytext-config", "list-default-viewer"],
            capture_output=True,
            text=True,
            check=False,
        )

        if "python" not in result.stdout:
            subprocess.run(
                ["jupytext-config", "set-default-viewer", "python"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(
                "Configured JupyterLab to open .py files as notebooks by default"
            )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # jupytext-config might not be available in older versions

    # Create a paired .ipynb file using jupytext
    ipynb_path = notebook_path.with_suffix(".ipynb")

    try:
        # Set up pairing between .py and .ipynb files
        subprocess.run(
            ["jupytext", "--set-formats", "py:percent,ipynb", str(notebook_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Created paired notebook: {ipynb_path}")

        # Sync to create the .ipynb file
        subprocess.run(
            ["jupytext", "--sync", str(notebook_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        
        # If we have a custom kernel, update the .ipynb file to use it
        if kernel_name:
            try:
                with open(ipynb_path, 'r') as f:
                    notebook_data = json.load(f)
                
                # Set the kernel metadata
                if 'metadata' not in notebook_data:
                    notebook_data['metadata'] = {}
                if 'kernelspec' not in notebook_data['metadata']:
                    notebook_data['metadata']['kernelspec'] = {}
                
                notebook_data['metadata']['kernelspec']['name'] = kernel_name
                notebook_data['metadata']['kernelspec']['display_name'] = f"Notebookize - {kernel_name}"
                notebook_data['metadata']['kernelspec']['language'] = 'python'
                
                with open(ipynb_path, 'w') as f:
                    json.dump(notebook_data, f, indent=1)
                
                logger.info(f"Set notebook to use kernel: {kernel_name}")
            except Exception as e:
                logger.warning(f"Could not set kernel metadata: {e}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create paired notebook: {e}")
        return
    except FileNotFoundError:
        logger.error("jupytext not found. Please install with: pip install jupytext")
        return

    # Open JupyterLab with the .ipynb file
    try:
        subprocess.Popen(
            ["jupyter", "lab", str(ipynb_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Opened JupyterLab with notebook: {ipynb_path}")
        logger.info(
            f"Note: Changes will sync between {notebook_path.name} and {ipynb_path.name}"
        )
    except FileNotFoundError:
        logger.error(
            "JupyterLab not found. Please install with: pip install jupyterlab"
        )


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

    # Rewrite the function in the source file
    success = _rewrite_function_in_file(source_file, func_name, new_body)

    if success:
        logger.info(f"Successfully updated {func_name}")
        logger.info("New function body:")
        logger.info(new_body)
    else:
        logger.error(f"Failed to update {func_name}")

    return success


def _watch_notebook_for_changes(
    notebook_path: Path, source_file: str, func_name: str, logger: logging.Logger,
    write_back: bool = True
) -> None:
    """Watch notebook files for changes and optionally update the source file when detected."""
    # Check if we have a paired .ipynb file
    ipynb_path = notebook_path.with_suffix(".ipynb")
    has_ipynb = ipynb_path.exists()

    if has_ipynb:
        logger.info("Watching paired notebooks for changes:")
        logger.info(f"  - {notebook_path}")
        logger.info(f"  - {ipynb_path}")
    else:
        logger.info(f"Watching {notebook_path} for changes...")
    
    if not write_back:
        logger.info("Note: write_back is disabled - changes will NOT be written to source file")

    logger.info("Press Ctrl+C to stop watching")

    last_py_mtime = notebook_path.stat().st_mtime
    last_ipynb_mtime = ipynb_path.stat().st_mtime if has_ipynb else 0
    check_interval = float(os.environ.get("NOTEBOOKIZE_CHECK_INTERVAL", "1.0"))

    try:
        while True:
            time.sleep(check_interval)

            try:
                # Check .py file
                current_py_mtime = notebook_path.stat().st_mtime
                changed = False

                if current_py_mtime > last_py_mtime:
                    changed = True
                    last_py_mtime = current_py_mtime

                # Check .ipynb file if it exists
                if has_ipynb and ipynb_path.exists():
                    current_ipynb_mtime = ipynb_path.stat().st_mtime
                    if current_ipynb_mtime > last_ipynb_mtime:
                        # Sync from .ipynb to .py
                        subprocess.run(
                            ["jupytext", "--sync", str(ipynb_path)],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        changed = True
                        last_ipynb_mtime = current_ipynb_mtime
                        last_py_mtime = notebook_path.stat().st_mtime

                if changed:
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
                               user_module: Optional[Any] = None) -> Tuple[Optional[str], Optional[threading.Thread]]:
    """Set up a kernel for the notebook if requested.
    
    Args:
        func_name: Name of the decorated function
        kernel_enabled: Whether to enable kernel
        logger: Logger instance
        user_ns: User namespace (locals) to provide to the kernel
        user_module: Module to provide to the kernel
    
    Returns:
        Tuple of (kernel_name, kernel_thread) or (None, None) if kernel not set up.
    """
    if not kernel_enabled:
        return None, None
    
    if IPKernelApp is None or extract_module_locals is None:
        logger.warning("ipykernel not installed, kernel mode disabled")
        return None, None
    
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
            sys.executable, "-m", "ipykernel_launcher",
            "-f", "{connection_file}"
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
    
    # Start kernel in a thread
    kernel_thread = None
    kernel_app = None
    
    def start_kernel() -> None:
        nonlocal kernel_app
        try:
            # Create IPKernelApp instance
            kernel_app = IPKernelApp.instance()
            kernel_app.connection_file = connection_file_path
            kernel_app.initialize([])
            
            # Set the user namespace if provided
            if user_ns is not None:
                kernel_app.kernel.user_ns = user_ns
            if user_module is not None:
                kernel_app.kernel.user_module = user_module
            if user_ns is not None or user_module is not None:
                kernel_app.shell.set_completer_frame()  # type: ignore[union-attr]
            
            logger.info(f"Starting kernel with connection file: {connection_file_path}")
            kernel_app.start()
        except Exception as e:
            logger.error(f"Failed to start kernel: {e}")
    
    kernel_thread = threading.Thread(target=start_kernel, daemon=True)
    kernel_thread.start()
    
    # Give the kernel a moment to start
    time.sleep(1)
    
    return kernel_name, kernel_thread


F = TypeVar("F", bound=Callable[..., Any])


def notebookize(
    func: Optional[F] = None, *, 
    open_jupyterlab: bool = True,
    write_back: bool = True,
    kernel: bool = False
) -> Union[F, Callable[[F], F]]:
    """
    Decorator that generates a jupytext notebook and watches for changes,
    optionally updating the original function source when the notebook is modified.
    
    Args:
        func: The function to decorate
        open_jupyterlab: Whether to open the notebook in JupyterLab
        write_back: Whether to write changes back to the source file (default: True)
        kernel: Whether to set up a custom kernel for the notebook (default: False)
    """
    if func is None:
        return functools.partial(  # type: ignore[return-value]
            notebookize, open_jupyterlab=open_jupyterlab, write_back=write_back, kernel=kernel
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

        # Generate jupytext notebook
        try:
            notebook_path = _generate_jupytext_notebook(func.__name__, body_source)
            logger.info(f"Notebook saved to: {notebook_path}")
        except Exception as e:
            logger.error(f"Failed to generate notebook: {e}")
            return func(*args, **kwargs)

        # Set up kernel if requested
        # Capture the calling frame's namespace if kernel is enabled
        user_ns = None
        user_module = None
        if kernel and extract_module_locals is not None:
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
        
        kernel_name, kernel_thread = _setup_kernel_if_requested(
            func.__name__, kernel, logger, user_ns, user_module
        )
        
        # Open in JupyterLab if requested
        if open_jupyterlab:
            _open_notebook_in_jupyterlab(notebook_path, logger, kernel_name)

        # Watch for changes
        try:
            _watch_notebook_for_changes(notebook_path, source_file, func.__name__, logger, write_back)
        finally:
            # Clean up kernel if it was registered
            if kernel_name:
                _unregister_kernel(kernel_name)
                logger.info(f"Cleaned up kernel: {kernel_name}")

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


