# notebookize

A Python decorator that turns functions back into jupyter notebooks, complete with their context.

### Installation:

```
pip install notebookize
```

### Usage:

```python
from notebookize import notebookize

@notebookize
def my_function(my_function_args, ...):
    ...
```