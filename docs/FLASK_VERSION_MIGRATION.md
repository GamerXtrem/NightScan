# Flask Version Migration Guide

## Flask 3.2 Deprecation Warning

### Issue
Flask 3.1.1 deprecates the `__version__` attribute:
```
DeprecationWarning: The '__version__' attribute is deprecated and will be removed in Flask 3.2. Use feature detection or 'importlib.metadata.version("flask")' instead.
```

### Solution

#### Before (Deprecated):
```python
import flask
version = flask.__version__
```

#### After (Recommended):
```python
import importlib.metadata
version = importlib.metadata.version("flask")
```

### Example Implementation

```python
def get_flask_version():
    """Get Flask version in a future-compatible way."""
    try:
        import importlib.metadata
        return importlib.metadata.version("flask")
    except ImportError:
        # Fallback not needed for Python 3.13
        # import pkg_resources
        # return pkg_resources.get_distribution("flask").version
        raise ImportError("importlib.metadata not available")
```

### Python 3.13 Compatibility

The `importlib.metadata` module is part of the standard library in Python 3.8+, making it fully compatible with Python 3.13.

#### Usage in NightScan:

```python
# In version display or logging
import importlib.metadata

def get_app_versions():
    """Get all relevant package versions."""
    packages = ['flask', 'torch', 'numpy', 'sqlalchemy']
    versions = {}
    
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    
    return versions
```

### Migration Timeline

- **Flask 3.1**: `__version__` deprecated with warning
- **Flask 3.2**: `__version__` will be removed (estimated)
- **Action required**: Update all version checks before Flask 3.2

### Testing

```python
# Test both methods work
import flask
import importlib.metadata

# This will show deprecation warning
old_version = flask.__version__

# This is the new way
new_version = importlib.metadata.version("flask")

assert old_version == new_version  # Should be True
```

### Impact Assessment

✅ **No breaking changes** - This is only a deprecation warning
✅ **Python 3.13 compatible** - importlib.metadata is standard library
✅ **Python 3.13 compatible** - Works with Python 3.13+
⚠️ **Action needed** - Update version checks before Flask 3.2