# Python Version Compatibility Guide

## Supported Python Versions

NightScan supports Python versions **3.9 through 3.13**.

### Version Matrix

| Python Version | Status | Notes |
|---------------|---------|-------|
| 3.9           | ✅ Full Support | Minimum required version |
| 3.10          | ✅ Full Support | Recommended for production |
| 3.11          | ✅ Full Support | Recommended for development |
| 3.12          | ✅ Full Support | Stable and well-tested |
| 3.13          | ✅ Full Support | Latest supported version |
| 3.14          | ❌ Not Supported | Future version, compatibility unknown |

## Key Compatibility Considerations

### Dependencies
- **PyTorch**: Versions 2.1.1-2.7.x are compatible with Python 3.9-3.13
- **NumPy**: Using <3.0.0 constraint for Python 3.13 compatibility
- **Flask**: Version 3.x supports Python 3.9+ including 3.13
- **SQLAlchemy**: Version 2.x supports Python 3.9+ including 3.13
- **audioop-lts**: Required for Python 3.13 as audioop module was removed
- **OpenCV**: opencv-python 4.12.0+ supports Python 3.13
- **Plotly**: Version 6.2.0+ supports Python 3.13
- **psycopg2-binary**: Version 2.9.10+ supports Python 3.13

### CUDA Support
CUDA dependencies are included for GPU acceleration:
- Compatible with CUDA 12.x
- Optional for CPU-only deployment
- All NVIDIA packages constrained to compatible versions

### Audio Processing
- **PyAudio**: May require additional system dependencies
- **sounddevice**: Cross-platform audio I/O
- **pydub**: Requires FFmpeg for advanced audio processing

## Installation Instructions

### Standard Installation
```bash
# Ensure you have Python 3.9-3.12
python --version

# Install dependencies
pip install -r requirements.txt
```

### Development Installation
```bash
# Install with development dependencies
pip install -r requirements.txt
pip install -e .[dev]
```

### Docker Installation
The Docker images are built with Python 3.11 for optimal compatibility.

## Version-Specific Notes

### Python 3.9
- All features supported
- Some performance optimizations available in newer versions

### Python 3.10
- Recommended for production deployment
- Better performance with pattern matching features

### Python 3.11
- Recommended for development
- Improved error messages and debugging
- Better performance overall

### Python 3.12
- Stable and well-tested
- Good performance optimizations
- Recommended for most use cases

### Python 3.13
- Latest supported version
- Requires audioop-lts package for audio processing
- All core features fully functional
- Most recent Python optimizations

## Testing Across Versions

To test compatibility across Python versions:

```bash
# Using tox (if configured)
tox

# Or test specific versions
python3.9 -m pytest tests/
python3.10 -m pytest tests/
python3.11 -m pytest tests/
python3.12 -m pytest tests/
python3.13 -m pytest tests/
```

## Troubleshooting

### Common Issues

1. **Package conflicts**: Use virtual environments
2. **CUDA compatibility**: Check GPU driver version
3. **Audio dependencies**: Install system audio libraries
4. **Build tools**: Ensure gcc/clang available for native extensions
5. **Multiprocessing**: Use module-level functions, not inline (see MULTIPROCESSING_PYTHON313.md)
6. **Flask deprecation**: Use importlib.metadata.version() instead of __version__ (see FLASK_VERSION_MIGRATION.md)

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

## Future Compatibility

- **Python 3.14**: Will be evaluated once all dependencies support it
- **PyTorch 2.8+**: Will be tested and integrated when stable
- **NumPy 2.x**: Now supported with Python 3.13 compatibility
- **Flask 3.2+**: Will require migration from __version__ to importlib.metadata

## Continuous Integration

Our CI/CD pipeline tests against:
- Python 3.9 (minimum)
- Python 3.10 (production baseline)
- Python 3.11 (development recommended)
- Python 3.12 (stable)
- Python 3.13 (latest)

All tests must pass on these versions before merging.