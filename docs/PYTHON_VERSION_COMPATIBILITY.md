# Python Version Compatibility Guide

## Supported Python Versions

NightScan supports **Python 3.13** exclusively.

### Version Matrix

| Python Version | Status | Notes |
|---------------|---------|-------|
| 3.9           | ❌ Not Supported | Legacy version, no longer supported |
| 3.10          | ❌ Not Supported | Legacy version, no longer supported |
| 3.11          | ❌ Not Supported | Legacy version, no longer supported |
| 3.12          | ❌ Not Supported | Legacy version, no longer supported |
| 3.13          | ✅ Full Support | **Required version** |
| 3.14          | ❌ Not Supported | Future version, compatibility unknown |

## Key Compatibility Considerations

### Dependencies
- **PyTorch**: Versions 2.1.1-2.7.x are compatible with Python 3.13
- **NumPy**: Using <3.0.0 constraint for Python 3.13 compatibility
- **Flask**: Version 3.x supports Python 3.13
- **SQLAlchemy**: Version 2.x supports Python 3.13
- **audioop**: Native module available in Python 3.13
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
# Ensure you have Python 3.13
python3.13 --version

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
The Docker images are built with Python 3.13 for optimal compatibility.

## Version-Specific Notes

### Python 3.13
- **Required version** for NightScan
- Native audioop module available (no audioop-lts needed)
- All core features fully functional
- Latest Python optimizations and performance improvements
- Improved error messages and debugging capabilities
- Better memory management and performance

## Testing Across Versions

To test compatibility across Python versions:

```bash
# Using tox (if configured)
tox

# Test with Python 3.13
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
- **NumPy 2.x**: Fully supported with Python 3.13
- **Flask 3.2+**: Ready for migration from __version__ to importlib.metadata

## Continuous Integration

Our CI/CD pipeline tests against:
- Python 3.13 (required version)

All tests must pass on Python 3.13 before merging.