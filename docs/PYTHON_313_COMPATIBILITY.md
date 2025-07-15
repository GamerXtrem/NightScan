# Python 3.13 Compatibility Guide for NightScan

## Overview

NightScan has been updated to support Python 3.13 with some important considerations and limitations. This guide explains the current compatibility status and provides instructions for working with Python 3.13.

## Current Status

### ✅ **Fully Compatible**
- **PyTorch**: Fully compatible with Python 3.13 as of January 2025
- **Flask**: Compatible with Python 3.13 (Flask 3.1.1+)
- **All core NightScan components**: Audio processing, image processing, web interface
- **Database operations**: PostgreSQL, Redis, SQLAlchemy
- **Authentication and security**: All security modules work with Python 3.13

### ⚠️ **Partially Compatible**
- **TensorFlow**: Not yet compatible with Python 3.13
  - Expected support: Early 2025
  - Current workaround: Use PyTorch for model training and inference
  - Alternative: Use ONNX for model interoperability

### ✅ **Enhanced Features**
- **Performance**: Python 3.13 provides better performance for CPU-bound tasks
- **Type hints**: Modern PEP 585 syntax supported
- **Error handling**: Improved error messages and debugging
- **Memory usage**: Better memory management for large model inference

## Installation

### Using Python 3.13

```bash
# Create virtual environment with Python 3.13
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python 3.13 compatible requirements
pip install -r requirements-python313.txt
```

### Using Docker with Python 3.13

```bash
# Build with Python 3.13
docker build --build-arg PYTHON_VERSION=3.13 -t nightscan:python313 .

# Or use docker-compose
PYTHON_VERSION=3.13 docker-compose up --build
```

## TensorFlow Limitations

### What's Not Available
- Model optimization with TensorFlow Lite
- TensorFlow-based data processing
- TensorFlow Serving integration

### Workarounds
1. **Use PyTorch**: All model training and inference can be done with PyTorch
2. **ONNX Export**: Export models to ONNX format for interoperability
3. **Model Quantization**: Use PyTorch's quantization instead of TensorFlow Lite

### Example: PyTorch Model Export to ONNX

```python
import torch
import torch.onnx

# Load your PyTorch model
model = torch.load('models/best_model.pth')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'models/model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
```

## Development Environment

### VS Code/PyCharm Setup
```json
{
  "python.defaultInterpreterPath": "/path/to/python3.13/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--target-version", "py313"]
}
```

### Type Checking with mypy
```bash
# Run mypy with Python 3.13 compatibility
mypy . --python-version 3.13 --ignore-missing-imports
```

## Testing

### Running Tests with Python 3.13
```bash
# Run all tests
pytest tests/ -v

# Run tests excluding TensorFlow-dependent tests
pytest tests/ -v -m "not tensorflow"

# Run performance tests (Python 3.13 optimized)
pytest tests/ -v -m "performance"
```

### CI/CD Integration
The CI/CD pipeline now includes Python 3.13 testing:
- Unit tests with Python 3.13
- Integration tests (excluding TensorFlow components)
- Performance benchmarks
- Docker build tests

## Performance Considerations

### Python 3.13 Improvements
- **Free-threaded mode**: Better multiprocessing performance
- **JIT compilation**: Improved performance for CPU-bound tasks
- **Memory management**: Reduced memory usage for large applications

### Recommended Settings
```python
# For optimal performance with Python 3.13
import os
os.environ['PYTHONOPTIMIZE'] = '1'  # Enable optimizations
os.environ['PYTHONUNBUFFERED'] = '1'  # Disable buffering
```

## Migration Guide

### From Python 3.12 to 3.13
1. Update your virtual environment
2. Install `requirements-python313.txt`
3. Run tests to ensure compatibility
4. Update Docker images if needed

### Handling TensorFlow Dependencies
```python
# Conditional import pattern
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not available - using PyTorch alternative")

# Use PyTorch alternative
if not HAS_TENSORFLOW:
    import torch
    # Implement PyTorch-based functionality
```

## Known Issues

### Current Limitations
1. **TensorFlow model optimization**: Not available until TensorFlow supports Python 3.13
2. **Some third-party packages**: May not have Python 3.13 wheels yet
3. **Legacy code**: Some older ML libraries may need updates

### Solutions
1. Use `requirements-python313.txt` for tested dependencies
2. Consider using ONNX for model interoperability
3. Report issues to the NightScan team for priority fixes

## Future Roadmap

### Q1 2025
- [x] Python 3.13 compatibility for core components
- [x] PyTorch integration improvements
- [x] Docker support for Python 3.13

### Q2 2025 (Expected)
- [ ] TensorFlow Python 3.13 support
- [ ] Complete TensorFlow Lite integration
- [ ] Performance optimization benchmarks
- [ ] Enhanced model quantization support

## Getting Help

### Resources
- [Python 3.13 Release Notes](https://docs.python.org/3.13/whatsnew/3.13.html)
- [NightScan Documentation](./README.md)
- [PyTorch Python 3.13 Guide](https://pytorch.org/docs/stable/notes/python_api.html)

### Support
- Open an issue on the [NightScan GitHub repository](https://github.com/GamerXtrem/NightScan/issues)
- Check the [CI/CD status](https://github.com/GamerXtrem/NightScan/actions) for Python 3.13 builds
- Join the community discussions for Python 3.13 migration tips

## Examples

### Basic Model Training with Python 3.13
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Modern type hints (Python 3.13)
from collections.abc import Iterator

class NightScanModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

# Training loop optimized for Python 3.13
def train_model(model: NightScanModel, dataloader: DataLoader) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        # Training code here
        pass
```

This guide will be updated as TensorFlow adds Python 3.13 support and new compatibility issues are discovered.