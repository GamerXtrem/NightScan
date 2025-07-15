# Multiprocessing Issues with Python 3.13

## Problem Identified

When using `multiprocessing.Pool.map()` with inline functions in Python 3.13, the following error occurs:

```
AttributeError: Can't get attribute 'test_worker' on <module '__main__' (<class '_frozen_importlib.BuiltinImporter'>)>
```

## Root Cause

Python 3.13 changed how multiprocessing handles function serialization for spawned processes. Functions defined in `__main__` (like inline scripts) cannot be pickled properly.

## Solutions

### ✅ Solution 1: Use Module-Level Functions

**Problem:**
```python
# This fails in Python 3.13
def test_worker(x):
    return x ** 2

with mp.Pool(processes=2) as pool:
    result = pool.map(test_worker, [1, 2, 3, 4])
```

**Solution:**
```python
# Create a separate module: workers.py
def test_worker(x):
    return x ** 2

# In main script:
from workers import test_worker
import multiprocessing as mp

if __name__ == '__main__':
    with mp.Pool(processes=2) as pool:
        result = pool.map(test_worker, [1, 2, 3, 4])
```

### ✅ Solution 2: Use Lambdas with Process Pool

**Problem:**
```python
# This fails
with mp.Pool() as pool:
    result = pool.map(lambda x: x**2, [1, 2, 3, 4])
```

**Solution:**
```python
# Use list comprehension or regular loops for simple operations
result = [x**2 for x in [1, 2, 3, 4]]

# Or use concurrent.futures for more complex cases
from concurrent.futures import ProcessPoolExecutor

def worker_function(x):
    return x ** 2

with ProcessPoolExecutor() as executor:
    result = list(executor.map(worker_function, [1, 2, 3, 4]))
```

### ✅ Solution 3: Use Threading for I/O-bound Tasks

```python
from concurrent.futures import ThreadPoolExecutor

def io_worker(item):
    # For I/O-bound tasks, threading is often better
    return process_item(item)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(io_worker, items))
```

## NightScan Implementation

### Audio Processing Module

```python
# audio_processing/workers.py
def process_audio_chunk(chunk_data):
    """Process a single audio chunk."""
    # Audio processing logic here
    return processed_chunk

# In main audio processing:
from audio_processing.workers import process_audio_chunk
from concurrent.futures import ProcessPoolExecutor

def process_audio_file(file_path):
    chunks = split_audio_file(file_path)
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_audio_chunk, chunks))
    
    return combine_results(results)
```

### ML Model Inference

```python
# ml_processing/inference_workers.py
def run_model_inference(input_data):
    """Run model inference on input data."""
    # Model inference logic
    return predictions

# In prediction service:
from ml_processing.inference_workers import run_model_inference
from concurrent.futures import ProcessPoolExecutor

def batch_predict(input_batch):
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_model_inference, input_batch))
    
    return results
```

## Testing Workaround

For testing purposes, you can use the `fork` start method on Unix systems:

```python
import multiprocessing as mp

def test_multiprocessing():
    # Set start method to fork (Unix only)
    mp.set_start_method('fork', force=True)
    
    def test_worker(x):
        return x ** 2
    
    with mp.Pool(processes=2) as pool:
        result = pool.map(test_worker, [1, 2, 3, 4])
    
    return result
```

## Migration Checklist

- [ ] Move worker functions to separate modules
- [ ] Update imports in main processing files
- [ ] Test multiprocessing functionality
- [ ] Update documentation
- [ ] Consider using concurrent.futures for new code

## Performance Considerations

- **CPU-bound**: Use `ProcessPoolExecutor` or `multiprocessing.Pool`
- **I/O-bound**: Use `ThreadPoolExecutor` 
- **Mixed workloads**: Consider async/await patterns

## Python 3.13 Compatibility

✅ **concurrent.futures**: Fully compatible
✅ **multiprocessing with modules**: Works correctly
✅ **Threading**: No changes
⚠️ **Inline functions**: Requires refactoring

This change ensures better code organization and Python 3.13 compatibility.