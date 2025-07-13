# NightScan File Upload Streaming Implementation

## Overview

This document describes the streaming file upload implementation that prevents large files from being loaded entirely into memory, improving scalability and preventing out-of-memory errors on resource-constrained environments.

## Problem Statement

Previously, several endpoints loaded entire files into memory:
- `web/app.py`: `data = file.read()` - Loaded 100MB files entirely
- `unified_prediction_api.py`: `file.save()` - Potentially loaded entire file
- `optimized_api_integration.py`: `audio_bytes = file.read()` - Loaded entire audio file
- `api_v1.py`: Reloaded entire file after streaming for cache computation

This could cause:
- **Memory exhaustion**: 10 concurrent 100MB uploads = 1GB RAM
- **Poor scalability**: VPS Lite (4GB RAM) could only handle ~20 uploads
- **Performance issues**: Large memory allocations slow down the system

## Solution: Streaming Implementation

### 1. Core Streaming Module (`streaming_utils.py`)

Created a reusable streaming module with:

```python
class StreamingFileHandler:
    def save_file_streaming(file, destination, max_size=None, calculate_hash=False)
    def copy_file_streaming(source, destination, calculate_hash=False)
    def calculate_file_hash(file_path)
    def get_file_size_from_storage(file)
    def save_to_temp_file(file, suffix=None, max_size=None)
    def read_file_chunks(file_path)
```

Key features:
- **Chunk-based processing**: Default 64KB chunks
- **Hash calculation during streaming**: No need to reload files
- **Size validation**: Fails fast if file exceeds limit
- **Progress callbacks**: Optional progress tracking

### 2. Updated Endpoints

#### web/app.py
```python
# Before: Loaded entire file
data = file.read()
run_prediction.delay(pred_id, filename, data, api_url)

# After: Streams to temp file, passes path
handler = StreamingFileHandler()
saved_size, file_hash = handler.save_file_streaming(file, Path(temp_path))
run_prediction.delay(pred_id, filename, temp_path, api_url)
```

#### web/tasks.py
```python
# Before: Received file content as bytes
def run_prediction(pred_id, filename, data: bytes, api_url)

# After: Receives file path, reads from disk
def run_prediction(pred_id, filename, file_path: str, api_url)
# Cleans up temp file after processing
```

#### unified_prediction_api.py
```python
# Before: Direct save (may load entire file)
file.save(filepath)

# After: Streaming save with size limit
handler = StreamingFileHandler()
file_size, file_hash = handler.save_file_streaming(file, filepath, max_size=100*1024*1024)
```

#### optimized_api_integration.py
```python
# Before: Loaded entire audio file
audio_bytes = file.read()

# After: Streams to temp file, processes from disk
handler.save_file_streaming(file, Path(temp_path))
# Process audio from disk without loading entirely
```

#### api_v1.py
```python
# Before: Calculated hash after streaming by reloading file
with open(tmp.name, 'rb') as f:
    audio_data = f.read()  # Reloaded entire file!

# After: Calculates hash during initial streaming
hasher = hashlib.sha256()
while streaming:
    hasher.update(chunk)
file_hash = hasher.hexdigest()
```

### 3. Enhanced Cache Support

Updated `cache_utils.py` with hash-based methods:
```python
def get_prediction_by_hash(audio_hash: str) -> Optional[List[Dict]]
def cache_prediction_by_hash(audio_hash: str, result: List[Dict]) -> bool
```

This allows caching without needing the file content.

## Performance Impact

### Memory Usage

**Before:**
- Each 100MB upload: 100MB RAM
- 10 concurrent uploads: 1GB RAM
- Double memory for web + celery: 2GB total

**After:**
- Each upload: ~64KB RAM (chunk size)
- 10 concurrent uploads: ~640KB RAM
- 1000x reduction in memory usage

### Scalability

**Before:**
- VPS Lite (4GB RAM): ~20 concurrent uploads max
- Risk of OOM crashes

**After:**
- VPS Lite (4GB RAM): Hundreds of concurrent uploads
- Stable memory usage

### Processing Time

- **Slightly slower**: Disk I/O instead of memory
- **Better overall**: No GC pressure from large allocations
- **More predictable**: Consistent performance

## Configuration

### Chunk Size
Default: 64KB
```python
DEFAULT_CHUNK_SIZE = 64 * 1024  # 64KB
```

Adjust based on:
- Network speed
- Disk I/O performance
- Memory constraints

### File Size Limits
Enforced during streaming:
```python
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
```

Fails fast without consuming resources.

## Best Practices

1. **Always use streaming for large files** (>1MB)
2. **Calculate hashes during initial stream** to avoid reloading
3. **Clean up temp files** in finally blocks
4. **Use context managers** for file operations
5. **Implement progress callbacks** for large uploads

## Migration Guide

To use streaming in new endpoints:

```python
from streaming_utils import StreamingFileHandler

# Initialize handler
handler = StreamingFileHandler()

# Save uploaded file
file_size, file_hash = handler.save_file_streaming(
    request.files['file'],
    Path('/path/to/destination'),
    max_size=100 * 1024 * 1024
)

# Process file from disk instead of memory
```

## Monitoring

Monitor these metrics:
- **Memory usage**: Should remain constant
- **Temp file cleanup**: No accumulation
- **Upload success rate**: Should improve
- **Response times**: May increase slightly

## Security Considerations

1. **Temp file permissions**: Set to 0600
2. **Cleanup on error**: Always delete temp files
3. **Path validation**: Prevent directory traversal
4. **Hash verification**: Ensure file integrity

## Future Improvements

1. **Async streaming**: Use async I/O for better concurrency
2. **Direct streaming to S3**: Skip local disk entirely
3. **Resumable uploads**: Support chunked uploads
4. **Compression**: Stream with on-the-fly compression