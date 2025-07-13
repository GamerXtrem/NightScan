"""
Streaming utilities for efficient file handling.

Provides functions to handle large file uploads without loading
entire files into memory.
"""

import os
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Callable
from werkzeug.datastructures import FileStorage
import logging

logger = logging.getLogger(__name__)

# Default chunk size for streaming operations (64KB)
DEFAULT_CHUNK_SIZE = 64 * 1024


class StreamingFileHandler:
    """Handle file operations with streaming to avoid memory exhaustion."""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize streaming file handler.
        
        Args:
            chunk_size: Size of chunks to read/write at a time
        """
        self.chunk_size = chunk_size
    
    def save_file_streaming(
        self, 
        file: FileStorage, 
        destination: Path,
        max_size: Optional[int] = None,
        calculate_hash: bool = False,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Tuple[int, Optional[str]]:
        """
        Save uploaded file using streaming to avoid loading into memory.
        
        Args:
            file: Werkzeug FileStorage object
            destination: Path where to save the file
            max_size: Maximum allowed file size (raises ValueError if exceeded)
            calculate_hash: Whether to calculate SHA-256 hash while streaming
            progress_callback: Optional callback function called with bytes written
            
        Returns:
            Tuple of (file_size, file_hash)
            
        Raises:
            ValueError: If file exceeds max_size
        """
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize hash if requested
        hasher = hashlib.sha256() if calculate_hash else None
        
        total_bytes = 0
        
        try:
            # Stream file to destination
            with open(destination, 'wb') as output:
                while True:
                    chunk = file.stream.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    # Check size limit before writing
                    if max_size and total_bytes + len(chunk) > max_size:
                        # Clean up partial file
                        output.close()
                        if destination.exists():
                            os.unlink(destination)
                        raise ValueError(f"File exceeds maximum size of {max_size} bytes")
                    
                    # Write chunk
                    output.write(chunk)
                    total_bytes += len(chunk)
                    
                    # Update hash if calculating
                    if hasher:
                        hasher.update(chunk)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(total_bytes)
            
            # Calculate final hash
            file_hash = hasher.hexdigest() if hasher else None
            
            logger.debug(f"Streamed {total_bytes} bytes to {destination}")
            return total_bytes, file_hash
            
        except Exception as e:
            # Clean up on error
            if destination.exists():
                os.unlink(destination)
            raise
    
    def copy_file_streaming(
        self,
        source: Path,
        destination: Path,
        calculate_hash: bool = False
    ) -> Tuple[int, Optional[str]]:
        """
        Copy file using streaming.
        
        Args:
            source: Source file path
            destination: Destination file path
            calculate_hash: Whether to calculate hash while copying
            
        Returns:
            Tuple of (file_size, file_hash)
        """
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        hasher = hashlib.sha256() if calculate_hash else None
        total_bytes = 0
        
        with open(source, 'rb') as src, open(destination, 'wb') as dst:
            while True:
                chunk = src.read(self.chunk_size)
                if not chunk:
                    break
                
                dst.write(chunk)
                total_bytes += len(chunk)
                
                if hasher:
                    hasher.update(chunk)
        
        file_hash = hasher.hexdigest() if hasher else None
        return total_bytes, file_hash
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file using streaming.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hasher = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def get_file_size_from_storage(self, file: FileStorage) -> int:
        """
        Get file size from FileStorage without loading into memory.
        
        Args:
            file: Werkzeug FileStorage object
            
        Returns:
            File size in bytes
        """
        # Save current position
        current_pos = file.stream.tell()
        
        # Seek to end to get size
        file.stream.seek(0, os.SEEK_END)
        file_size = file.stream.tell()
        
        # Restore position
        file.stream.seek(current_pos)
        
        return file_size
    
    def save_to_temp_file(
        self,
        file: FileStorage,
        suffix: Optional[str] = None,
        max_size: Optional[int] = None
    ) -> Tuple[tempfile.NamedTemporaryFile, int, str]:
        """
        Save uploaded file to a temporary file using streaming.
        
        Args:
            file: Werkzeug FileStorage object
            suffix: Optional suffix for temp file (e.g., '.wav')
            max_size: Maximum allowed file size
            
        Returns:
            Tuple of (temp_file, file_size, file_hash)
            
        Note:
            The caller is responsible for closing/deleting the temp file.
        """
        # Create temp file (don't auto-delete)
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        
        try:
            # Stream to temp file
            file_size, file_hash = self.save_file_streaming(
                file,
                Path(temp_file.name),
                max_size=max_size,
                calculate_hash=True
            )
            
            # Reset temp file position
            temp_file.seek(0)
            
            return temp_file, file_size, file_hash
            
        except Exception:
            # Clean up on error
            temp_file.close()
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise
    
    def read_file_chunks(self, file_path: Path):
        """
        Generator that yields file content in chunks.
        
        Args:
            file_path: Path to file to read
            
        Yields:
            Chunks of file content
        """
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk


class StreamingCache:
    """Cache implementation that supports streaming operations."""
    
    def __init__(self, cache_dir: Path, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize streaming cache.
        
        Args:
            cache_dir: Directory for cache storage
            chunk_size: Chunk size for streaming operations
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.handler = StreamingFileHandler(chunk_size)
    
    def get_cache_path(self, file_hash: str) -> Path:
        """Get cache file path for given hash."""
        # Use first 2 chars of hash for directory sharding
        shard = file_hash[:2]
        return self.cache_dir / shard / f"{file_hash}.cache"
    
    def save_to_cache(self, source_path: Path, file_hash: str) -> None:
        """
        Save file to cache using streaming.
        
        Args:
            source_path: Path to source file
            file_hash: Hash of file content
        """
        cache_path = self.get_cache_path(file_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy using streaming
        self.handler.copy_file_streaming(source_path, cache_path)
        logger.debug(f"Cached file with hash {file_hash}")
    
    def get_from_cache(self, file_hash: str) -> Optional[Path]:
        """
        Get file from cache if it exists.
        
        Args:
            file_hash: Hash of file content
            
        Returns:
            Path to cached file or None if not found
        """
        cache_path = self.get_cache_path(file_hash)
        if cache_path.exists():
            logger.debug(f"Cache hit for hash {file_hash}")
            return cache_path
        return None
    
    def copy_from_cache(self, file_hash: str, destination: Path) -> bool:
        """
        Copy file from cache to destination using streaming.
        
        Args:
            file_hash: Hash of cached file
            destination: Where to copy the file
            
        Returns:
            True if copied successfully, False if not in cache
        """
        cache_path = self.get_from_cache(file_hash)
        if cache_path:
            self.handler.copy_file_streaming(cache_path, destination)
            return True
        return False


def create_streaming_response(file_path: Path, mimetype: str = 'application/octet-stream'):
    """
    Create a Flask streaming response for a file.
    
    Args:
        file_path: Path to file to stream
        mimetype: MIME type of the file
        
    Returns:
        Flask Response object that streams the file
    """
    from flask import Response
    
    def generate():
        handler = StreamingFileHandler()
        for chunk in handler.read_file_chunks(file_path):
            yield chunk
    
    return Response(
        generate(),
        mimetype=mimetype,
        headers={
            'Content-Disposition': f'attachment; filename={file_path.name}',
            'Content-Length': str(file_path.stat().st_size)
        }
    )


# Convenience functions for common operations
def save_upload_streaming(
    file: FileStorage,
    destination: Path,
    max_size: Optional[int] = None
) -> Tuple[int, str]:
    """
    Convenience function to save an upload with hash calculation.
    
    Args:
        file: Uploaded file
        destination: Where to save it
        max_size: Maximum allowed size
        
    Returns:
        Tuple of (file_size, file_hash)
    """
    handler = StreamingFileHandler()
    return handler.save_file_streaming(
        file,
        destination,
        max_size=max_size,
        calculate_hash=True
    )


def process_large_file(
    file_path: Path,
    process_func: Callable[[bytes], None],
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> None:
    """
    Process a large file in chunks without loading it entirely.
    
    Args:
        file_path: Path to file
        process_func: Function to call with each chunk
        chunk_size: Size of chunks to process
    """
    handler = StreamingFileHandler(chunk_size)
    for chunk in handler.read_file_chunks(file_path):
        process_func(chunk)