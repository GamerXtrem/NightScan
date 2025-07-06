"""
Secure File Upload Handler for NightScan
Prevents malicious file uploads and path traversal attacks.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Optional, Tuple
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

class SecureFileUploader:
    """Secure file upload handler."""
    
    def __init__(self, upload_dir: str, max_file_size: int = 100 * 1024 * 1024):
        self.upload_dir = Path(upload_dir)
        self.max_file_size = max_file_size
        self.allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
        self.allowed_mimetypes = {
            'audio/wav', 'audio/x-wav', 'audio/mpeg', 
            'audio/flac', 'audio/ogg', 'audio/vorbis'
        }
        
        # Create upload directory if it doesn't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_file(self, file: FileStorage) -> Tuple[bool, str]:
        """Comprehensive file validation."""
        if not file or not file.filename:
            return False, "No file provided"
            
        # Check file size
        if hasattr(file, 'content_length') and file.content_length:
            if file.content_length > self.max_file_size:
                return False, f"File too large (max {self.max_file_size // (1024*1024)}MB)"
                
        # Validate filename
        filename = secure_filename(file.filename)
        if not filename:
            return False, "Invalid filename"
            
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            return False, f"File type not allowed. Allowed: {', '.join(self.allowed_extensions)}"
            
        # Check MIME type
        if file.mimetype not in self.allowed_mimetypes:
            # Double-check with python-magic if available
            try:
                import magic
                file.seek(0)
                file_header = file.read(1024)
                file.seek(0)
                detected_mime = magic.from_buffer(file_header, mime=True)
                if detected_mime not in self.allowed_mimetypes:
                    return False, f"File content type not allowed: {detected_mime}"
            except ImportError:
                # Fallback to basic mimetype check
                guessed_type = mimetypes.guess_type(filename)[0]
                if guessed_type not in self.allowed_mimetypes:
                    return False, f"File type not allowed: {file.mimetype}"
                    
        # Validate file header for audio files
        if not self._validate_audio_header(file):
            return False, "Invalid audio file format"
            
        return True, "File validation passed"
        
    def _validate_audio_header(self, file: FileStorage) -> bool:
        """Validate audio file headers."""
        file.seek(0)
        header = file.read(12)
        file.seek(0)
        
        # WAV file validation
        if header.startswith(b'RIFF') and header[8:12] == b'WAVE':
            return True
            
        # MP3 file validation
        if header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
            return True
            
        # FLAC file validation
        if header.startswith(b'fLaC'):
            return True
            
        # OGG file validation
        if header.startswith(b'OggS'):
            return True
            
        return False
        
    def generate_safe_filename(self, original_filename: str) -> str:
        """Generate safe, unique filename."""
        # Secure the filename
        safe_name = secure_filename(original_filename)
        if not safe_name:
            safe_name = "upload"
            
        # Add timestamp and hash for uniqueness
        import time
        timestamp = int(time.time())
        file_hash = hashlib.sha256(f"{safe_name}{timestamp}".encode()).hexdigest()[:8]
        
        name_part = Path(safe_name).stem
        ext_part = Path(safe_name).suffix
        
        return f"{name_part}_{timestamp}_{file_hash}{ext_part}"
        
    def save_file(self, file: FileStorage, custom_filename: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """Safely save uploaded file."""
        # Validate file first
        is_valid, message = self.validate_file(file)
        if not is_valid:
            return False, message, None
            
        try:
            # Generate safe filename
            if custom_filename:
                filename = self.generate_safe_filename(custom_filename)
            else:
                filename = self.generate_safe_filename(file.filename)
                
            # Ensure we're not overwriting existing files
            file_path = self.upload_dir / filename
            counter = 1
            original_filename = filename
            while file_path.exists():
                name_part = Path(original_filename).stem
                ext_part = Path(original_filename).suffix
                filename = f"{name_part}_{counter}{ext_part}"
                file_path = self.upload_dir / filename
                counter += 1
                
            # Save file
            file.save(str(file_path))
            
            # Verify file was saved correctly
            if not file_path.exists():
                return False, "Failed to save file", None
                
            # Set restrictive permissions
            os.chmod(file_path, 0o644)
            
            return True, "File uploaded successfully", str(file_path)
            
        except Exception as e:
            return False, f"Upload failed: {str(e)}", None
            
    def delete_file(self, filename: str) -> Tuple[bool, str]:
        """Safely delete uploaded file."""
        try:
            # Validate filename to prevent path traversal
            safe_name = secure_filename(filename)
            if not safe_name or safe_name != filename:
                return False, "Invalid filename"
                
            file_path = self.upload_dir / safe_name
            
            # Ensure file is within upload directory
            if not str(file_path.resolve()).startswith(str(self.upload_dir.resolve())):
                return False, "Access denied"
                
            if file_path.exists():
                file_path.unlink()
                return True, "File deleted successfully"
            else:
                return False, "File not found"
                
        except Exception as e:
            return False, f"Delete failed: {str(e)}"
            
    def list_files(self) -> List[dict]:
        """List uploaded files with metadata."""
        files = []
        try:
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        'filename': file_path.name,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'extension': file_path.suffix.lower()
                    })
        except Exception:
            pass
            
        return sorted(files, key=lambda x: x['modified'], reverse=True)

# Global uploader instance
_uploader = None

def get_secure_uploader(upload_dir: str = "uploads") -> SecureFileUploader:
    """Get global secure uploader instance."""
    global _uploader
    if _uploader is None:
        _uploader = SecureFileUploader(upload_dir)
    return _uploader
