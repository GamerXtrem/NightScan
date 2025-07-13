"""
Secure File Upload Handler Module

Handles secure file uploads with validation and scanning.
"""

import os
import logging
import hashlib
import mimetypes
import magic
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import uuid
from PIL import Image
import zipfile
import tarfile
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename as werkzeug_secure_filename

logger = logging.getLogger(__name__)


class SecureFileHandler:
    """Handles secure file uploads and processing."""
    
    def __init__(self, config):
        self.config = config
        
        # File type configurations
        self.allowed_extensions = set(getattr(config.upload, 'allowed_extensions', [
            'jpg', 'jpeg', 'png', 'gif', 'webp',  # Images
            'mp3', 'wav', 'flac', 'ogg',  # Audio
            'mp4', 'avi', 'mov', 'webm',  # Video
            'pdf', 'txt', 'csv', 'json',  # Documents
            'zip', 'tar', 'gz'  # Archives
        ]))
        
        self.mime_types = {
            # Images
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            # Audio
            'mp3': 'audio/mpeg',
            'wav': 'audio/wav',
            'flac': 'audio/flac',
            'ogg': 'audio/ogg',
            # Video
            'mp4': 'video/mp4',
            'avi': 'video/x-msvideo',
            'mov': 'video/quicktime',
            'webm': 'video/webm',
            # Documents
            'pdf': 'application/pdf',
            'txt': 'text/plain',
            'csv': 'text/csv',
            'json': 'application/json',
            # Archives
            'zip': 'application/zip',
            'tar': 'application/x-tar',
            'gz': 'application/gzip'
        }
        
        # Security settings
        self.max_file_size = config.upload.max_file_size
        self.scan_for_malware = getattr(config.upload, 'scan_for_malware', True)
        self.quarantine_suspicious = getattr(config.upload, 'quarantine_suspicious', True)
        
        # Upload directories
        self.upload_dir = Path(config.paths.uploads)
        self.temp_dir = Path(config.paths.temp) / 'uploads'
        self.quarantine_dir = Path(config.paths.temp) / 'quarantine'
        
        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    def init_app(self, app) -> None:
        """Initialize with Flask app."""
        self.app = app
        logger.info("Secure file handler initialized")
    
    def validate_file(self, file: FileStorage) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file or file.filename == '':
            return False, "No file provided"
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > self.max_file_size:
            return False, f"File size exceeds maximum allowed ({self.max_file_size} bytes)"
        
        if file_size == 0:
            return False, "Empty file"
        
        # Check extension
        ext = self._get_file_extension(file.filename)
        if ext not in self.allowed_extensions:
            return False, f"File type '{ext}' not allowed"
        
        # Check MIME type
        mime_type = self._get_mime_type(file)
        expected_mime = self.mime_types.get(ext)
        
        if expected_mime and mime_type != expected_mime:
            # Check for common MIME type variations
            if not self._is_mime_type_acceptable(mime_type, expected_mime):
                return False, f"MIME type mismatch: expected {expected_mime}, got {mime_type}"
        
        return True, None
    
    def secure_save(self, file: FileStorage, subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """
        Securely save uploaded file.
        
        Returns:
            Dictionary with file information
        """
        # Validate file first
        is_valid, error = self.validate_file(file)
        if not is_valid:
            raise ValueError(error)
        
        # Generate secure filename
        original_filename = file.filename
        secure_name = self._generate_secure_filename(original_filename)
        
        # Create subdirectory if specified
        if subdirectory:
            save_dir = self.upload_dir / subdirectory
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.upload_dir
        
        # Save to temporary location first
        temp_path = self.temp_dir / secure_name
        file.save(str(temp_path))
        
        try:
            # Perform security checks
            if self.scan_for_malware:
                is_safe, threat_info = self._scan_file(temp_path)
                if not is_safe:
                    # Move to quarantine
                    if self.quarantine_suspicious:
                        quarantine_path = self.quarantine_dir / f"{datetime.utcnow().isoformat()}_{secure_name}"
                        shutil.move(str(temp_path), str(quarantine_path))
                        
                        logger.warning(f"File quarantined: {original_filename} - {threat_info}")
                    else:
                        os.unlink(temp_path)
                    
                    raise ValueError(f"File failed security scan: {threat_info}")
            
            # Process file based on type
            file_info = self._process_file(temp_path)
            
            # Move to final location
            final_path = save_dir / secure_name
            shutil.move(str(temp_path), str(final_path))
            
            # Set secure permissions
            os.chmod(final_path, 0o644)
            
            # Generate file metadata
            metadata = {
                'original_filename': original_filename,
                'secure_filename': secure_name,
                'path': str(final_path.relative_to(self.upload_dir)),
                'size': final_path.stat().st_size,
                'mime_type': self._get_mime_type_from_file(final_path),
                'extension': self._get_file_extension(secure_name),
                'hash': self._calculate_file_hash(final_path),
                'uploaded_at': datetime.utcnow().isoformat(),
                **file_info
            }
            
            logger.info(f"File saved: {metadata['secure_filename']}")
            return metadata
            
        except Exception as e:
            # Clean up temporary file
            if temp_path.exists():
                os.unlink(temp_path)
            raise
    
    def _generate_secure_filename(self, filename: str) -> str:
        """Generate secure filename."""
        # Use werkzeug's secure_filename as base
        secure_name = werkzeug_secure_filename(filename)
        
        # If filename is empty after sanitization, generate one
        if not secure_name:
            secure_name = f"upload_{uuid.uuid4().hex}"
        
        # Split name and extension
        name = Path(secure_name).stem
        ext = Path(secure_name).suffix
        
        # Add timestamp and random component to prevent collisions
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        random_str = uuid.uuid4().hex[:8]
        
        # Construct final filename
        final_name = f"{name}_{timestamp}_{random_str}{ext}"
        
        return final_name
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension."""
        return Path(filename).suffix.lower().lstrip('.')
    
    def _get_mime_type(self, file: FileStorage) -> str:
        """Get MIME type from file."""
        # Reset file position
        file.seek(0)
        
        # Read first few bytes for magic number detection
        header = file.read(1024)
        file.seek(0)
        
        # Use python-magic for detection
        try:
            mime = magic.from_buffer(header, mime=True)
            return mime
        except Exception:
            # Fall back to mimetypes
            return mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
    
    def _get_mime_type_from_file(self, file_path: Path) -> str:
        """Get MIME type from file path."""
        try:
            mime = magic.from_file(str(file_path), mime=True)
            return mime
        except Exception:
            return mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'
    
    def _is_mime_type_acceptable(self, detected: str, expected: str) -> bool:
        """Check if detected MIME type is acceptable."""
        # Handle common variations
        mime_variations = {
            'image/jpeg': ['image/jpg', 'image/pjpeg'],
            'text/plain': ['text/x-log', 'text/x-plain'],
            'application/zip': ['application/x-zip-compressed']
        }
        
        if detected == expected:
            return True
        
        variations = mime_variations.get(expected, [])
        return detected in variations
    
    def _scan_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Scan file for malware and suspicious content.
        
        Returns:
            Tuple of (is_safe, threat_description)
        """
        threats = []
        
        # Check for suspicious file patterns
        content_sample = file_path.read_bytes()[:4096]
        
        # Check for executable signatures
        executable_signatures = [
            b'MZ',  # PE executables
            b'\x7fELF',  # ELF executables
            b'\xca\xfe\xba\xbe',  # Mach-O
            b'\xfe\xed\xfa',  # Mach-O
            b'#!/',  # Shell scripts
        ]
        
        for sig in executable_signatures:
            if content_sample.startswith(sig):
                ext = file_path.suffix.lower()
                if ext not in ['.exe', '.dll', '.so', '.dylib', '.sh', '.bat']:
                    threats.append(f"Executable content in {ext} file")
        
        # Check for embedded scripts in images
        if self._get_file_extension(file_path.name) in ['jpg', 'jpeg', 'png', 'gif']:
            if b'<script' in content_sample or b'<?php' in content_sample:
                threats.append("Embedded script in image file")
        
        # Check ZIP files for dangerous content
        if file_path.suffix.lower() == '.zip':
            zip_threats = self._scan_zip_file(file_path)
            threats.extend(zip_threats)
        
        # Check for double extensions
        if file_path.name.count('.') > 1:
            parts = file_path.name.split('.')
            if parts[-2].lower() in ['exe', 'scr', 'bat', 'cmd', 'com']:
                threats.append("Suspicious double extension")
        
        # Return results
        if threats:
            return False, "; ".join(threats)
        
        return True, None
    
    def _scan_zip_file(self, zip_path: Path) -> List[str]:
        """Scan ZIP file for threats."""
        threats = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for info in zf.filelist:
                    # Check for path traversal
                    if '..' in info.filename or info.filename.startswith('/'):
                        threats.append(f"Path traversal attempt: {info.filename}")
                    
                    # Check for suspicious files
                    ext = Path(info.filename).suffix.lower()
                    if ext in ['.exe', '.dll', '.scr', '.bat', '.cmd', '.com', '.vbs', '.js']:
                        threats.append(f"Executable in archive: {info.filename}")
                    
                    # Check for zip bombs
                    if info.file_size > 0 and info.compress_size > 0:
                        ratio = info.file_size / info.compress_size
                        if ratio > 100:
                            threats.append(f"Suspicious compression ratio: {info.filename}")
        
        except Exception as e:
            threats.append(f"Failed to scan ZIP: {str(e)}")
        
        return threats
    
    def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process file based on type."""
        ext = self._get_file_extension(file_path.name)
        info = {}
        
        # Process images
        if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            try:
                with Image.open(file_path) as img:
                    info['width'] = img.width
                    info['height'] = img.height
                    info['format'] = img.format
                    info['mode'] = img.mode
                    
                    # Check for suspicious image dimensions
                    if img.width * img.height > 100_000_000:  # 100 megapixels
                        logger.warning(f"Very large image: {img.width}x{img.height}")
                    
                    # Strip EXIF data for privacy
                    if ext in ['jpg', 'jpeg']:
                        self._strip_exif(file_path)
            
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
        
        return info
    
    def _strip_exif(self, image_path: Path) -> None:
        """Strip EXIF data from image."""
        try:
            img = Image.open(image_path)
            data = list(img.getdata())
            image_without_exif = Image.new(img.mode, img.size)
            image_without_exif.putdata(data)
            image_without_exif.save(image_path)
        except Exception as e:
            logger.error(f"Failed to strip EXIF: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def delete_file(self, file_path: str) -> bool:
        """Securely delete uploaded file."""
        try:
            full_path = self.upload_dir / file_path
            if full_path.exists() and full_path.is_file():
                # Overwrite with random data before deletion (optional)
                if hasattr(self.config.upload, 'secure_delete') and self.config.upload.secure_delete:
                    with open(full_path, 'ba+', buffering=0) as f:
                        length = f.tell()
                        f.seek(0)
                        f.write(os.urandom(length))
                
                # Delete file
                os.unlink(full_path)
                logger.info(f"File deleted: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """Get upload statistics."""
        total_files = sum(1 for _ in self.upload_dir.rglob('*') if _.is_file())
        total_size = sum(f.stat().st_size for f in self.upload_dir.rglob('*') if f.is_file())
        quarantined = sum(1 for _ in self.quarantine_dir.rglob('*') if _.is_file())
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'quarantined_files': quarantined,
            'allowed_extensions': list(self.allowed_extensions),
            'max_file_size': self.max_file_size
        }