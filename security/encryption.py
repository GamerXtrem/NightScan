"""
Encryption and Secrets Management Module

Handles encryption, decryption, and secure secret management.
"""

import os
import base64
import secrets
import hashlib
import logging
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import hmac

logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages encryption and decryption operations."""
    
    def __init__(self, config):
        self.config = config
        self.master_key = self._get_or_create_master_key()
        self.cipher_suite = Fernet(self.master_key)
        
        # Key rotation tracking
        self.key_rotation_interval = timedelta(days=90)  # Rotate every 90 days
        self.last_rotation = self._get_last_rotation_date()
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create the master encryption key."""
        # First, check environment variable
        env_key = os.environ.get('NIGHTSCAN_MASTER_KEY')
        if env_key:
            try:
                return base64.urlsafe_b64decode(env_key.encode())
            except Exception:
                logger.error("Invalid master key in environment variable")
        
        # Check for key file
        key_file = Path(self.config.security.secret_key_file) if hasattr(self.config.security, 'secret_key_file') else Path('.secrets/master.key')
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read master key file: {e}")
        
        # Generate new key
        logger.warning("Generating new master encryption key")
        new_key = Fernet.generate_key()
        
        # Try to save it
        try:
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(new_key)
            os.chmod(key_file, 0o600)  # Read/write for owner only
            logger.info(f"Master key saved to {key_file}")
        except Exception as e:
            logger.error(f"Failed to save master key: {e}")
            logger.warning("Store this key securely: " + base64.urlsafe_b64encode(new_key).decode())
        
        return new_key
    
    def _get_last_rotation_date(self) -> datetime:
        """Get the date of last key rotation."""
        rotation_file = Path('.secrets/rotation.json')
        if rotation_file.exists():
            try:
                with open(rotation_file, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data['last_rotation'])
            except Exception:
                pass
        
        return datetime.utcnow()
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data using Fernet symmetric encryption.
        
        Args:
            data: Data to encrypt (string or bytes)
            
        Returns:
            Base64 encoded encrypted data
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted = self.cipher_suite.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted string
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt a dictionary."""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt a dictionary."""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)
    
    def derive_key(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Derive an encryption key from a password using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Optional salt (will be generated if not provided)
            
        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key, salt
    
    def generate_token(self, length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        # Generate random bytes
        random_bytes = os.urandom(32)
        
        # Create a hash to ensure consistent format
        hash_obj = hashlib.sha256(random_bytes)
        
        # Format as API key
        api_key = f"nsk_{base64.urlsafe_b64encode(hash_obj.digest()).decode('utf-8').rstrip('=')}"
        
        return api_key
    
    def hash_data(self, data: Union[str, bytes], algorithm: str = 'sha256') -> str:
        """
        Create a hash of data.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm to use
            
        Returns:
            Hex encoded hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == 'sha256':
            return hashlib.sha256(data).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(data).hexdigest()
        elif algorithm == 'md5':
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def create_hmac(self, data: Union[str, bytes], key: Optional[bytes] = None) -> str:
        """
        Create HMAC for data integrity verification.
        
        Args:
            data: Data to create HMAC for
            key: Optional HMAC key (uses master key if not provided)
            
        Returns:
            Hex encoded HMAC
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if key is None:
            key = self.master_key
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    
    def verify_hmac(self, data: Union[str, bytes], signature: str, key: Optional[bytes] = None) -> bool:
        """
        Verify HMAC signature.
        
        Args:
            data: Data to verify
            signature: HMAC signature to verify
            key: Optional HMAC key
            
        Returns:
            True if signature is valid
        """
        expected_signature = self.create_hmac(data, key)
        return hmac.compare_digest(expected_signature, signature)
    
    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Encrypt a file.
        
        Args:
            file_path: Path to file to encrypt
            output_path: Optional output path (defaults to file_path.enc)
            
        Returns:
            Path to encrypted file
        """
        if output_path is None:
            output_path = file_path.with_suffix(file_path.suffix + '.enc')
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = self.cipher_suite.encrypt(file_data)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"File encrypted: {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to encrypt file: {e}")
            raise
    
    def decrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Decrypt a file.
        
        Args:
            file_path: Path to encrypted file
            output_path: Optional output path
            
        Returns:
            Path to decrypted file
        """
        if output_path is None:
            # Remove .enc extension if present
            if file_path.suffix == '.enc':
                output_path = file_path.with_suffix('')
            else:
                output_path = file_path.with_suffix('.dec')
        
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info(f"File decrypted: {file_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to decrypt file: {e}")
            raise
    
    def should_rotate_keys(self) -> bool:
        """Check if keys should be rotated."""
        return datetime.utcnow() - self.last_rotation > self.key_rotation_interval
    
    def rotate_master_key(self) -> bool:
        """
        Rotate the master encryption key.
        
        This is a complex operation that should be done carefully.
        """
        logger.warning("Key rotation requested - manual implementation required for production")
        # Key rotation requires careful planning and data re-encryption
        # Implementation depends on specific encryption patterns in use
        # For production: implement with database migration strategy
        raise NotImplementedError("Key rotation requires production-specific implementation")
    
    def get_encryption_info(self) -> Dict[str, Any]:
        """Get information about encryption configuration."""
        return {
            'algorithm': 'Fernet (AES-128)',
            'key_derivation': 'PBKDF2-SHA256',
            'key_rotation_interval_days': self.key_rotation_interval.days,
            'last_rotation': self.last_rotation.isoformat(),
            'needs_rotation': self.should_rotate_keys()
        }


class SecureSecrets:
    """Secure storage and retrieval of secrets."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption = encryption_manager
        self.secrets_file = Path('.secrets/secrets.enc')
        self.secrets_cache = {}
        self._load_secrets()
    
    def _load_secrets(self) -> None:
        """Load secrets from encrypted file."""
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    encrypted_data = f.read()
                
                self.secrets_cache = self.encryption.decrypt_dict(encrypted_data)
            except Exception as e:
                logger.error(f"Failed to load secrets: {e}")
                self.secrets_cache = {}
        else:
            self.secrets_cache = {}
    
    def _save_secrets(self) -> None:
        """Save secrets to encrypted file."""
        try:
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            encrypted_data = self.encryption.encrypt_dict(self.secrets_cache)
            
            with open(self.secrets_file, 'w') as f:
                f.write(encrypted_data)
            
            os.chmod(self.secrets_file, 0o600)
        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value."""
        # Check environment first
        env_key = f"NIGHTSCAN_{key.upper()}"
        env_value = os.environ.get(env_key)
        if env_value:
            return env_value
        
        # Check cache
        return self.secrets_cache.get(key, default)
    
    def set_secret(self, key: str, value: str) -> None:
        """Set a secret value."""
        self.secrets_cache[key] = value
        self._save_secrets()
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        if key in self.secrets_cache:
            del self.secrets_cache[key]
            self._save_secrets()
            return True
        return False
    
    def list_secrets(self) -> list:
        """List all secret keys (not values)."""
        return list(self.secrets_cache.keys())