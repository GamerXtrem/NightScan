"""
Secure Secrets Management for NightScan
Replaces hardcoded secrets with environment-based configuration.
"""

import os
import base64
import secrets
from cryptography.fernet import Fernet
from typing import Optional, Dict, Any

class SecretsManager:
    """Secure secrets management system."""
    
    def __init__(self):
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get encryption key from environment or create new one."""
        key_env = os.environ.get('NIGHTSCAN_ENCRYPTION_KEY')
        if key_env:
            return base64.urlsafe_b64decode(key_env.encode())
        
        # Generate new key (should be stored securely in production)
        key = Fernet.generate_key()
        print(f"⚠️  Generated new encryption key. Store this securely:")
        print(f"   NIGHTSCAN_ENCRYPTION_KEY={base64.urlsafe_b64encode(key).decode()}")
        return key
        
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from environment with optional decryption."""
        # Try environment variable first
        value = os.environ.get(key, default)
        if not value:
            return None
            
        # Check if value is encrypted (starts with 'encrypted:')
        if value.startswith('encrypted:'):
            try:
                encrypted_data = value[10:]  # Remove 'encrypted:' prefix
                decrypted = self.cipher.decrypt(encrypted_data.encode())
                return decrypted.decode()
            except Exception:
                print(f"⚠️  Failed to decrypt secret: {key}")
                return None
                
        return value
        
    def encrypt_secret(self, value: str) -> str:
        """Encrypt a secret value."""
        encrypted = self.cipher.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"
        
    def generate_secure_secret(self, length: int = 32) -> str:
        """Generate a cryptographically secure secret."""
        return secrets.token_urlsafe(length)
        
    # Database credentials
    def get_database_url(self) -> str:
        """Get secure database URL."""
        host = self.get_secret('DB_HOST', 'localhost')
        port = self.get_secret('DB_PORT', '5432')
        name = self.get_secret('DB_NAME', 'nightscan')
        user = self.get_secret('DB_USER', 'nightscan')
        password = self.get_secret('DB_PASSWORD')
        
        if not password:
            raise ValueError("DB_PASSWORD environment variable required")
            
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        
    # Flask secrets
    def get_flask_secret_key(self) -> str:
        """Get Flask secret key."""
        secret = self.get_secret('FLASK_SECRET_KEY')
        if not secret:
            secret = self.generate_secure_secret(64)
            print(f"⚠️  Generated Flask secret key. Set environment variable:")
            print(f"   FLASK_SECRET_KEY={secret}")
        return secret
        
    def get_csrf_secret_key(self) -> str:
        """Get CSRF secret key."""
        secret = self.get_secret('CSRF_SECRET_KEY')
        if not secret:
            secret = self.generate_secure_secret(64)
            print(f"⚠️  Generated CSRF secret key. Set environment variable:")
            print(f"   CSRF_SECRET_KEY={secret}")
        return secret
        
    # API keys
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for external service."""
        return self.get_secret(f'{service.upper()}_API_KEY')
        
    # JWT secrets
    def get_jwt_secret(self) -> str:
        """Get JWT signing secret."""
        secret = self.get_secret('JWT_SECRET_KEY')
        if not secret:
            secret = self.generate_secure_secret(64)
            print(f"⚠️  Generated JWT secret key. Set environment variable:")
            print(f"   JWT_SECRET_KEY={secret}")
        return secret
        
    # Redis credentials
    def get_redis_url(self) -> str:
        """Get secure Redis URL."""
        host = self.get_secret('REDIS_HOST', 'localhost')
        port = self.get_secret('REDIS_PORT', '6379')
        password = self.get_secret('REDIS_PASSWORD')
        db = self.get_secret('REDIS_DB', '0')
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"

# Global secrets manager instance
_secrets_manager = None

def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

# Convenience functions
def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get secret using global secrets manager."""
    return get_secrets_manager().get_secret(key, default)

def get_database_url() -> str:
    """Get database URL using global secrets manager."""
    return get_secrets_manager().get_database_url()

def get_flask_secret_key() -> str:
    """Get Flask secret key using global secrets manager."""
    return get_secrets_manager().get_flask_secret_key()
