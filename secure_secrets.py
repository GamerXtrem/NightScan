"""
Secure Secrets Management for NightScan - Wrapper for centralized config.

This module now uses the centralized configuration system.
It's kept for backward compatibility.
"""

import os
import secrets
from config import get_config

# Get centralized config
_config = get_config()

class SecretsManager:
    """Wrapper for backward compatibility."""
    
    def __init__(self):
        self.config = _config
        
    def get_secret(self, key: str, default: str = None) -> str:
        """Get secret from environment."""
        return os.environ.get(key, default)
        
    def generate_secure_secret(self, length: int = 32) -> str:
        """Generate a cryptographically secure secret."""
        return secrets.token_urlsafe(length)
        
    def get_database_url(self) -> str:
        """Get database URL."""
        # First try from config
        if self.config.database.uri:
            return self.config.database.uri
            
        # Then try environment variables
        host = os.environ.get('DB_HOST', 'localhost')
        port = os.environ.get('DB_PORT', '5432')
        name = os.environ.get('DB_NAME', 'nightscan')
        user = os.environ.get('DB_USER', 'nightscan')
        password = os.environ.get('DB_PASSWORD')
        
        if not password:
            raise ValueError("DB_PASSWORD environment variable required")
            
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        
    def get_flask_secret_key(self) -> str:
        """Get Flask secret key."""
        if self.config.security.secret_key:
            return self.config.security.secret_key
            
        secret = os.environ.get('FLASK_SECRET_KEY') or os.environ.get('SECRET_KEY')
        if not secret:
            secret = self.generate_secure_secret(64)
            print(f"⚠️  Generated Flask secret key. Set environment variable:")
            print(f"   FLASK_SECRET_KEY={secret}")
        return secret
        
    def get_csrf_secret_key(self) -> str:
        """Get CSRF secret key."""
        if self.config.security.csrf_secret_key:
            return self.config.security.csrf_secret_key
            
        secret = os.environ.get('CSRF_SECRET_KEY')
        if not secret:
            secret = self.generate_secure_secret(64)
            print(f"⚠️  Generated CSRF secret key. Set environment variable:")
            print(f"   CSRF_SECRET_KEY={secret}")
        return secret
        
    def get_jwt_secret(self) -> str:
        """Get JWT signing secret."""
        secret = os.environ.get('JWT_SECRET_KEY')
        if not secret:
            secret = self.generate_secure_secret(64)
            print(f"⚠️  Generated JWT secret key. Set environment variable:")
            print(f"   JWT_SECRET_KEY={secret}")
        return secret
        
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        if self.config.redis.url:
            return self.config.redis.url
            
        host = os.environ.get('REDIS_HOST', 'localhost')
        port = os.environ.get('REDIS_PORT', '6379')
        password = os.environ.get('REDIS_PASSWORD')
        db = os.environ.get('REDIS_DB', '0')
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"
        
    # Stub methods for features not in simple config
    def encrypt_secret(self, value: str) -> str:
        """Encryption not implemented in simple config."""
        return value
        
    def get_api_key(self, service: str) -> str:
        """Get API key for external service."""
        return os.environ.get(f'{service.upper()}_API_KEY')

# Global secrets manager instance
_secrets_manager = None

def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

# Convenience functions for backward compatibility
def get_secret(key: str, default: str = None) -> str:
    """Get secret using global secrets manager."""
    return get_secrets_manager().get_secret(key, default)

def get_database_url() -> str:
    """Get database URL using global secrets manager."""
    return get_secrets_manager().get_database_url()

def get_flask_secret_key() -> str:
    """Get Flask secret key using global secrets manager."""
    return get_secrets_manager().get_flask_secret_key()