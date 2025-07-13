"""
Enhanced Secrets Management System for NightScan.

This module provides secure handling of secrets with support for:
- Environment variable validation
- Secret rotation
- Encryption at rest
- Integration with external secret managers
"""

import os
import json
import secrets
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""
    pass


class SecretValidationError(Exception):
    """Raised when a secret fails validation."""
    pass


class SecretsManager:
    """Enhanced secrets management with encryption and validation."""
    
    def __init__(self, config_path: Optional[str] = None, use_encryption: bool = True):
        """
        Initialize the secrets manager.
        
        Args:
            config_path: Path to secrets configuration file
            use_encryption: Whether to encrypt secrets at rest
        """
        self.config_path = config_path or os.environ.get('SECRETS_CONFIG_PATH')
        self.use_encryption = use_encryption
        self._cipher = None
        self._secrets_cache = {}
        self._rotation_schedule = {}
        
        # Initialize encryption if enabled
        if self.use_encryption:
            self._init_encryption()
            
    def _init_encryption(self):
        """Initialize encryption cipher."""
        # Get or generate master key
        master_key = os.environ.get('NIGHTSCAN_MASTER_KEY')
        if not master_key:
            # Generate from machine-specific data for development
            # In production, this should come from a secure source
            import uuid
            machine_id = str(uuid.getnode()).encode()
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'nightscan-salt',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(machine_id))
            self._cipher = Fernet(key)
            logger.warning("Using machine-generated encryption key. Set NIGHTSCAN_MASTER_KEY for production.")
        else:
            self._cipher = Fernet(master_key.encode())
            
    def get_secret(self, key: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            key: Secret key name
            required: Whether the secret is required
            default: Default value if not found
            
        Returns:
            Secret value or default
            
        Raises:
            SecretNotFoundError: If required secret is not found
        """
        # Check cache first
        if key in self._secrets_cache:
            return self._secrets_cache[key]
            
        # Try environment variables
        value = os.environ.get(key)
        
        # Try configuration file
        if not value and self.config_path:
            value = self._get_from_config(key)
            
        # Try external secret managers
        if not value:
            value = self._get_from_external(key)
            
        # Handle not found
        if not value:
            if required and default is None:
                raise SecretNotFoundError(f"Required secret '{key}' not found")
            value = default
            
        # Validate and cache
        if value:
            self._validate_secret(key, value)
            self._secrets_cache[key] = value
            
        return value
        
    def _get_from_config(self, key: str) -> Optional[str]:
        """Get secret from configuration file."""
        if not os.path.exists(self.config_path):
            return None
            
        try:
            with open(self.config_path, 'r') as f:
                if self.use_encryption:
                    encrypted_data = f.read()
                    decrypted_data = self._cipher.decrypt(encrypted_data.encode())
                    config = json.loads(decrypted_data)
                else:
                    config = json.load(f)
                    
            return config.get(key)
        except Exception as e:
            logger.error(f"Error reading config file: {e}")
            return None
            
    def _get_from_external(self, key: str) -> Optional[str]:
        """Get secret from external secret manager."""
        # AWS Secrets Manager
        if os.environ.get('USE_AWS_SECRETS_MANAGER'):
            return self._get_from_aws_secrets_manager(key)
            
        # HashiCorp Vault
        if os.environ.get('VAULT_ADDR'):
            return self._get_from_vault(key)
            
        # Azure Key Vault
        if os.environ.get('AZURE_KEY_VAULT_NAME'):
            return self._get_from_azure_key_vault(key)
            
        return None
        
    def _validate_secret(self, key: str, value: str):
        """Validate a secret value."""
        # Check for common weak values
        weak_values = [
            'password', '123456', 'admin', 'default', 'test',
            'your-secret-key-here', 'change-me', 'todo', 'fixme'
        ]
        
        if value.lower() in weak_values:
            raise SecretValidationError(f"Secret '{key}' contains a weak value")
            
        # Check minimum length for certain keys
        min_lengths = {
            'SECRET_KEY': 32,
            'CSRF_SECRET_KEY': 32,
            'JWT_SECRET_KEY': 32,
            'DB_PASSWORD': 16,
            'REDIS_PASSWORD': 16,
        }
        
        if key in min_lengths and len(value) < min_lengths[key]:
            raise SecretValidationError(
                f"Secret '{key}' must be at least {min_lengths[key]} characters"
            )
            
    def set_secret(self, key: str, value: str, rotate_days: Optional[int] = None):
        """
        Set a secret value.
        
        Args:
            key: Secret key name
            value: Secret value
            rotate_days: Days until rotation required
        """
        # Validate before setting
        self._validate_secret(key, value)
        
        # Update cache
        self._secrets_cache[key] = value
        
        # Set rotation schedule if specified
        if rotate_days:
            self._rotation_schedule[key] = datetime.now() + timedelta(days=rotate_days)
            
        # Persist if config path is set
        if self.config_path:
            self._save_to_config()
            
    def _save_to_config(self):
        """Save secrets to configuration file."""
        config = {
            'secrets': self._secrets_cache,
            'rotation_schedule': {
                k: v.isoformat() for k, v in self._rotation_schedule.items()
            }
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            if self.use_encryption:
                encrypted_data = self._cipher.encrypt(json.dumps(config).encode())
                f.write(encrypted_data.decode())
            else:
                json.dump(config, f, indent=2)
                
        # Set secure permissions
        os.chmod(self.config_path, 0o600)
        
    def generate_secret(self, length: int = 32) -> str:
        """Generate a cryptographically secure secret."""
        return secrets.token_urlsafe(length)
        
    def rotate_secret(self, key: str) -> str:
        """
        Rotate a secret to a new value.
        
        Args:
            key: Secret key to rotate
            
        Returns:
            New secret value
        """
        new_value = self.generate_secret()
        old_value = self.get_secret(key, required=False)
        
        # Store old value for rollback
        if old_value:
            self._secrets_cache[f"{key}_OLD"] = old_value
            
        # Set new value
        self.set_secret(key, new_value)
        
        logger.info(f"Rotated secret '{key}'")
        return new_value
        
    def check_rotation_needed(self) -> List[str]:
        """Check which secrets need rotation."""
        needs_rotation = []
        now = datetime.now()
        
        for key, rotation_date in self._rotation_schedule.items():
            if isinstance(rotation_date, str):
                rotation_date = datetime.fromisoformat(rotation_date)
                
            if rotation_date <= now:
                needs_rotation.append(key)
                
        return needs_rotation
        
    def get_database_url(self, 
                        host: Optional[str] = None,
                        port: Optional[int] = None,
                        name: Optional[str] = None,
                        user: Optional[str] = None) -> str:
        """Build database URL from components."""
        host = host or os.environ.get('DB_HOST', 'localhost')
        port = port or int(os.environ.get('DB_PORT', '5432'))
        name = name or os.environ.get('DB_NAME', 'nightscan')
        user = user or os.environ.get('DB_USER', 'nightscan')
        password = self.get_secret('DB_PASSWORD')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
        
    def get_redis_url(self,
                     host: Optional[str] = None,
                     port: Optional[int] = None,
                     db: Optional[int] = None) -> str:
        """Build Redis URL from components."""
        host = host or os.environ.get('REDIS_HOST', 'localhost')
        port = port or int(os.environ.get('REDIS_PORT', '6379'))
        db = db or int(os.environ.get('REDIS_DB', '0'))
        password = self.get_secret('REDIS_PASSWORD', required=False)
        
        if password:
            return f"redis://:{password}@{host}:{port}/{db}"
        return f"redis://{host}:{port}/{db}"
        
    # External secret manager integrations (stubs)
    def _get_from_aws_secrets_manager(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            import boto3
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId=f"nightscan/{key}")
            return response.get('SecretString')
        except Exception as e:
            logger.error(f"Error getting secret from AWS: {e}")
            return None
            
    def _get_from_vault(self, key: str) -> Optional[str]:
        """Get secret from HashiCorp Vault."""
        try:
            import hvac
            client = hvac.Client(url=os.environ['VAULT_ADDR'])
            client.token = os.environ.get('VAULT_TOKEN')
            response = client.secrets.kv.v2.read_secret_version(
                path=f"nightscan/{key}"
            )
            return response['data']['data'].get('value')
        except Exception as e:
            logger.error(f"Error getting secret from Vault: {e}")
            return None
            
    def _get_from_azure_key_vault(self, key: str) -> Optional[str]:
        """Get secret from Azure Key Vault."""
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            
            vault_name = os.environ['AZURE_KEY_VAULT_NAME']
            vault_url = f"https://{vault_name}.vault.azure.net"
            
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=vault_url, credential=credential)
            
            secret = client.get_secret(key.replace('_', '-'))
            return secret.value
        except Exception as e:
            logger.error(f"Error getting secret from Azure: {e}")
            return None


# Global instance
_secrets_manager = None


def get_secrets_manager() -> SecretsManager:
    """Get or create global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions
def get_secret(key: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    """Get a secret value."""
    return get_secrets_manager().get_secret(key, required, default)


def set_secret(key: str, value: str, rotate_days: Optional[int] = None):
    """Set a secret value."""
    get_secrets_manager().set_secret(key, value, rotate_days)


def generate_secret(length: int = 32) -> str:
    """Generate a cryptographically secure secret."""
    return get_secrets_manager().generate_secret(length)


def rotate_secret(key: str) -> str:
    """Rotate a secret to a new value."""
    return get_secrets_manager().rotate_secret(key)