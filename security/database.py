"""
Database Security Module

Provides database security features and query protection.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import hashlib
import secrets
from pathlib import Path
import sqlite3
import json

logger = logging.getLogger(__name__)


class DatabaseSecurity:
    """Handles database security and query protection."""
    
    def __init__(self, config):
        self.config = config
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            re.compile(r'(\bunion\b.*\bselect\b)', re.IGNORECASE),
            re.compile(r'(\bdelete\b.*\bfrom\b)', re.IGNORECASE),
            re.compile(r'(\bdrop\b.*\b(table|database)\b)', re.IGNORECASE),
            re.compile(r'(\binsert\b.*\binto\b)', re.IGNORECASE),
            re.compile(r'(\bupdate\b.*\bset\b)', re.IGNORECASE),
            re.compile(r'(\bexec\b|\bexecute\b)', re.IGNORECASE),
            re.compile(r'(\bscript\b)', re.IGNORECASE),
            re.compile(r'(-{2,}|/\*|\*/)', re.IGNORECASE),  # SQL comments
            re.compile(r"(\bor\b.*=.*)", re.IGNORECASE),  # OR conditions
            re.compile(r"('\s*;\s*--)", re.IGNORECASE),  # Statement termination
        ]
        
        # Parameterized query templates
        self.query_templates = {}
        
        # Database encryption settings
        self.encrypt_sensitive_fields = getattr(config.security, 'encrypt_db_fields', True)
        self.encrypted_fields = set(getattr(config.security, 'encrypted_fields', [
            'password', 'api_key', 'secret', 'token', 'ssn', 'credit_card'
        ]))
    
    def init_app(self, app) -> None:
        """Initialize with Flask app."""
        self.app = app
        logger.info("Database security initialized")
    
    def sanitize_identifier(self, identifier: str) -> str:
        """
        Sanitize database identifier (table/column name).
        
        Args:
            identifier: Identifier to sanitize
            
        Returns:
            Sanitized identifier
        """
        # Remove all non-alphanumeric characters except underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', identifier)
        
        # Ensure it starts with letter or underscore
        if sanitized and not re.match(r'^[a-zA-Z_]', sanitized):
            sanitized = '_' + sanitized
        
        # Limit length
        return sanitized[:64]
    
    def validate_query_parameter(self, param: Any, param_type: str = 'string') -> Tuple[bool, Any]:
        """
        Validate and sanitize query parameter.
        
        Args:
            param: Parameter to validate
            param_type: Expected parameter type
            
        Returns:
            Tuple of (is_valid, sanitized_value)
        """
        if param is None:
            return True, None
        
        if param_type == 'string':
            if not isinstance(param, str):
                return False, None
            
            # Check for SQL injection
            if self._contains_sql_injection(param):
                return False, None
            
            # Basic sanitization
            sanitized = param.strip()
            
            # Remove null bytes
            sanitized = sanitized.replace('\x00', '')
            
            return True, sanitized
        
        elif param_type == 'integer':
            try:
                return True, int(param)
            except (ValueError, TypeError):
                return False, None
        
        elif param_type == 'float':
            try:
                return True, float(param)
            except (ValueError, TypeError):
                return False, None
        
        elif param_type == 'boolean':
            if isinstance(param, bool):
                return True, param
            if isinstance(param, str):
                if param.lower() in ['true', '1', 'yes']:
                    return True, True
                elif param.lower() in ['false', '0', 'no']:
                    return True, False
            return False, None
        
        elif param_type == 'list':
            if not isinstance(param, list):
                return False, None
            
            # Validate each item
            sanitized_list = []
            for item in param:
                is_valid, sanitized_item = self.validate_query_parameter(item, 'string')
                if not is_valid:
                    return False, None
                sanitized_list.append(sanitized_item)
            
            return True, sanitized_list
        
        return False, None
    
    def build_safe_query(self, query_template: str, params: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        Build safe parameterized query.
        
        Args:
            query_template: Query template with placeholders
            params: Parameters to insert
            
        Returns:
            Tuple of (query, parameter_values)
        """
        # Validate query template
        if self._contains_sql_injection(query_template):
            raise ValueError("Query template contains suspicious patterns")
        
        # Extract placeholders
        placeholders = re.findall(r':{(\w+)}', query_template)
        
        # Build parameterized query
        query = query_template
        values = []
        
        for placeholder in placeholders:
            if placeholder not in params:
                raise ValueError(f"Missing parameter: {placeholder}")
            
            # Replace with ? for SQLite
            query = query.replace(f':{{{placeholder}}}', '?')
            values.append(params[placeholder])
        
        return query, values
    
    def encrypt_field(self, value: Any, field_name: str) -> str:
        """
        Encrypt sensitive field value.
        
        Args:
            value: Value to encrypt
            field_name: Field name (for key derivation)
            
        Returns:
            Encrypted value
        """
        if not self.encrypt_sensitive_fields:
            return str(value)
        
        # Import encryption manager
        from .encryption import EncryptionManager
        encryption = EncryptionManager(self.config)
        
        # Convert to string if needed
        str_value = str(value)
        
        # Encrypt with field-specific context
        encrypted = encryption.encrypt(f"{field_name}:{str_value}")
        
        return f"encrypted:{encrypted}"
    
    def decrypt_field(self, encrypted_value: str, field_name: str) -> str:
        """
        Decrypt sensitive field value.
        
        Args:
            encrypted_value: Encrypted value
            field_name: Field name
            
        Returns:
            Decrypted value
        """
        if not encrypted_value.startswith('encrypted:'):
            return encrypted_value
        
        # Import encryption manager
        from .encryption import EncryptionManager
        encryption = EncryptionManager(self.config)
        
        # Remove prefix and decrypt
        encrypted_data = encrypted_value[10:]
        decrypted = encryption.decrypt(encrypted_data)
        
        # Remove field name prefix
        if decrypted.startswith(f"{field_name}:"):
            return decrypted[len(field_name) + 1:]
        
        return decrypted
    
    def prepare_insert_data(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for safe insertion.
        
        Args:
            table: Table name
            data: Data to insert
            
        Returns:
            Prepared data
        """
        prepared = {}
        
        for field, value in data.items():
            # Sanitize field name
            safe_field = self.sanitize_identifier(field)
            
            # Check if field should be encrypted
            if field.lower() in self.encrypted_fields:
                prepared[safe_field] = self.encrypt_field(value, field)
            else:
                # Validate value
                is_valid, sanitized = self.validate_query_parameter(value)
                if not is_valid:
                    raise ValueError(f"Invalid value for field {field}")
                prepared[safe_field] = sanitized
        
        return prepared
    
    def execute_safe_query(self, db_path: str, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Execute query safely with proper error handling.
        
        Args:
            db_path: Database file path
            query: Query to execute
            params: Query parameters
            
        Returns:
            Query results
        """
        # Validate query
        if self._contains_sql_injection(query):
            raise ValueError("Query contains suspicious patterns")
        
        results = []
        conn = None
        
        try:
            # Connect with security settings
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # Set security pragmas
            conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
            conn.execute("PRAGMA foreign_keys=ON")   # Enforce foreign keys
            
            # Execute query
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch results
            if query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
            else:
                conn.commit()
                results = [{'affected_rows': cursor.rowcount}]
            
            cursor.close()
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        
        finally:
            if conn:
                conn.close()
        
        return results
    
    def create_secure_connection(self, db_path: str) -> sqlite3.Connection:
        """
        Create secure database connection.
        
        Args:
            db_path: Database file path
            
        Returns:
            Secure connection
        """
        # Ensure database directory exists
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect with security settings
        conn = sqlite3.connect(db_path)
        
        # Set security pragmas
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA secure_delete=ON")
        conn.execute("PRAGMA auto_vacuum=FULL")
        
        # Set permissions on database file
        if db_file.exists():
            db_file.chmod(0o600)  # Read/write for owner only
        
        return conn
    
    def backup_database(self, db_path: str, backup_dir: str) -> str:
        """
        Create secure database backup.
        
        Args:
            db_path: Database to backup
            backup_dir: Backup directory
            
        Returns:
            Backup file path
        """
        # Create backup filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{timestamp}_{secrets.token_hex(4)}.db"
        backup_path = Path(backup_dir) / backup_name
        
        # Ensure backup directory exists
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup
        conn = sqlite3.connect(db_path)
        backup_conn = sqlite3.connect(str(backup_path))
        
        with backup_conn:
            conn.backup(backup_conn)
        
        conn.close()
        backup_conn.close()
        
        # Set secure permissions
        backup_path.chmod(0o600)
        
        logger.info(f"Database backed up to {backup_path}")
        return str(backup_path)
    
    def _contains_sql_injection(self, text: str) -> bool:
        """Check if text contains SQL injection patterns."""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check patterns
        for pattern in self.sql_injection_patterns:
            if pattern.search(text):
                return True
        
        # Check for suspicious character sequences
        suspicious_sequences = [
            "' or '", '" or "', "' and '", '" and "',
            "'='", '"="', "1=1", "1' = '1", '1" = "1'
        ]
        
        for seq in suspicious_sequences:
            if seq in text_lower:
                return True
        
        return False
    
    def audit_database_access(self, user_id: str, operation: str, 
                            table: str, query: str, success: bool) -> None:
        """Audit database access."""
        # Import secure logger
        from .logging import SecureLogger
        logger = SecureLogger(self.config)
        
        # Log audit event
        logger.log_audit_event(
            user_id,
            f"DB_{operation.upper()}",
            table,
            'SUCCESS' if success else 'FAILURE',
            {
                'query_hash': hashlib.sha256(query.encode()).hexdigest()[:16],
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    def get_query_plan(self, db_path: str, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """
        Get query execution plan for optimization.
        
        Args:
            db_path: Database path
            query: Query to analyze
            params: Query parameters
            
        Returns:
            Query plan
        """
        plan_query = f"EXPLAIN QUERY PLAN {query}"
        return self.execute_safe_query(db_path, plan_query, params)
    
    def analyze_table(self, db_path: str, table: str) -> Dict[str, Any]:
        """
        Analyze table for optimization.
        
        Args:
            db_path: Database path
            table: Table name
            
        Returns:
            Table statistics
        """
        safe_table = self.sanitize_identifier(table)
        
        # Get table info
        info_query = f"PRAGMA table_info({safe_table})"
        columns = self.execute_safe_query(db_path, info_query)
        
        # Get row count
        count_query = f"SELECT COUNT(*) as count FROM {safe_table}"
        count_result = self.execute_safe_query(db_path, count_query)
        row_count = count_result[0]['count'] if count_result else 0
        
        # Get indexes
        index_query = f"PRAGMA index_list({safe_table})"
        indexes = self.execute_safe_query(db_path, index_query)
        
        return {
            'table': table,
            'row_count': row_count,
            'columns': columns,
            'indexes': indexes
        }