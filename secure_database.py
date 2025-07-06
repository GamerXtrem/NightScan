"""
Secure Database Utilities for NightScan
Prevents SQL injection and implements secure query patterns.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class SecureQueryBuilder:
    """Secure query builder to prevent SQL injection."""
    
    def __init__(self, session: Session):
        self.session = session
        
    def safe_execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute query with parameterized inputs."""
        try:
            if params:
                # Use SQLAlchemy's text() with bound parameters
                result = self.session.execute(text(query), params)
            else:
                result = self.session.execute(text(query))
            return result
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise
            
    def safe_select(self, table: str, columns: List[str], 
                   where_conditions: Optional[Dict[str, Any]] = None,
                   order_by: Optional[str] = None,
                   limit: Optional[int] = None) -> Any:
        """Build safe SELECT query."""
        # Validate table and column names (whitelist approach)
        if not self._validate_identifier(table):
            raise ValueError(f"Invalid table name: {table}")
            
        validated_columns = []
        for col in columns:
            if self._validate_identifier(col):
                validated_columns.append(col)
            else:
                raise ValueError(f"Invalid column name: {col}")
                
        # Build query with validated identifiers
        query = f"SELECT {', '.join(validated_columns)} FROM {table}"
        params = {}
        
        if where_conditions:
            where_parts = []
            for i, (column, value) in enumerate(where_conditions.items()):
                if not self._validate_identifier(column):
                    raise ValueError(f"Invalid column name in WHERE: {column}")
                param_name = f"param_{i}"
                where_parts.append(f"{column} = :{param_name}")
                params[param_name] = value
            query += f" WHERE {' AND '.join(where_parts)}"
            
        if order_by and self._validate_identifier(order_by):
            query += f" ORDER BY {order_by}"
            
        if limit and isinstance(limit, int) and limit > 0:
            query += f" LIMIT {limit}"
            
        return self.safe_execute(query, params)
        
    def safe_insert(self, table: str, data: Dict[str, Any]) -> Any:
        """Build safe INSERT query."""
        if not self._validate_identifier(table):
            raise ValueError(f"Invalid table name: {table}")
            
        validated_columns = []
        params = {}
        
        for i, (column, value) in enumerate(data.items()):
            if self._validate_identifier(column):
                validated_columns.append(column)
                params[f"param_{i}"] = value
            else:
                raise ValueError(f"Invalid column name: {column}")
                
        placeholders = [f":param_{i}" for i in range(len(validated_columns))]
        query = f"INSERT INTO {table} ({', '.join(validated_columns)}) VALUES ({', '.join(placeholders)})"
        
        return self.safe_execute(query, params)
        
    def safe_update(self, table: str, data: Dict[str, Any], 
                   where_conditions: Dict[str, Any]) -> Any:
        """Build safe UPDATE query."""
        if not self._validate_identifier(table):
            raise ValueError(f"Invalid table name: {table}")
            
        set_parts = []
        params = {}
        
        # Build SET clause
        for i, (column, value) in enumerate(data.items()):
            if self._validate_identifier(column):
                param_name = f"set_param_{i}"
                set_parts.append(f"{column} = :{param_name}")
                params[param_name] = value
            else:
                raise ValueError(f"Invalid column name: {column}")
                
        # Build WHERE clause
        where_parts = []
        for i, (column, value) in enumerate(where_conditions.items()):
            if self._validate_identifier(column):
                param_name = f"where_param_{i}"
                where_parts.append(f"{column} = :{param_name}")
                params[param_name] = value
            else:
                raise ValueError(f"Invalid column name in WHERE: {column}")
                
        query = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        
        return self.safe_execute(query, params)
        
    def _validate_identifier(self, identifier: str) -> bool:
        """Validate SQL identifier (table/column name)."""
        # Allow only alphanumeric characters and underscores
        # Must start with letter or underscore
        import re
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        return bool(re.match(pattern, identifier)) and len(identifier) <= 63
        
class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_user_input(input_value: str, max_length: int = 255,
                           allowed_chars: Optional[str] = None) -> str:
        """Validate and sanitize user input."""
        if not isinstance(input_value, str):
            raise ValueError("Input must be a string")
            
        # Remove null bytes and control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', input_value)
        
        # Limit length
        if len(cleaned) > max_length:
            raise ValueError(f"Input too long (max {max_length} characters)")
            
        # Check allowed characters
        if allowed_chars:
            if not re.match(f'^[{re.escape(allowed_chars)}]*$', cleaned):
                raise ValueError("Input contains invalid characters")
                
        return cleaned.strip()
        
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email address."""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        email = email.strip().lower()
        
        if not re.match(email_pattern, email):
            raise ValueError("Invalid email address")
            
        if len(email) > 254:
            raise ValueError("Email address too long")
            
        return email
        
    @staticmethod
    def validate_filename(filename: str) -> str:
        """Validate and sanitize filename."""
        if not filename:
            raise ValueError("Filename cannot be empty")
            
        # Remove path separators and dangerous characters
        dangerous_chars = r'[<>:"/\|?*\x00-\x1f]'
        cleaned = re.sub(dangerous_chars, '', filename)
        
        # Remove leading/trailing dots and spaces
        cleaned = cleaned.strip('. ')
        
        if not cleaned:
            raise ValueError("Invalid filename")
            
        if len(cleaned) > 255:
            raise ValueError("Filename too long")
            
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        if cleaned.upper() in reserved_names:
            raise ValueError("Reserved filename")
            
        return cleaned

def get_secure_query_builder(session: Session) -> SecureQueryBuilder:
    """Get secure query builder instance."""
    return SecureQueryBuilder(session)
