"""
Input Validation and Sanitization Module

Provides comprehensive input validation and sanitization.
"""

import re
import logging
import html
import urllib.parse
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import ipaddress
import email_validator
import phonenumbers
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Handles input validation and sanitization."""
    
    def __init__(self, config):
        self.config = config
        
        # Compile regex patterns for performance
        self.patterns = {
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'alpha': re.compile(r'^[a-zA-Z]+$'),
            'numeric': re.compile(r'^[0-9]+$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$', re.IGNORECASE),
            'phone': re.compile(r'^\+?[1-9]\d{1,14}$'),  # E.164 format
            'username': re.compile(r'^[a-zA-Z0-9_-]{3,32}$'),
            'slug': re.compile(r'^[a-zA-Z0-9-]+$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE),
            'hex_color': re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'),
            'safe_filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'sql_identifier': re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
        }
        
        # XSS patterns to detect
        self.xss_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'<iframe', re.IGNORECASE),
            re.compile(r'<object', re.IGNORECASE),
            re.compile(r'<embed', re.IGNORECASE),
            re.compile(r'<form', re.IGNORECASE),
            re.compile(r'expression\s*\(', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'<meta', re.IGNORECASE)
        ]
    
    # String validation methods
    
    def validate_string(self, value: str, min_length: int = 0, max_length: int = 1000, 
                       pattern: Optional[str] = None, allowed_chars: Optional[str] = None) -> bool:
        """Validate a string against various criteria."""
        if not isinstance(value, str):
            return False
        
        # Length check
        if len(value) < min_length or len(value) > max_length:
            return False
        
        # Pattern check
        if pattern and pattern in self.patterns:
            if not self.patterns[pattern].match(value):
                return False
        
        # Allowed characters check
        if allowed_chars:
            if not all(c in allowed_chars for c in value):
                return False
        
        return True
    
    def validate_email(self, email: str) -> bool:
        """Validate email address."""
        try:
            # Use email-validator for comprehensive validation
            validation = email_validator.validate_email(email)
            return True
        except email_validator.EmailNotValidError:
            return False
    
    def validate_url(self, url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
        """Validate URL."""
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Check scheme
            if parsed.scheme not in allowed_schemes:
                return False
            
            # Check for basic structure
            if not parsed.netloc:
                return False
            
            # Additional checks
            if '..' in url or '//' in url[8:]:  # Skip initial http://
                return False
            
            return True
        except Exception:
            return False
    
    def validate_phone(self, phone: str, region: str = 'US') -> bool:
        """Validate phone number."""
        try:
            parsed = phonenumbers.parse(phone, region)
            return phonenumbers.is_valid_number(parsed)
        except Exception:
            return False
    
    def validate_ip_address(self, ip: str, version: Optional[int] = None) -> bool:
        """Validate IP address."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            
            if version:
                return ip_obj.version == version
            
            return True
        except ValueError:
            return False
    
    def validate_date(self, date_str: str, format: str = '%Y-%m-%d') -> bool:
        """Validate date string."""
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False
    
    def validate_json(self, json_str: str) -> bool:
        """Validate JSON string."""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False
    
    # Sanitization methods
    
    def sanitize_html(self, html_str: str, allowed_tags: Optional[List[str]] = None) -> str:
        """Sanitize HTML content."""
        # For now, escape all HTML
        # In production, use a library like bleach for selective tag allowance
        return html.escape(html_str)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path components
        filename = Path(filename).name
        
        # Replace dangerous characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Remove multiple dots
        filename = re.sub(r'\.+', '.', filename)
        
        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = Path(filename).stem, Path(filename).suffix
            filename = name[:max_length - len(ext)] + ext
        
        return filename
    
    def sanitize_sql_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifier (table/column name)."""
        # Remove all non-alphanumeric characters except underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', identifier)
        
        # Ensure it starts with a letter or underscore
        if sanitized and not re.match(r'^[a-zA-Z_]', sanitized):
            sanitized = '_' + sanitized
        
        return sanitized[:64]  # Limit length
    
    def sanitize_for_log(self, value: str, max_length: int = 100) -> str:
        """Sanitize value for safe logging."""
        # Remove control characters
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length] + '...'
        
        return value
    
    def remove_null_bytes(self, value: str) -> str:
        """Remove null bytes from string."""
        return value.replace('\x00', '')
    
    # Security checks
    
    def contains_xss(self, value: str) -> bool:
        """Check if value contains potential XSS."""
        for pattern in self.xss_patterns:
            if pattern.search(value):
                return True
        return False
    
    def contains_sql_injection(self, value: str) -> bool:
        """Check if value contains potential SQL injection."""
        sql_keywords = [
            'union', 'select', 'insert', 'update', 'delete', 'drop',
            'create', 'alter', 'exec', 'execute', 'script', '--', '/*', '*/'
        ]
        
        value_lower = value.lower()
        
        # Check for SQL keywords with word boundaries
        for keyword in sql_keywords:
            if re.search(r'\b' + keyword + r'\b', value_lower):
                # Check if it's in a suspicious context
                if any(op in value_lower for op in ['=', ';', '(', ')']):
                    return True
        
        # Check for common SQL injection patterns
        if re.search(r"(\d+)\s*=\s*\1", value):  # 1=1
            return True
        
        if re.search(r"'\s*or\s*'?\d*'?\s*=\s*'?\d*", value_lower):  # ' or '1'='1
            return True
        
        return False
    
    def contains_path_traversal(self, value: str) -> bool:
        """Check if value contains path traversal attempts."""
        patterns = [
            '../', '..\\',
            '%2e%2e/', '%2e%2e\\',
            '..%2f', '..%5c',
            '%252e%252e%252f',
            '..../', '....\\'
        ]
        
        value_lower = value.lower()
        return any(pattern in value_lower for pattern in patterns)
    
    # Data type validation
    
    def validate_integer(self, value: Any, min_val: Optional[int] = None, 
                        max_val: Optional[int] = None) -> bool:
        """Validate integer value."""
        try:
            int_val = int(value)
            
            if min_val is not None and int_val < min_val:
                return False
            
            if max_val is not None and int_val > max_val:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    def validate_float(self, value: Any, min_val: Optional[float] = None,
                      max_val: Optional[float] = None) -> bool:
        """Validate float value."""
        try:
            float_val = float(value)
            
            if min_val is not None and float_val < min_val:
                return False
            
            if max_val is not None and float_val > max_val:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    def validate_boolean(self, value: Any) -> bool:
        """Validate boolean value."""
        if isinstance(value, bool):
            return True
        
        if isinstance(value, str):
            return value.lower() in ['true', 'false', '1', '0', 'yes', 'no']
        
        if isinstance(value, (int, float)):
            return value in [0, 1]
        
        return False
    
    # Batch validation
    
    def validate_dict(self, data: Dict[str, Any], schema: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        Validate dictionary against schema.
        
        Schema format:
        {
            'field_name': {
                'type': 'string|integer|float|boolean|email|url|etc',
                'required': True/False,
                'min_length': int,
                'max_length': int,
                'min_value': number,
                'max_value': number,
                'pattern': 'pattern_name',
                'custom': callable
            }
        }
        """
        errors = {}
        
        for field, rules in schema.items():
            # Check required fields
            if rules.get('required', False) and field not in data:
                errors.setdefault(field, []).append('Field is required')
                continue
            
            if field not in data:
                continue
            
            value = data[field]
            field_type = rules.get('type', 'string')
            
            # Type validation
            if field_type == 'string':
                if not isinstance(value, str):
                    errors.setdefault(field, []).append('Must be a string')
                else:
                    min_len = rules.get('min_length', 0)
                    max_len = rules.get('max_length', 1000)
                    if not self.validate_string(value, min_len, max_len, rules.get('pattern')):
                        errors.setdefault(field, []).append('Invalid string format')
            
            elif field_type == 'integer':
                if not self.validate_integer(value, rules.get('min_value'), rules.get('max_value')):
                    errors.setdefault(field, []).append('Must be a valid integer')
            
            elif field_type == 'float':
                if not self.validate_float(value, rules.get('min_value'), rules.get('max_value')):
                    errors.setdefault(field, []).append('Must be a valid number')
            
            elif field_type == 'boolean':
                if not self.validate_boolean(value):
                    errors.setdefault(field, []).append('Must be a boolean')
            
            elif field_type == 'email':
                if not self.validate_email(value):
                    errors.setdefault(field, []).append('Must be a valid email')
            
            elif field_type == 'url':
                if not self.validate_url(value):
                    errors.setdefault(field, []).append('Must be a valid URL')
            
            # Custom validation
            if 'custom' in rules and callable(rules['custom']):
                try:
                    if not rules['custom'](value):
                        errors.setdefault(field, []).append('Custom validation failed')
                except Exception as e:
                    errors.setdefault(field, []).append(f'Validation error: {str(e)}')
        
        return errors
    
    def create_validator(self, **kwargs) -> Callable:
        """Create a custom validator function."""
        def validator(value):
            if 'type' in kwargs:
                if kwargs['type'] == 'string':
                    return self.validate_string(value, **{k: v for k, v in kwargs.items() if k != 'type'})
                # Add other types as needed
            
            return True
        
        return validator