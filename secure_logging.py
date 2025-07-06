"""
Secure Logging System for NightScan
Implements secure logging with sensitive data protection.
"""

import logging
import re
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime

class SecureLogger:
    """Secure logging with PII protection."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Sensitive patterns to mask
        self.sensitive_patterns = [
            (r'password["\s]*[:=]["\s]*([^"\s,}]+)', 'password="***"'),
            (r'api_key["\s]*[:=]["\s]*([^"\s,}]+)', 'api_key="***"'),
            (r'token["\s]*[:=]["\s]*([^"\s,}]+)', 'token="***"'),
            (r'secret["\s]*[:=]["\s]*([^"\s,}]+)', 'secret="***"'),
            (r'Authorization:\s*Bearer\s+([^\s]+)', 'Authorization: Bearer ***'),
            (r'(\d{16})', lambda m: f"****{m.group(1)[-4:]}"),  # Credit card numbers
            (r'(\d{3}-\d{2}-\d{4})', lambda m: f"***-**-{m.group(1)[-4:]}"),  # SSN
        ]
        
    def _sanitize_message(self, message: str) -> str:
        """Remove sensitive information from log messages."""
        for pattern, replacement in self.sensitive_patterns:
            if callable(replacement):
                message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
            else:
                message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
        return message
        
    def _create_security_context(self) -> Dict[str, Any]:
        """Create security context for logs."""
        from flask import request, g
        context = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'nightscan'
        }
        
        try:
            if request:
                context.update({
                    'ip': request.remote_addr,
                    'user_agent': request.headers.get('User-Agent', '')[:200],
                    'method': request.method,
                    'path': request.path,
                    'user_id': getattr(g, 'user_id', None)
                })
        except RuntimeError:
            # Outside request context
            pass
            
        return context
        
    def security_event(self, event_type: str, details: Dict[str, Any], level: str = 'INFO'):
        """Log security-related events."""
        context = self._create_security_context()
        
        log_entry = {
            'event_type': event_type,
            'level': level,
            'details': details,
            'context': context
        }
        
        sanitized_message = self._sanitize_message(json.dumps(log_entry))
        
        if level == 'CRITICAL':
            self.logger.critical(f"SECURITY: {sanitized_message}")
        elif level == 'ERROR':
            self.logger.error(f"SECURITY: {sanitized_message}")
        elif level == 'WARNING':
            self.logger.warning(f"SECURITY: {sanitized_message}")
        else:
            self.logger.info(f"SECURITY: {sanitized_message}")
            
    def audit_log(self, action: str, resource: str, result: str, details: Optional[Dict] = None):
        """Log audit events."""
        self.security_event('AUDIT', {
            'action': action,
            'resource': resource,
            'result': result,
            'details': details or {}
        })

def get_security_logger(name: str = 'nightscan.security') -> SecureLogger:
    """Get secure logger instance."""
    return SecureLogger(name)
