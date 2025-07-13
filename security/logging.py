"""
Secure Logging Module

Provides secure logging with sensitive data filtering.
"""

import logging
import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import hashlib
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading
from collections import deque

logger = logging.getLogger(__name__)


class SecureLogger:
    """Secure logging with sensitive data filtering."""
    
    def __init__(self, config):
        self.config = config
        
        # Patterns for sensitive data
        self.sensitive_patterns = [
            # API keys and tokens
            (r'(api[_-]?key|apikey)[\s:=]+[\'"]?([\w-]+)[\'"]?', 'API_KEY'),
            (r'(token|access[_-]?token)[\s:=]+[\'"]?([\w-]+)[\'"]?', 'TOKEN'),
            (r'(secret|secret[_-]?key)[\s:=]+[\'"]?([\w-]+)[\'"]?', 'SECRET'),
            
            # Passwords
            (r'(password|passwd|pwd)[\s:=]+[\'"]?([^\s\'"]+)[\'"]?', 'PASSWORD'),
            
            # Credit cards
            (r'\b(?:\d[ -]*?){13,16}\b', 'CREDIT_CARD'),
            
            # Social Security Numbers
            (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
            
            # Email addresses (optional)
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),
            
            # IP addresses (optional)
            (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP_ADDRESS'),
            
            # JWT tokens
            (r'eyJ[A-Za-z0-9-_=]+\.eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_.+/=]+', 'JWT_TOKEN'),
            
            # Database URLs
            (r'(postgres|mysql|mongodb)://[^:]+:[^@]+@[^/]+', 'DATABASE_URL'),
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self.sensitive_patterns
        ]
        
        # Security event storage
        self.security_events = deque(maxlen=10000)
        self.event_lock = threading.Lock()
        
        # Setup log directories
        self.log_dir = Path(config.paths.logs)
        self.security_log_dir = self.log_dir / 'security'
        self.audit_log_dir = self.log_dir / 'audit'
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.security_log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup various loggers."""
        # Security logger
        self.security_logger = logging.getLogger('nightscan.security')
        self.security_logger.setLevel(logging.INFO)
        
        security_handler = RotatingFileHandler(
            self.security_log_dir / 'security.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        security_handler.setFormatter(self._get_security_formatter())
        self.security_logger.addHandler(security_handler)
        
        # Audit logger
        self.audit_logger = logging.getLogger('nightscan.audit')
        self.audit_logger.setLevel(logging.INFO)
        
        audit_handler = TimedRotatingFileHandler(
            self.audit_log_dir / 'audit.log',
            when='midnight',
            interval=1,
            backupCount=90  # Keep 90 days
        )
        audit_handler.setFormatter(self._get_audit_formatter())
        self.audit_logger.addHandler(audit_handler)
        
        # Access logger
        self.access_logger = logging.getLogger('nightscan.access')
        self.access_logger.setLevel(logging.INFO)
        
        access_handler = TimedRotatingFileHandler(
            self.log_dir / 'access.log',
            when='midnight',
            interval=1,
            backupCount=30
        )
        access_handler.setFormatter(self._get_access_formatter())
        self.access_logger.addHandler(access_handler)
    
    def _get_security_formatter(self) -> logging.Formatter:
        """Get formatter for security logs."""
        return logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_audit_formatter(self) -> logging.Formatter:
        """Get formatter for audit logs."""
        return logging.Formatter(
            '%(asctime)s - %(levelname)s - [AUDIT] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_access_formatter(self) -> logging.Formatter:
        """Get formatter for access logs."""
        return logging.Formatter(
            '%(asctime)s - %(remote_addr)s - "%(request_line)s" - %(status_code)s - %(response_size)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def init_app(self, app) -> None:
        """Initialize with Flask app."""
        self.app = app
        
        # Override Flask logger
        app.logger.addFilter(self.SensitiveDataFilter(self))
        
        logger.info("Secure logging initialized")
    
    def filter_sensitive_data(self, message: str) -> str:
        """Filter sensitive data from log message."""
        filtered_message = message
        
        for pattern, replacement in self.compiled_patterns:
            filtered_message = pattern.sub(f'[REDACTED-{replacement}]', filtered_message)
        
        return filtered_message
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = 'INFO') -> None:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        """
        # Create event record
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'severity': severity,
            'details': details,
            'event_id': self._generate_event_id()
        }
        
        # Store in memory
        with self.event_lock:
            self.security_events.append(event)
        
        # Log to file
        log_message = f"Security Event: {event_type} - {json.dumps(details)}"
        filtered_message = self.filter_sensitive_data(log_message)
        
        if severity == 'CRITICAL':
            self.security_logger.critical(filtered_message)
        elif severity == 'ERROR':
            self.security_logger.error(filtered_message)
        elif severity == 'WARNING':
            self.security_logger.warning(filtered_message)
        else:
            self.security_logger.info(filtered_message)
    
    def log_audit_event(self, user_id: Optional[str], action: str, 
                       resource: str, result: str, details: Optional[Dict] = None) -> None:
        """
        Log an audit event.
        
        Args:
            user_id: User performing action
            action: Action performed
            resource: Resource affected
            result: Result of action (SUCCESS, FAILURE)
            details: Additional details
        """
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id or 'anonymous',
            'action': action,
            'resource': resource,
            'result': result,
            'details': details or {},
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent()
        }
        
        # Filter sensitive data
        audit_json = json.dumps(audit_entry)
        filtered_audit = self.filter_sensitive_data(audit_json)
        
        self.audit_logger.info(filtered_audit)
    
    def log_access(self, request, response) -> None:
        """Log HTTP access."""
        # Create custom log record
        extra = {
            'remote_addr': self._get_client_ip(),
            'request_line': f"{request.method} {request.path} {request.environ.get('SERVER_PROTOCOL', '')}",
            'status_code': response.status_code,
            'response_size': response.content_length or 0
        }
        
        self.access_logger.info('', extra=extra)
    
    def log_authentication_attempt(self, username: str, success: bool, 
                                  method: str = 'password', details: Optional[Dict] = None) -> None:
        """Log authentication attempt."""
        event_details = {
            'username': self._hash_username(username),
            'method': method,
            'success': success,
            'ip_address': self._get_client_ip(),
            'user_agent': self._get_user_agent()
        }
        
        if details:
            event_details.update(details)
        
        self.log_security_event(
            'authentication_attempt',
            event_details,
            'INFO' if success else 'WARNING'
        )
        
        # Also log to audit
        self.log_audit_event(
            username,
            'LOGIN',
            'authentication',
            'SUCCESS' if success else 'FAILURE',
            event_details
        )
    
    def log_authorization_failure(self, user_id: str, resource: str, 
                                 action: str, required_permission: str) -> None:
        """Log authorization failure."""
        self.log_security_event(
            'authorization_failure',
            {
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'required_permission': required_permission,
                'ip_address': self._get_client_ip()
            },
            'WARNING'
        )
    
    def log_data_access(self, user_id: str, data_type: str, 
                       operation: str, record_id: Optional[str] = None) -> None:
        """Log data access for compliance."""
        self.log_audit_event(
            user_id,
            f'DATA_{operation.upper()}',
            f'{data_type}:{record_id}' if record_id else data_type,
            'SUCCESS'
        )
    
    def log_configuration_change(self, user_id: str, setting: str, 
                               old_value: Any, new_value: Any) -> None:
        """Log configuration changes."""
        # Filter sensitive values
        filtered_old = self.filter_sensitive_data(str(old_value))
        filtered_new = self.filter_sensitive_data(str(new_value))
        
        self.log_audit_event(
            user_id,
            'CONFIG_CHANGE',
            setting,
            'SUCCESS',
            {
                'old_value': filtered_old,
                'new_value': filtered_new
            }
        )
    
    def get_security_events(self, event_type: Optional[str] = None, 
                          severity: Optional[str] = None,
                          hours: int = 24) -> List[Dict]:
        """Get recent security events."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self.event_lock:
            events = list(self.security_events)
        
        # Filter events
        filtered_events = []
        for event in events:
            event_time = datetime.fromisoformat(event['timestamp'])
            if event_time < cutoff_time:
                continue
            
            if event_type and event['type'] != event_type:
                continue
            
            if severity and event['severity'] != severity:
                continue
            
            filtered_events.append(event)
        
        return filtered_events
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = datetime.utcnow().isoformat()
        random_data = os.urandom(8).hex()
        return hashlib.sha256(f"{timestamp}{random_data}".encode()).hexdigest()[:16]
    
    def _hash_username(self, username: str) -> str:
        """Hash username for privacy."""
        # Use a consistent salt for username hashing
        salt = self.config.security.secret_key[:16].encode()
        return hashlib.sha256(salt + username.encode()).hexdigest()[:8]
    
    def _get_client_ip(self) -> str:
        """Get client IP address."""
        try:
            from flask import request, g
            if hasattr(g, 'client_ip'):
                return g.client_ip
            return request.remote_addr or 'unknown'
        except:
            return 'unknown'
    
    def _get_user_agent(self) -> str:
        """Get user agent."""
        try:
            from flask import request
            return request.headers.get('User-Agent', 'unknown')
        except:
            return 'unknown'
    
    def rotate_logs(self) -> None:
        """Manually rotate all logs."""
        for handler in self.security_logger.handlers:
            if hasattr(handler, 'doRollover'):
                handler.doRollover()
        
        for handler in self.audit_logger.handlers:
            if hasattr(handler, 'doRollover'):
                handler.doRollover()
        
        for handler in self.access_logger.handlers:
            if hasattr(handler, 'doRollover'):
                handler.doRollover()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        # Count log files
        security_logs = list(self.security_log_dir.glob('*.log*'))
        audit_logs = list(self.audit_log_dir.glob('*.log*'))
        access_logs = list(self.log_dir.glob('access.log*'))
        
        # Calculate sizes
        security_size = sum(f.stat().st_size for f in security_logs if f.is_file())
        audit_size = sum(f.stat().st_size for f in audit_logs if f.is_file())
        access_size = sum(f.stat().st_size for f in access_logs if f.is_file())
        
        return {
            'security_logs': {
                'count': len(security_logs),
                'size_bytes': security_size,
                'size_mb': round(security_size / (1024 * 1024), 2)
            },
            'audit_logs': {
                'count': len(audit_logs),
                'size_bytes': audit_size,
                'size_mb': round(audit_size / (1024 * 1024), 2)
            },
            'access_logs': {
                'count': len(access_logs),
                'size_bytes': access_size,
                'size_mb': round(access_size / (1024 * 1024), 2)
            },
            'recent_events': len(self.security_events),
            'log_directory': str(self.log_dir)
        }
    
    class SensitiveDataFilter(logging.Filter):
        """Logging filter to remove sensitive data."""
        
        def __init__(self, secure_logger):
            self.secure_logger = secure_logger
        
        def filter(self, record):
            # Filter the message
            if hasattr(record, 'msg'):
                record.msg = self.secure_logger.filter_sensitive_data(str(record.msg))
            
            # Filter arguments
            if hasattr(record, 'args'):
                filtered_args = []
                for arg in record.args:
                    filtered_args.append(self.secure_logger.filter_sensitive_data(str(arg)))
                record.args = tuple(filtered_args)
            
            return True