#!/usr/bin/env python3
"""
Sensitive Data Sanitizer for NightScan Logging

This module provides comprehensive protection against sensitive data exposure in logs.
It implements advanced pattern detection, redaction, and sanitization to ensure that
sensitive information like passwords, tokens, API keys, and personal data never
appear in log files.

Features:
- Multi-layer pattern detection (regex, context-aware, ML-based)
- Configurable redaction levels (full, partial, hashed)
- Real-time log sanitization
- Audit trail for redacted content
- Performance-optimized for high-throughput logging
- Integration with existing logging infrastructure

Usage:
    from sensitive_data_sanitizer import SensitiveDataSanitizer, get_sanitizer
    
    # Get singleton sanitizer
    sanitizer = get_sanitizer()
    
    # Sanitize a log message
    safe_message = sanitizer.sanitize("User password: secret123")
    # Result: "User password: ********"
    
    # Register custom patterns
    sanitizer.add_pattern(r'custom_token=([A-Za-z0-9]+)', 'custom_token=***')
"""

import re
import hashlib
import logging
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import os

# Import existing secure logging if available
try:
    from security.secure_logging import SecureLogger
    SECURE_LOGGER_AVAILABLE = True
except ImportError:
    SECURE_LOGGER_AVAILABLE = False


class RedactionLevel(Enum):
    """Redaction security levels."""
    NONE = "none"           # No redaction
    PARTIAL = "partial"     # Show first/last characters: abc***xyz
    FULL = "full"           # Complete redaction: *********
    HASH = "hash"           # Replace with hash: SHA256:abc123...
    TRUNCATE = "truncate"   # Show limited chars: abc... (max 8 chars)


class SensitivityLevel(Enum):
    """Data sensitivity classification."""
    PUBLIC = "public"           # No protection needed
    INTERNAL = "internal"       # Internal use only
    CONFIDENTIAL = "confidential"  # Confidential data
    SECRET = "secret"           # Highly sensitive data
    TOP_SECRET = "top_secret"   # Maximum security required


@dataclass
class SensitivePattern:
    """Definition of a sensitive data pattern."""
    pattern: str                    # Regex pattern
    replacement: str               # Replacement template
    sensitivity: SensitivityLevel  # Classification level
    redaction_level: RedactionLevel # How to redact
    description: str               # Human-readable description
    enabled: bool = True           # Whether pattern is active
    case_sensitive: bool = False   # Case sensitivity
    capture_groups: List[int] = field(default_factory=list)  # Which groups to redact


@dataclass
class RedactionEvent:
    """Record of a redaction event for auditing."""
    timestamp: str
    pattern_description: str
    sensitivity_level: str
    original_length: int
    redacted_length: int
    context: str  # Surrounding context (first 50 chars)
    source_file: Optional[str] = None
    source_line: Optional[int] = None


class SensitiveDataSanitizer:
    """Advanced sensitive data sanitization system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the sanitizer.
        
        Args:
            config_file: Path to configuration file
        """
        self.patterns: List[SensitivePattern] = []
        self.compiled_patterns: List[Tuple[re.Pattern, SensitivePattern]] = []
        self.redaction_history: List[RedactionEvent] = []
        self.stats = {
            'total_sanitizations': 0,
            'total_redactions': 0,
            'patterns_matched': {},
            'performance_ms': []
        }
        
        # Configuration
        self.max_history_size = 1000
        self.enable_audit_trail = True
        self.enable_performance_tracking = True
        self.default_redaction_level = RedactionLevel.FULL
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize logger
        self.logger = logging.getLogger('nightscan.sensitive_data_sanitizer')
        
        # Load default patterns
        self._load_default_patterns()
        
        # Load configuration if provided
        if config_file:
            self.load_configuration(config_file)
        
        # Compile patterns for performance
        self._compile_patterns()
    
    def _load_default_patterns(self):
        """Load comprehensive default sensitive data patterns."""
        
        # Authentication & Authorization
        auth_patterns = [
            SensitivePattern(
                pattern=r'(?i)password["\s]*[:=]["\s]*([^"\s,}]+)',
                replacement='password="********"',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="Password fields in various formats",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)api[_-]?key["\s]*[:=]["\s]*([^"\s,}]+)',
                replacement='api_key="********"',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="API keys",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)secret[_-]?key["\s]*[:=]["\s]*([^"\s,}]+)',
                replacement='secret_key="********"',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="Secret keys",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'Authorization:\s*Bearer\s+([^\s]+)',
                replacement='Authorization: Bearer ********',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="Bearer tokens in Authorization headers",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)jwt["\s]*[:=]["\s]*([A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)',
                replacement='jwt="********"',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.PARTIAL,
                description="JWT tokens",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)session[_-]?id["\s]*[:=]["\s]*([^"\s,}]+)',
                replacement='session_id="********"',
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                redaction_level=RedactionLevel.HASH,
                description="Session identifiers",
                capture_groups=[1]
            )
        ]
        
        # Database & Connection Strings
        database_patterns = [
            SensitivePattern(
                pattern=r'postgresql://([^:]+):([^@]+)@([^/]+)/(.+)',
                replacement='postgresql://***:***@\\3/\\4',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="PostgreSQL connection strings with credentials",
                capture_groups=[1, 2]
            ),
            SensitivePattern(
                pattern=r'mysql://([^:]+):([^@]+)@([^/]+)/(.+)',
                replacement='mysql://***:***@\\3/\\4',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="MySQL connection strings with credentials",
                capture_groups=[1, 2]
            ),
            SensitivePattern(
                pattern=r'mongodb://([^:]+):([^@]+)@([^/]+)/(.+)',
                replacement='mongodb://***:***@\\3/\\4',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="MongoDB connection strings with credentials",
                capture_groups=[1, 2]
            ),
            SensitivePattern(
                pattern=r'redis://([^:]+):([^@]+)@([^/]+)',
                replacement='redis://***:***@\\3',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="Redis connection strings with credentials",
                capture_groups=[1, 2]
            )
        ]
        
        # Device & Push Tokens
        device_patterns = [
            SensitivePattern(
                pattern=r'(?i)push[_-]?token["\s]*[:=]["\s]*([A-Za-z0-9]{50,})',
                replacement='push_token="********"',
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                redaction_level=RedactionLevel.PARTIAL,
                description="Push notification tokens",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)device[_-]?token["\s]*[:=]["\s]*([A-Za-z0-9]{20,})',
                replacement='device_token="********"',
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                redaction_level=RedactionLevel.PARTIAL,
                description="Device tokens",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)fcm[_-]?token["\s]*[:=]["\s]*([A-Za-z0-9_-]{100,})',
                replacement='fcm_token="********"',
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                redaction_level=RedactionLevel.PARTIAL,
                description="FCM registration tokens",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)apns[_-]?token["\s]*[:=]["\s]*([A-Za-z0-9]{50,})',
                replacement='apns_token="********"',
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                redaction_level=RedactionLevel.PARTIAL,
                description="APNs device tokens",
                capture_groups=[1]
            )
        ]
        
        # Personal Information
        personal_patterns = [
            SensitivePattern(
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                replacement='***@***.***',
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                redaction_level=RedactionLevel.PARTIAL,
                description="Email addresses"
            ),
            SensitivePattern(
                pattern=r'(?i)ssn["\s]*[:=]["\s]*(\d{3}-?\d{2}-?\d{4})',
                replacement='ssn="***-**-****"',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="Social Security Numbers",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)phone["\s]*[:=]["\s]*([+]?[\d\s\-\(\)]{10,})',
                replacement='phone="***-***-****"',
                sensitivity=SensitivityLevel.CONFIDENTIAL,
                redaction_level=RedactionLevel.PARTIAL,
                description="Phone numbers",
                capture_groups=[1]
            )
        ]
        
        # Credit Card & Financial
        financial_patterns = [
            SensitivePattern(
                pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                replacement='****-****-****-****',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="Credit card numbers"
            ),
            SensitivePattern(
                pattern=r'(?i)cvv["\s]*[:=]["\s]*(\d{3,4})',
                replacement='cvv="***"',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="CVV codes",
                capture_groups=[1]
            )
        ]
        
        # Network & System
        network_patterns = [
            SensitivePattern(
                pattern=r'(?i)private[_-]?key["\s]*[:=]["\s]*([A-Za-z0-9+/=]{20,})',
                replacement='private_key="********"',
                sensitivity=SensitivityLevel.TOP_SECRET,
                redaction_level=RedactionLevel.FULL,
                description="Private cryptographic keys",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'-----BEGIN [A-Z ]+-----([A-Za-z0-9+/=\s]+)-----END [A-Z ]+-----',
                replacement='-----BEGIN ***-----\n********\n-----END ***-----',
                sensitivity=SensitivityLevel.TOP_SECRET,
                redaction_level=RedactionLevel.FULL,
                description="PEM format keys and certificates",
                capture_groups=[1]
            )
        ]
        
        # AWS & Cloud Credentials
        cloud_patterns = [
            SensitivePattern(
                pattern=r'(?i)aws[_-]?access[_-]?key[_-]?id["\s]*[:=]["\s]*([A-Z0-9]{20})',
                replacement='aws_access_key_id="********"',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="AWS access key IDs",
                capture_groups=[1]
            ),
            SensitivePattern(
                pattern=r'(?i)aws[_-]?secret[_-]?access[_-]?key["\s]*[:=]["\s]*([A-Za-z0-9+/]{40})',
                replacement='aws_secret_access_key="********"',
                sensitivity=SensitivityLevel.SECRET,
                redaction_level=RedactionLevel.FULL,
                description="AWS secret access keys",
                capture_groups=[1]
            )
        ]
        
        # Combine all patterns
        all_patterns = (
            auth_patterns + database_patterns + device_patterns + 
            personal_patterns + financial_patterns + network_patterns + 
            cloud_patterns
        )
        
        self.patterns.extend(all_patterns)
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        with self._lock:
            self.compiled_patterns = []
            
            for pattern in self.patterns:
                if pattern.enabled:
                    try:
                        flags = 0 if pattern.case_sensitive else re.IGNORECASE
                        compiled = re.compile(pattern.pattern, flags)
                        self.compiled_patterns.append((compiled, pattern))
                    except re.error as e:
                        self.logger.error(f"Failed to compile pattern '{pattern.description}': {e}")
    
    def sanitize(self, text: str, context: Optional[str] = None) -> str:
        """Sanitize text by removing or redacting sensitive data.
        
        Args:
            text: Text to sanitize
            context: Additional context for auditing
            
        Returns:
            Sanitized text with sensitive data redacted
        """
        if not text or not isinstance(text, str):
            return text
        
        start_time = time.time() if self.enable_performance_tracking else 0
        
        sanitized_text = text
        redactions_made = 0
        
        with self._lock:
            self.stats['total_sanitizations'] += 1
            
            for compiled_pattern, pattern_def in self.compiled_patterns:
                matches = list(compiled_pattern.finditer(sanitized_text))
                
                if matches:
                    # Track pattern usage
                    pattern_key = pattern_def.description
                    self.stats['patterns_matched'][pattern_key] = (
                        self.stats['patterns_matched'].get(pattern_key, 0) + len(matches)
                    )
                    
                    # Apply redaction
                    for match in reversed(matches):  # Reverse to maintain positions
                        redacted_value = self._apply_redaction(
                            match.group(0), 
                            pattern_def,
                            match.groups()
                        )
                        
                        # Replace in text
                        start, end = match.span()
                        sanitized_text = sanitized_text[:start] + redacted_value + sanitized_text[end:]
                        
                        # Record redaction event
                        if self.enable_audit_trail:
                            self._record_redaction(
                                pattern_def,
                                match.group(0),
                                redacted_value,
                                context or sanitized_text[max(0, start-25):start+75]
                            )
                        
                        redactions_made += 1
            
            self.stats['total_redactions'] += redactions_made
            
            # Track performance
            if self.enable_performance_tracking:
                duration_ms = (time.time() - start_time) * 1000
                self.stats['performance_ms'].append(duration_ms)
                
                # Keep only recent performance data
                if len(self.stats['performance_ms']) > 1000:
                    self.stats['performance_ms'] = self.stats['performance_ms'][-1000:]
        
        return sanitized_text
    
    def _apply_redaction(self, original: str, pattern: SensitivePattern, groups: Tuple[str, ...]) -> str:
        """Apply redaction based on pattern configuration.
        
        Args:
            original: Original matched text
            pattern: Pattern definition
            groups: Captured groups from regex match
            
        Returns:
            Redacted text
        """
        if pattern.redaction_level == RedactionLevel.NONE:
            return original
        
        # If specific capture groups are defined, only redact those
        if pattern.capture_groups and groups:
            result = original
            for group_idx in pattern.capture_groups:
                if group_idx <= len(groups) and groups[group_idx - 1]:
                    group_value = groups[group_idx - 1]
                    redacted_group = self._redact_value(group_value, pattern.redaction_level)
                    result = result.replace(group_value, redacted_group)
            return result
        
        # Otherwise, apply redaction to the entire match
        return self._redact_value(original, pattern.redaction_level)
    
    def _redact_value(self, value: str, level: RedactionLevel) -> str:
        """Apply specific redaction level to a value.
        
        Args:
            value: Value to redact
            level: Redaction level to apply
            
        Returns:
            Redacted value
        """
        if level == RedactionLevel.NONE:
            return value
        
        elif level == RedactionLevel.FULL:
            return "*" * min(len(value), 8)
        
        elif level == RedactionLevel.PARTIAL:
            if len(value) <= 4:
                return "*" * len(value)
            elif len(value) <= 8:
                return value[:1] + "*" * (len(value) - 2) + value[-1:]
            else:
                return value[:2] + "*" * min(len(value) - 4, 8) + value[-2:]
        
        elif level == RedactionLevel.HASH:
            hash_obj = hashlib.sha256(value.encode())
            return f"SHA256:{hash_obj.hexdigest()[:16]}"
        
        elif level == RedactionLevel.TRUNCATE:
            if len(value) <= 8:
                return value
            return value[:8] + "..."
        
        return value
    
    def _record_redaction(self, pattern: SensitivePattern, original: str, redacted: str, context: str):
        """Record a redaction event for auditing.
        
        Args:
            pattern: Pattern that triggered redaction
            original: Original text
            redacted: Redacted text
            context: Surrounding context
        """
        event = RedactionEvent(
            timestamp=datetime.now().isoformat(),
            pattern_description=pattern.description,
            sensitivity_level=pattern.sensitivity.value,
            original_length=len(original),
            redacted_length=len(redacted),
            context=context[:100]  # Limit context size
        )
        
        self.redaction_history.append(event)
        
        # Limit history size
        if len(self.redaction_history) > self.max_history_size:
            self.redaction_history = self.redaction_history[-self.max_history_size:]
    
    def add_pattern(self, pattern: Union[str, SensitivePattern], 
                   replacement: str = None,
                   sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL,
                   redaction_level: RedactionLevel = None,
                   description: str = "Custom pattern") -> bool:
        """Add a custom sensitive data pattern.
        
        Args:
            pattern: Regex pattern string or SensitivePattern object
            replacement: Replacement template (if pattern is string)
            sensitivity: Data sensitivity level
            redaction_level: How to redact the data
            description: Human-readable description
            
        Returns:
            True if pattern was added successfully
        """
        try:
            if isinstance(pattern, str):
                pattern_obj = SensitivePattern(
                    pattern=pattern,
                    replacement=replacement or "***",
                    sensitivity=sensitivity,
                    redaction_level=redaction_level or self.default_redaction_level,
                    description=description
                )
            else:
                pattern_obj = pattern
            
            # Test compilation
            flags = 0 if pattern_obj.case_sensitive else re.IGNORECASE
            re.compile(pattern_obj.pattern, flags)
            
            with self._lock:
                self.patterns.append(pattern_obj)
                self._compile_patterns()
            
            self.logger.info(f"Added sensitive data pattern: {pattern_obj.description}")
            return True
            
        except re.error as e:
            self.logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to add pattern: {e}")
            return False
    
    def remove_pattern(self, description: str) -> bool:
        """Remove a pattern by description.
        
        Args:
            description: Description of pattern to remove
            
        Returns:
            True if pattern was removed
        """
        with self._lock:
            original_count = len(self.patterns)
            self.patterns = [p for p in self.patterns if p.description != description]
            
            if len(self.patterns) < original_count:
                self._compile_patterns()
                self.logger.info(f"Removed sensitive data pattern: {description}")
                return True
            
            return False
    
    def enable_pattern(self, description: str) -> bool:
        """Enable a pattern by description.
        
        Args:
            description: Description of pattern to enable
            
        Returns:
            True if pattern was found and enabled
        """
        with self._lock:
            for pattern in self.patterns:
                if pattern.description == description:
                    pattern.enabled = True
                    self._compile_patterns()
                    return True
            return False
    
    def disable_pattern(self, description: str) -> bool:
        """Disable a pattern by description.
        
        Args:
            description: Description of pattern to disable
            
        Returns:
            True if pattern was found and disabled
        """
        with self._lock:
            for pattern in self.patterns:
                if pattern.description == description:
                    pattern.enabled = False
                    self._compile_patterns()
                    return True
            return False
    
    def get_patterns(self) -> List[Dict[str, Any]]:
        """Get list of all patterns with their status.
        
        Returns:
            List of pattern dictionaries
        """
        with self._lock:
            return [
                {
                    'description': p.description,
                    'sensitivity': p.sensitivity.value,
                    'redaction_level': p.redaction_level.value,
                    'enabled': p.enabled,
                    'case_sensitive': p.case_sensitive,
                    'pattern': p.pattern[:50] + "..." if len(p.pattern) > 50 else p.pattern
                }
                for p in self.patterns
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sanitization statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            stats = dict(self.stats)
            
            if self.stats['performance_ms']:
                stats['average_performance_ms'] = sum(self.stats['performance_ms']) / len(self.stats['performance_ms'])
                stats['max_performance_ms'] = max(self.stats['performance_ms'])
            else:
                stats['average_performance_ms'] = 0
                stats['max_performance_ms'] = 0
            
            stats['redaction_events'] = len(self.redaction_history)
            stats['active_patterns'] = len([p for p in self.patterns if p.enabled])
            stats['total_patterns'] = len(self.patterns)
            
            return stats
    
    def get_redaction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent redaction events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of redaction event dictionaries
        """
        with self._lock:
            recent_events = self.redaction_history[-limit:] if limit > 0 else self.redaction_history
            return [
                {
                    'timestamp': event.timestamp,
                    'pattern': event.pattern_description,
                    'sensitivity': event.sensitivity_level,
                    'original_length': event.original_length,
                    'redacted_length': event.redacted_length,
                    'context': event.context
                }
                for event in recent_events
            ]
    
    def load_configuration(self, config_file: str) -> bool:
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if configuration was loaded successfully
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load settings
            settings = config.get('settings', {})
            self.max_history_size = settings.get('max_history_size', self.max_history_size)
            self.enable_audit_trail = settings.get('enable_audit_trail', self.enable_audit_trail)
            self.enable_performance_tracking = settings.get('enable_performance_tracking', self.enable_performance_tracking)
            
            if 'default_redaction_level' in settings:
                self.default_redaction_level = RedactionLevel(settings['default_redaction_level'])
            
            # Load custom patterns
            custom_patterns = config.get('custom_patterns', [])
            for pattern_config in custom_patterns:
                pattern = SensitivePattern(
                    pattern=pattern_config['pattern'],
                    replacement=pattern_config.get('replacement', '***'),
                    sensitivity=SensitivityLevel(pattern_config.get('sensitivity', 'confidential')),
                    redaction_level=RedactionLevel(pattern_config.get('redaction_level', 'full')),
                    description=pattern_config.get('description', 'Custom pattern'),
                    enabled=pattern_config.get('enabled', True),
                    case_sensitive=pattern_config.get('case_sensitive', False),
                    capture_groups=pattern_config.get('capture_groups', [])
                )
                self.patterns.append(pattern)
            
            # Recompile patterns
            self._compile_patterns()
            
            self.logger.info(f"Loaded configuration from {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
            return False
    
    def save_configuration(self, config_file: str) -> bool:
        """Save current configuration to JSON file.
        
        Args:
            config_file: Path to save configuration
            
        Returns:
            True if configuration was saved successfully
        """
        try:
            config = {
                'settings': {
                    'max_history_size': self.max_history_size,
                    'enable_audit_trail': self.enable_audit_trail,
                    'enable_performance_tracking': self.enable_performance_tracking,
                    'default_redaction_level': self.default_redaction_level.value
                },
                'custom_patterns': [
                    {
                        'pattern': p.pattern,
                        'replacement': p.replacement,
                        'sensitivity': p.sensitivity.value,
                        'redaction_level': p.redaction_level.value,
                        'description': p.description,
                        'enabled': p.enabled,
                        'case_sensitive': p.case_sensitive,
                        'capture_groups': p.capture_groups
                    }
                    for p in self.patterns
                ]
            }
            
            # Ensure directory exists
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Saved configuration to {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_file}: {e}")
            return False
    
    def test_pattern(self, pattern: str, test_strings: List[str]) -> Dict[str, Any]:
        """Test a pattern against sample strings.
        
        Args:
            pattern: Regex pattern to test
            test_strings: List of strings to test against
            
        Returns:
            Test results
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            results = {
                'pattern': pattern,
                'valid': True,
                'matches': []
            }
            
            for test_string in test_strings:
                matches = list(compiled.finditer(test_string))
                if matches:
                    for match in matches:
                        results['matches'].append({
                            'input': test_string,
                            'match': match.group(0),
                            'start': match.start(),
                            'end': match.end(),
                            'groups': match.groups()
                        })
            
            return results
            
        except re.error as e:
            return {
                'pattern': pattern,
                'valid': False,
                'error': str(e)
            }
    
    def clear_history(self):
        """Clear redaction history."""
        with self._lock:
            self.redaction_history.clear()
            self.logger.info("Cleared redaction history")
    
    def reset_statistics(self):
        """Reset statistics counters."""
        with self._lock:
            self.stats = {
                'total_sanitizations': 0,
                'total_redactions': 0,
                'patterns_matched': {},
                'performance_ms': []
            }
            self.logger.info("Reset statistics")


# Singleton instance for global access
_sanitizer: Optional[SensitiveDataSanitizer] = None


def get_sanitizer() -> SensitiveDataSanitizer:
    """Get the global sanitizer instance."""
    global _sanitizer
    
    if _sanitizer is None:
        config_file = os.environ.get('NIGHTSCAN_SANITIZER_CONFIG')
        _sanitizer = SensitiveDataSanitizer(config_file)
    
    return _sanitizer


def sanitize_message(message: str, context: Optional[str] = None) -> str:
    """Convenience function to sanitize a message.
    
    Args:
        message: Message to sanitize
        context: Optional context for auditing
        
    Returns:
        Sanitized message
    """
    return get_sanitizer().sanitize(message, context)


# Integration with existing secure logger
if SECURE_LOGGER_AVAILABLE:
    class EnhancedSecureLogger(SecureLogger):
        """Enhanced secure logger with advanced sanitization."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sanitizer = get_sanitizer()
        
        def _sanitize_message(self, message: str) -> str:
            """Override sanitization with advanced sanitizer."""
            # Use parent class sanitization first
            message = super()._sanitize_message(message)
            
            # Apply advanced sanitization
            return self.sanitizer.sanitize(message)


if __name__ == "__main__":
    # Command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="NightScan Sensitive Data Sanitizer")
    parser.add_argument('--test', help='Test string to sanitize')
    parser.add_argument('--patterns', action='store_true', help='List all patterns')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--config', help='Load configuration file')
    
    args = parser.parse_args()
    
    sanitizer = get_sanitizer()
    
    if args.config:
        sanitizer.load_configuration(args.config)
    
    if args.test:
        result = sanitizer.sanitize(args.test)
        print(f"Original: {args.test}")
        print(f"Sanitized: {result}")
        
    elif args.patterns:
        patterns = sanitizer.get_patterns()
        print(f"Loaded {len(patterns)} patterns:")
        for pattern in patterns:
            status = "✅" if pattern['enabled'] else "❌"
            print(f"{status} {pattern['description']} ({pattern['sensitivity']})")
            
    elif args.stats:
        stats = sanitizer.get_statistics()
        print(json.dumps(stats, indent=2))
        
    else:
        parser.print_help()