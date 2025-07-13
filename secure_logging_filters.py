#!/usr/bin/env python3
"""
Secure Logging Filters for NightScan

This module provides logging filters and formatters that automatically sanitize
sensitive data in log messages. It integrates with the existing logging
infrastructure to provide transparent protection against data leaks.

Features:
- Automatic sensitive data detection and redaction
- Integration with existing logging handlers
- Performance-optimized for high-throughput applications
- Configurable sensitivity levels
- Real-time monitoring and alerting

Usage:
    from secure_logging_filters import SecureLoggingFilter, SecureFormatter
    
    # Add to existing logger
    logger = logging.getLogger('myapp')
    secure_filter = SecureLoggingFilter()
    logger.addFilter(secure_filter)
    
    # Use secure formatter
    handler = logging.StreamHandler()
    handler.setFormatter(SecureFormatter())
    logger.addHandler(handler)
"""

import logging
import time
import threading
import json
import os
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

from sensitive_data_sanitizer import get_sanitizer, SensitiveDataSanitizer, RedactionLevel


class SecureLoggingFilter(logging.Filter):
    """Logging filter that sanitizes sensitive data in log records."""
    
    def __init__(self, 
                 name: str = "",
                 sanitizer: Optional[SensitiveDataSanitizer] = None,
                 enable_performance_tracking: bool = True):
        """Initialize the secure logging filter.
        
        Args:
            name: Filter name
            sanitizer: Custom sanitizer instance (uses global if None)
            enable_performance_tracking: Track sanitization performance
        """
        super().__init__(name)
        self.sanitizer = sanitizer or get_sanitizer()
        self.enable_performance_tracking = enable_performance_tracking
        
        # Performance tracking
        self.filter_stats = {
            'records_processed': 0,
            'records_sanitized': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logger for this filter
        self.logger = logging.getLogger('nightscan.secure_logging_filter')
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and sanitize a log record.
        
        Args:
            record: Log record to process
            
        Returns:
            True to allow the record through (always True after sanitization)
        """
        start_time = time.time() if self.enable_performance_tracking else 0
        
        try:
            with self._lock:
                self.filter_stats['records_processed'] += 1
                
                original_message = record.getMessage()
                sanitized = False
                
                # Sanitize the main message
                sanitized_message = self.sanitizer.sanitize(
                    original_message,
                    context=f"{record.name}:{record.funcName}:{record.lineno}"
                )
                
                if sanitized_message != original_message:
                    # Update the record's message
                    record.msg = sanitized_message
                    record.args = ()  # Clear args since we've formatted the message
                    sanitized = True
                
                # Sanitize extra fields that might contain sensitive data
                if hasattr(record, '__dict__'):
                    for attr_name, attr_value in record.__dict__.items():
                        if attr_name.startswith('custom_') and isinstance(attr_value, str):
                            sanitized_value = self.sanitizer.sanitize(attr_value)
                            if sanitized_value != attr_value:
                                setattr(record, attr_name, sanitized_value)
                                sanitized = True
                
                # Sanitize exception information
                if record.exc_info and record.exc_text:
                    sanitized_exc_text = self.sanitizer.sanitize(record.exc_text)
                    if sanitized_exc_text != record.exc_text:
                        record.exc_text = sanitized_exc_text
                        sanitized = True
                
                if sanitized:
                    self.filter_stats['records_sanitized'] += 1
                
                # Track performance
                if self.enable_performance_tracking:
                    processing_time = time.time() - start_time
                    self.filter_stats['total_processing_time'] += processing_time
                    self.filter_stats['avg_processing_time'] = (
                        self.filter_stats['total_processing_time'] / 
                        self.filter_stats['records_processed']
                    )
                
                return True
                
        except Exception as e:
            # Never block logging due to filter errors
            self.logger.error(f"Error in secure logging filter: {e}")
            return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics.
        
        Returns:
            Dictionary with filter performance statistics
        """
        with self._lock:
            return dict(self.filter_stats)
    
    def reset_statistics(self):
        """Reset filter statistics."""
        with self._lock:
            self.filter_stats = {
                'records_processed': 0,
                'records_sanitized': 0,
                'total_processing_time': 0.0,
                'avg_processing_time': 0.0
            }


class SecureFormatter(logging.Formatter):
    """Enhanced formatter with additional sanitization capabilities."""
    
    def __init__(self,
                 fmt: Optional[str] = None,
                 datefmt: Optional[str] = None,
                 style: str = '%',
                 sanitizer: Optional[SensitiveDataSanitizer] = None,
                 sanitize_output: bool = True):
        """Initialize the secure formatter.
        
        Args:
            fmt: Format string
            datefmt: Date format string
            style: Format style ('%', '{', or '$')
            sanitizer: Custom sanitizer instance
            sanitize_output: Whether to sanitize the final formatted output
        """
        super().__init__(fmt, datefmt, style)
        self.sanitizer = sanitizer or get_sanitizer()
        self.sanitize_output = sanitize_output
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with sanitization.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted and sanitized log message
        """
        # Format using parent formatter
        formatted = super().format(record)
        
        # Apply final sanitization if enabled
        if self.sanitize_output:
            formatted = self.sanitizer.sanitize(formatted)
        
        return formatted


class SecureJSONFormatter(logging.Formatter):
    """JSON formatter with sensitive data sanitization."""
    
    def __init__(self,
                 sanitizer: Optional[SensitiveDataSanitizer] = None,
                 include_extra_fields: bool = True,
                 sanitize_keys: List[str] = None):
        """Initialize the secure JSON formatter.
        
        Args:
            sanitizer: Custom sanitizer instance
            include_extra_fields: Include custom fields in JSON output
            sanitize_keys: Additional keys to sanitize
        """
        super().__init__()
        self.sanitizer = sanitizer or get_sanitizer()
        self.include_extra_fields = include_extra_fields
        self.sanitize_keys = sanitize_keys or [
            'message', 'msg', 'error', 'exception', 'traceback',
            'request_data', 'response_data', 'user_input'
        ]
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as sanitized JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log entry with sanitized data
        """
        # Create base log entry
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread_id': record.thread,
            'process_id': record.process
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields if enabled
        if self.include_extra_fields:
            for key, value in record.__dict__.items():
                if key.startswith('custom_') or key in ['duration', 'status_code', 'file_size']:
                    log_entry[key] = value
        
        # Sanitize sensitive fields
        for key in self.sanitize_keys:
            if key in log_entry and isinstance(log_entry[key], str):
                log_entry[key] = self.sanitizer.sanitize(log_entry[key])
        
        # Sanitize nested exception data
        if 'exception' in log_entry:
            for exc_key in ['message', 'traceback']:
                if exc_key in log_entry['exception'] and isinstance(log_entry['exception'][exc_key], str):
                    log_entry['exception'][exc_key] = self.sanitizer.sanitize(log_entry['exception'][exc_key])
        
        # Sanitize custom fields
        for key, value in log_entry.items():
            if key.startswith('custom_') and isinstance(value, str):
                log_entry[key] = self.sanitizer.sanitize(value)
        
        return json.dumps(log_entry, default=str)


class SensitiveDataAlert:
    """Alert system for sensitive data detection in logs."""
    
    def __init__(self,
                 alert_threshold: int = 10,
                 time_window: int = 300,  # 5 minutes
                 alert_callback: Optional[callable] = None):
        """Initialize the alert system.
        
        Args:
            alert_threshold: Number of detections before alerting
            time_window: Time window in seconds for counting detections
            alert_callback: Function to call when threshold is exceeded
        """
        self.alert_threshold = alert_threshold
        self.time_window = time_window
        self.alert_callback = alert_callback
        
        # Detection tracking
        self.detections = []
        self.alerts_sent = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger('nightscan.sensitive_data_alert')
    
    def record_detection(self, pattern_description: str, context: str = ""):
        """Record a sensitive data detection.
        
        Args:
            pattern_description: Description of the pattern that matched
            context: Context where the detection occurred
        """
        current_time = time.time()
        
        with self._lock:
            # Add new detection
            detection = {
                'timestamp': current_time,
                'pattern': pattern_description,
                'context': context
            }
            self.detections.append(detection)
            
            # Clean old detections outside time window
            cutoff_time = current_time - self.time_window
            self.detections = [d for d in self.detections if d['timestamp'] > cutoff_time]
            
            # Check if alert threshold is exceeded
            if len(self.detections) >= self.alert_threshold:
                self._trigger_alert()
    
    def _trigger_alert(self):
        """Trigger an alert for excessive sensitive data detections."""
        current_time = time.time()
        
        # Check if we've already alerted recently (avoid spam)
        recent_alert_time = current_time - 300  # 5 minutes
        if self.alerts_sent and self.alerts_sent[-1] > recent_alert_time:
            return
        
        self.alerts_sent.append(current_time)
        
        # Prepare alert data
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'threshold_exceeded': len(self.detections),
            'threshold_limit': self.alert_threshold,
            'time_window_minutes': self.time_window / 60,
            'recent_detections': [
                {
                    'pattern': d['pattern'],
                    'context': d['context'][:100],  # Truncate context
                    'time_ago_seconds': current_time - d['timestamp']
                }
                for d in self.detections[-10:]  # Last 10 detections
            ]
        }
        
        # Log the alert
        self.logger.warning(
            f"Sensitive data alert: {len(self.detections)} detections in {self.time_window/60:.1f} minutes",
            extra={'alert_data': alert_data}
        )
        
        # Call custom alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status.
        
        Returns:
            Alert status information
        """
        current_time = time.time()
        
        with self._lock:
            # Clean old detections
            cutoff_time = current_time - self.time_window
            self.detections = [d for d in self.detections if d['timestamp'] > cutoff_time]
            
            return {
                'current_detections': len(self.detections),
                'alert_threshold': self.alert_threshold,
                'time_window_minutes': self.time_window / 60,
                'alerts_sent_count': len(self.alerts_sent),
                'last_alert_time': self.alerts_sent[-1] if self.alerts_sent else None,
                'recent_patterns': list(set(d['pattern'] for d in self.detections[-10:]))
            }


class SecureLoggingManager:
    """Manager for secure logging configuration and monitoring."""
    
    def __init__(self):
        """Initialize the secure logging manager."""
        self.filters = {}
        self.formatters = {}
        self.alert_system = SensitiveDataAlert()
        self.sanitizer = get_sanitizer()
        
        # Statistics
        self.stats = {
            'start_time': time.time(),
            'total_records_processed': 0,
            'total_records_sanitized': 0,
            'filters_active': 0,
            'formatters_active': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger('nightscan.secure_logging_manager')
    
    def setup_secure_logging(self, 
                           logger_name: str = None,
                           use_json_format: bool = True,
                           enable_alerts: bool = True) -> logging.Logger:
        """Setup secure logging for a logger.
        
        Args:
            logger_name: Name of logger to secure (None for root logger)
            use_json_format: Use JSON formatter
            enable_alerts: Enable sensitive data alerts
            
        Returns:
            Configured logger instance
        """
        # Get or create logger
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()
        
        # Create and add secure filter
        filter_name = f"secure_filter_{logger_name or 'root'}"
        secure_filter = SecureLoggingFilter(name=filter_name)
        logger.addFilter(secure_filter)
        
        with self._lock:
            self.filters[filter_name] = secure_filter
            self.stats['filters_active'] += 1
        
        # Setup formatters for existing handlers
        for handler in logger.handlers:
            formatter_name = f"secure_formatter_{id(handler)}"
            
            if use_json_format:
                secure_formatter = SecureJSONFormatter()
            else:
                secure_formatter = SecureFormatter()
            
            handler.setFormatter(secure_formatter)
            
            with self._lock:
                self.formatters[formatter_name] = secure_formatter
                self.stats['formatters_active'] += 1
        
        self.logger.info(f"Secure logging configured for: {logger_name or 'root logger'}")
        return logger
    
    def add_custom_pattern(self, pattern: str, description: str, sensitivity: str = "confidential") -> bool:
        """Add a custom sensitive data pattern.
        
        Args:
            pattern: Regex pattern
            description: Pattern description
            sensitivity: Sensitivity level
            
        Returns:
            True if pattern was added successfully
        """
        from sensitive_data_sanitizer import SensitivityLevel, RedactionLevel
        
        try:
            sensitivity_level = SensitivityLevel(sensitivity.lower())
            success = self.sanitizer.add_pattern(
                pattern=pattern,
                description=description,
                sensitivity=sensitivity_level,
                redaction_level=RedactionLevel.FULL
            )
            
            if success:
                self.logger.info(f"Added custom sensitive data pattern: {description}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to add custom pattern: {e}")
            return False
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components.
        
        Returns:
            Complete statistics dictionary
        """
        with self._lock:
            # Aggregate filter statistics
            filter_stats = {}
            total_processed = 0
            total_sanitized = 0
            
            for name, filter_obj in self.filters.items():
                stats = filter_obj.get_statistics()
                filter_stats[name] = stats
                total_processed += stats.get('records_processed', 0)
                total_sanitized += stats.get('records_sanitized', 0)
            
            # Update global stats
            self.stats['total_records_processed'] = total_processed
            self.stats['total_records_sanitized'] = total_sanitized
            
            return {
                'manager_stats': dict(self.stats),
                'sanitizer_stats': self.sanitizer.get_statistics(),
                'filter_stats': filter_stats,
                'alert_status': self.alert_system.get_alert_status(),
                'uptime_hours': (time.time() - self.stats['start_time']) / 3600
            }
    
    def export_configuration(self, config_file: str) -> bool:
        """Export current secure logging configuration.
        
        Args:
            config_file: Path to save configuration
            
        Returns:
            True if export was successful
        """
        try:
            config = {
                'timestamp': datetime.now().isoformat(),
                'manager_settings': {
                    'filters_active': len(self.filters),
                    'formatters_active': len(self.formatters)
                },
                'sanitizer_patterns': self.sanitizer.get_patterns(),
                'alert_settings': {
                    'threshold': self.alert_system.alert_threshold,
                    'time_window': self.alert_system.time_window
                }
            }
            
            # Ensure directory exists
            Path(config_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Exported secure logging configuration to: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on secure logging system.
        
        Returns:
            Health check results
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {
                'sanitizer': 'healthy',
                'filters': 'healthy',
                'formatters': 'healthy',
                'alerts': 'healthy'
            },
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check sanitizer
            sanitizer_stats = self.sanitizer.get_statistics()
            if sanitizer_stats.get('total_sanitizations', 0) == 0:
                health['components']['sanitizer'] = 'inactive'
                health['recommendations'].append("Sanitizer hasn't processed any data yet")
            
            # Check filters
            if not self.filters:
                health['components']['filters'] = 'inactive'
                health['issues'].append("No secure filters are active")
                health['overall_status'] = 'degraded'
            
            # Check for excessive sanitization
            sanitization_rate = (
                sanitizer_stats.get('total_redactions', 0) / 
                max(sanitizer_stats.get('total_sanitizations', 1), 1)
            )
            
            if sanitization_rate > 0.1:  # More than 10% of messages sanitized
                health['recommendations'].append(
                    f"High sanitization rate ({sanitization_rate:.1%}) - review logging practices"
                )
            
            # Check alert status
            alert_status = self.alert_system.get_alert_status()
            if alert_status['current_detections'] >= alert_status['alert_threshold'] * 0.8:
                health['components']['alerts'] = 'warning'
                health['recommendations'].append("Approaching sensitive data detection threshold")
            
            return health
            
        except Exception as e:
            health['overall_status'] = 'unhealthy'
            health['issues'].append(f"Health check failed: {str(e)}")
            return health


# Global secure logging manager instance
_secure_logging_manager: Optional[SecureLoggingManager] = None


def get_secure_logging_manager() -> SecureLoggingManager:
    """Get the global secure logging manager instance."""
    global _secure_logging_manager
    
    if _secure_logging_manager is None:
        _secure_logging_manager = SecureLoggingManager()
    
    return _secure_logging_manager


def setup_secure_logging(logger_name: str = None, **kwargs) -> logging.Logger:
    """Convenience function to setup secure logging."""
    return get_secure_logging_manager().setup_secure_logging(logger_name, **kwargs)


if __name__ == "__main__":
    # Command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="NightScan Secure Logging Filters")
    parser.add_argument('--test', help='Test message to process')
    parser.add_argument('--setup', help='Setup secure logging for logger name')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--health', action='store_true', help='Perform health check')
    
    args = parser.parse_args()
    
    manager = get_secure_logging_manager()
    
    if args.test:
        # Test sanitization
        logger = setup_secure_logging('test_logger')
        logger.info(f"Test message: {args.test}")
        
        # Show results
        stats = manager.get_comprehensive_statistics()
        print(f"Processed: {stats['manager_stats']['total_records_processed']}")
        print(f"Sanitized: {stats['manager_stats']['total_records_sanitized']}")
        
    elif args.setup:
        logger = setup_secure_logging(args.setup)
        print(f"Secure logging configured for: {args.setup}")
        
    elif args.stats:
        stats = manager.get_comprehensive_statistics()
        print(json.dumps(stats, indent=2, default=str))
        
    elif args.health:
        health = manager.health_check()
        print(json.dumps(health, indent=2))
        
    else:
        parser.print_help()