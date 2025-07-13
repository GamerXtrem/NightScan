#!/usr/bin/env python3
"""
Security Tests for Sensitive Data Protection in NightScan

This test suite validates the sensitive data protection mechanisms
including sanitization, filtering, and secure logging.

Test Categories:
1. Sensitive Data Detection and Redaction
2. Logging Security and Filtering  
3. Configuration Security
4. Real-world Attack Scenarios
5. Performance and Stress Testing

Usage:
    # Run all security tests
    pytest tests/test_sensitive_data_security.py -v

    # Run specific test categories
    pytest tests/test_sensitive_data_security.py::TestSensitiveDataDetection -v
    pytest tests/test_sensitive_data_security.py::TestLoggingSecurity -v

    # Run with coverage
    pytest tests/test_sensitive_data_security.py --cov=sensitive_data_sanitizer --cov-report=html
"""

import pytest
import logging
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sensitive_data_sanitizer import (
    SensitiveDataSanitizer, SensitivePattern, SensitivityLevel, 
    RedactionLevel, get_sanitizer
)
from secure_logging_filters import (
    SecureLoggingFilter, SecureJSONFormatter, SecureLoggingManager,
    get_secure_logging_manager
)


class TestSensitiveDataDetection:
    """Test sensitive data detection and redaction capabilities."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.sanitizer = SensitiveDataSanitizer()
    
    def test_password_detection_and_redaction(self):
        """Test password detection in various formats."""
        test_cases = [
            ('password="secret123"', 'password="********"'),
            ('user password: mypassword', 'user password: ********'),
            ('PASSWORD=abc123', 'PASSWORD=abc123'),  # Different handling
            ('{"password": "test123"}', '{"password": "********"}'),
            ('password : "complex_pass_word"', 'password : "********"')
        ]
        
        for original, expected_pattern in test_cases:
            sanitized = self.sanitizer.sanitize(original)
            
            # Check that password was redacted
            assert 'secret123' not in sanitized
            assert 'mypassword' not in sanitized
            assert 'test123' not in sanitized
            assert 'complex_pass_word' not in sanitized
            
            # Check that structure is preserved
            if 'password' in original:
                assert 'password' in sanitized
    
    def test_token_detection_and_redaction(self):
        """Test various token formats."""
        test_cases = [
            'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature',
            'api_token="sk-1234567890abcdef"',
            'push_token: "APNs_token_1234567890abcdefghijklmnopqrstuvwxyz"',
            'jwt_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.signature'
        ]
        
        for test_string in test_cases:
            sanitized = self.sanitizer.sanitize(test_string)
            
            # Verify tokens are redacted
            assert 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9' not in sanitized
            assert 'sk-1234567890abcdef' not in sanitized
            assert 'APNs_token_1234567890abcdefghijklmnopqrstuvwxyz' not in sanitized
            
            # Verify structure is preserved
            assert ('Bearer' in test_string and 'Bearer' in sanitized) or 'Bearer' not in test_string
            assert ('api_token' in test_string and 'api_token' in sanitized) or 'api_token' not in test_string
    
    def test_database_connection_string_redaction(self):
        """Test database connection string credential redaction."""
        test_cases = [
            'postgresql://user:password@localhost:5432/dbname',
            'mysql://admin:secret123@db.example.com:3306/app_db',
            'mongodb://dbuser:dbpass@cluster.mongodb.net:27017/database',
            'redis://default:redispass@redis.example.com:6379'
        ]
        
        for connection_string in test_cases:
            sanitized = self.sanitizer.sanitize(connection_string)
            
            # Verify credentials are redacted
            assert 'password' not in sanitized
            assert 'secret123' not in sanitized
            assert 'dbpass' not in sanitized
            assert 'redispass' not in sanitized
            
            # Verify connection info is preserved
            assert 'localhost' in sanitized or 'localhost' not in connection_string
            assert 'mongodb.net' in sanitized or 'mongodb.net' not in connection_string
    
    def test_email_address_detection(self):
        """Test email address detection and redaction."""
        test_cases = [
            'User email: john.doe@example.com',
            'Contact support@nightscan.com for help',
            'Error from user test+user@company.co.uk'
        ]
        
        for test_string in test_cases:
            sanitized = self.sanitizer.sanitize(test_string)
            
            # Check email addresses are redacted
            assert 'john.doe@example.com' not in sanitized
            assert 'support@nightscan.com' not in sanitized
            assert 'test+user@company.co.uk' not in sanitized
            
            # Check structure hints remain
            assert '@' in sanitized or '@' not in test_string
    
    def test_custom_pattern_addition(self):
        """Test adding custom sensitive data patterns."""
        # Add custom pattern for internal IDs
        success = self.sanitizer.add_pattern(
            pattern=r'internal_id["\s]*[:=]["\s]*([A-Z0-9]{8,})',
            replacement='internal_id="***"',
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            description="Internal ID pattern"
        )
        
        assert success
        
        # Test custom pattern detection
        test_string = 'Processing internal_id="ABC12345678"'
        sanitized = self.sanitizer.sanitize(test_string)
        
        assert 'ABC12345678' not in sanitized
        assert 'internal_id' in sanitized
    
    def test_redaction_levels(self):
        """Test different redaction levels."""
        test_value = "sensitive_data_12345"
        
        # Test different redaction levels
        assert self.sanitizer._redact_value(test_value, RedactionLevel.FULL) == "********"
        
        partial = self.sanitizer._redact_value(test_value, RedactionLevel.PARTIAL)
        assert partial.startswith("se") and partial.endswith("45")
        assert "****" in partial
        
        hash_result = self.sanitizer._redact_value(test_value, RedactionLevel.HASH)
        assert hash_result.startswith("SHA256:")
        assert len(hash_result) == 23  # SHA256: + 16 chars
        
        truncate = self.sanitizer._redact_value(test_value, RedactionLevel.TRUNCATE)
        assert truncate == "sensitiv..." or len(test_value) <= 8
    
    def test_no_false_positives(self):
        """Test that normal log messages are not affected."""
        normal_messages = [
            "User login successful",
            "File uploaded: document.pdf",
            "Database connection established",
            "Processing completed in 1.5 seconds",
            "Cache hit rate: 95%"
        ]
        
        for message in normal_messages:
            sanitized = self.sanitizer.sanitize(message)
            assert sanitized == message  # Should be unchanged


class TestLoggingSecurity:
    """Test logging security filters and formatters."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.filter = SecureLoggingFilter()
        self.formatter = SecureJSONFormatter()
        
        # Create test logger
        self.logger = logging.getLogger('test_security_logger')
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        
        # Add filter and handler
        self.logger.addFilter(self.filter)
        
        # Create in-memory handler for testing
        from io import StringIO
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
    
    def test_logging_filter_sanitization(self):
        """Test that logging filter sanitizes messages."""
        # Log a message with sensitive data
        self.logger.info("User password is secret123")
        
        # Get logged output
        log_output = self.log_stream.getvalue()
        
        # Verify password was redacted
        assert 'secret123' not in log_output
        assert 'password' in log_output  # Structure preserved
    
    def test_json_formatter_structure(self):
        """Test JSON formatter maintains proper structure."""
        self.logger.info("Test message", extra={'custom_field': 'value'})
        
        log_output = self.log_stream.getvalue().strip()
        
        # Parse JSON to verify structure
        log_data = json.loads(log_output)
        
        required_fields = ['timestamp', 'level', 'logger', 'message']
        for field in required_fields:
            assert field in log_data
        
        assert log_data['level'] == 'INFO'
        assert log_data['message'] == 'Test message'
        assert log_data['custom_field'] == 'value'
    
    def test_exception_logging_sanitization(self):
        """Test that exceptions are sanitized in logs."""
        try:
            # Create exception with sensitive data
            user_password = "secret123"
            raise ValueError(f"Database error for user with password {user_password}")
        except Exception:
            self.logger.exception("Error occurred")
        
        log_output = self.log_stream.getvalue()
        
        # Verify password was redacted even in exception
        assert 'secret123' not in log_output
    
    def test_filter_performance(self):
        """Test filter performance under load."""
        start_time = time.time()
        
        # Log many messages
        for i in range(1000):
            self.logger.info(f"Message {i} with some data")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 messages in reasonable time (< 1 second)
        assert processing_time < 1.0
        
        # Verify statistics
        stats = self.filter.get_statistics()
        assert stats['records_processed'] >= 1000
    
    def test_secure_logging_manager(self):
        """Test secure logging manager functionality."""
        manager = get_secure_logging_manager()
        
        # Setup secure logging
        logger = manager.setup_secure_logging('test.manager')
        
        # Verify logger is configured
        assert len(logger.filters) > 0
        
        # Test statistics
        stats = manager.get_comprehensive_statistics()
        assert 'manager_stats' in stats
        assert 'sanitizer_stats' in stats


class TestConfigurationSecurity:
    """Test configuration and setup security."""
    
    def test_sanitizer_configuration_loading(self):
        """Test loading sanitizer configuration from file."""
        config_data = {
            'settings': {
                'max_history_size': 500,
                'default_redaction_level': 'full'
            },
            'custom_patterns': [
                {
                    'pattern': r'custom_secret["\s]*[:=]["\s]*([^"]+)',
                    'replacement': 'custom_secret="***"',
                    'sensitivity': 'secret',
                    'redaction_level': 'full',
                    'description': 'Custom test pattern'
                }
            ]
        }
        
        # Write config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            # Create sanitizer with config
            sanitizer = SensitiveDataSanitizer(config_file)
            
            # Test custom pattern works
            test_string = 'custom_secret="test123"'
            sanitized = sanitizer.sanitize(test_string)
            
            assert 'test123' not in sanitized
            assert 'custom_secret' in sanitized
            
        finally:
            # Cleanup
            Path(config_file).unlink()
    
    def test_pattern_management(self):
        """Test adding, removing, and managing patterns."""
        sanitizer = SensitiveDataSanitizer()
        
        initial_count = len(sanitizer.patterns)
        
        # Add pattern
        success = sanitizer.add_pattern(
            pattern=r'test_pattern_([0-9]+)',
            description='Test pattern for management'
        )
        assert success
        assert len(sanitizer.patterns) == initial_count + 1
        
        # Disable pattern
        success = sanitizer.disable_pattern('Test pattern for management')
        assert success
        
        # Remove pattern
        success = sanitizer.remove_pattern('Test pattern for management')
        assert success
        assert len(sanitizer.patterns) == initial_count
    
    def test_statistics_tracking(self):
        """Test statistics and metrics collection."""
        sanitizer = SensitiveDataSanitizer()
        
        # Clear stats
        sanitizer.reset_statistics()
        
        # Perform sanitizations
        test_strings = [
            'password="secret1"',
            'api_key="key123"',
            'normal message',
            'token="abc123def456"'
        ]
        
        for test_string in test_strings:
            sanitizer.sanitize(test_string)
        
        stats = sanitizer.get_statistics()
        
        assert stats['total_sanitizations'] == len(test_strings)
        assert stats['total_redactions'] > 0
        assert len(stats['patterns_matched']) > 0


class TestRealWorldScenarios:
    """Test real-world attack scenarios and edge cases."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.sanitizer = SensitiveDataSanitizer()
    
    def test_log_injection_attack(self):
        """Test protection against log injection attacks."""
        # Simulated log injection attempt
        injection_attempts = [
            'username=admin\npassword=secret123\n[FAKE] Authentication successful',
            'user\npassword="injected"\nINFO: Fake success message',
            'input\rpassword=override\rSUCCESS'
        ]
        
        for attempt in injection_attempts:
            sanitized = self.sanitizer.sanitize(attempt)
            
            # Verify sensitive parts are redacted
            assert 'secret123' not in sanitized
            assert 'injected' not in sanitized
            assert 'override' not in sanitized
    
    def test_base64_encoded_secrets(self):
        """Test detection of base64 encoded secrets."""
        # Add pattern for base64 encoded secrets
        self.sanitizer.add_pattern(
            pattern=r'(?i)secret["\s]*[:=]["\s]*([A-Za-z0-9+/=]{20,})',
            description='Base64 encoded secrets'
        )
        
        test_cases = [
            'secret="dGVzdF9zZWNyZXRfMTIz"',  # base64: test_secret_123
            'SECRET_KEY=YWJjZGVmZ2hpams='  # base64: abcdefghijk
        ]
        
        for test_case in test_cases:
            sanitized = self.sanitizer.sanitize(test_case)
            
            assert 'dGVzdF9zZWNyZXRfMTIz' not in sanitized
            assert 'YWJjZGVmZ2hpams=' not in sanitized
    
    def test_unicode_and_encoding_edge_cases(self):
        """Test handling of unicode and various encodings."""
        test_cases = [
            'password="pâsswörd123"',  # Unicode characters
            'token="tókèñ_wîth_áccénts"',
            'secret="🔒secret🔑"',  # Emoji
            'key="\u0070\u0061\u0073\u0073"'  # Unicode escapes
        ]
        
        for test_case in test_cases:
            sanitized = self.sanitizer.sanitize(test_case)
            
            # Should handle unicode without errors
            assert isinstance(sanitized, str)
            # Sensitive parts should be redacted
            assert 'pâsswörd123' not in sanitized
            assert 'tókèñ_wîth_áccénts' not in sanitized
    
    def test_very_long_messages(self):
        """Test handling of very long log messages."""
        # Create very long message with embedded secret
        long_message = "x" * 10000 + 'password="secret123"' + "y" * 10000
        
        sanitized = self.sanitizer.sanitize(long_message)
        
        # Should handle long messages without errors
        assert isinstance(sanitized, str)
        assert 'secret123' not in sanitized
        assert 'password=' in sanitized
    
    def test_nested_data_structures(self):
        """Test sanitization of nested data structures in logs."""
        nested_data = {
            'user': {
                'credentials': {
                    'password': 'secret123',
                    'api_key': 'key_abc123'
                }
            },
            'config': {
                'database_url': 'postgresql://user:pass@localhost/db'
            }
        }
        
        # Convert to string (as would happen in logging)
        log_message = f"Processing data: {json.dumps(nested_data)}"
        sanitized = self.sanitizer.sanitize(log_message)
        
        # Verify all sensitive data is redacted
        assert 'secret123' not in sanitized
        assert 'key_abc123' not in sanitized
        assert ':pass@' not in sanitized


class TestPerformanceAndStress:
    """Test performance and stress scenarios."""
    
    def test_sanitization_performance(self):
        """Test sanitization performance under load."""
        sanitizer = SensitiveDataSanitizer()
        
        test_messages = [
            "Normal log message",
            "User login successful",
            'password="secret123"',
            "API call completed",
            'token="abc123def456ghi789"',
            "Database query executed",
            "File upload completed",
            'api_key="key_123456789"',
            "Cache miss for key",
            "Processing completed"
        ]
        
        start_time = time.time()
        
        # Process many messages
        for _ in range(1000):
            for message in test_messages:
                sanitizer.sanitize(message)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should process 10,000 messages in reasonable time
        assert total_time < 2.0  # Less than 2 seconds
        
        # Check performance statistics
        stats = sanitizer.get_statistics()
        assert stats['total_sanitizations'] >= 10000
        
        if stats['performance_ms']:
            avg_time = stats['average_performance_ms']
            assert avg_time < 1.0  # Less than 1ms per sanitization
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively."""
        sanitizer = SensitiveDataSanitizer()
        
        # Clear history to start fresh
        sanitizer.clear_history()
        
        # Process many messages
        for i in range(5000):
            sanitizer.sanitize(f"Message {i} with password='secret{i}'")
        
        # Check history size is limited
        history = sanitizer.get_redaction_history()
        assert len(history) <= sanitizer.max_history_size
        
        # Statistics should be reasonable
        stats = sanitizer.get_statistics()
        assert stats['total_sanitizations'] == 5000
    
    def test_concurrent_sanitization(self):
        """Test thread safety under concurrent access."""
        import threading
        import queue
        
        sanitizer = SensitiveDataSanitizer()
        results = queue.Queue()
        errors = queue.Queue()
        
        def worker():
            """Worker function for concurrent testing."""
            try:
                for i in range(100):
                    message = f"Thread message {i} password='secret{i}'"
                    sanitized = sanitizer.sanitize(message)
                    results.put(sanitized)
            except Exception as e:
                errors.put(e)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert errors.empty()  # No errors should occur
        
        # Should have all results
        result_count = 0
        while not results.empty():
            sanitized = results.get()
            assert isinstance(sanitized, str)
            assert 'secret' not in sanitized  # All should be sanitized
            result_count += 1
        
        assert result_count == 1000  # 10 threads * 100 messages each


class TestIntegrationScenarios:
    """Test integration with existing NightScan components."""
    
    def test_flask_integration(self):
        """Test integration with Flask logging."""
        from unittest.mock import Mock
        
        # Mock Flask request context
        mock_request = Mock()
        mock_request.method = 'POST'
        mock_request.path = '/api/login'
        mock_request.remote_addr = '127.0.0.1'
        mock_request.headers = {'User-Agent': 'TestAgent'}
        
        mock_g = Mock()
        mock_g.request_id = 'test-request-123'
        mock_g.current_user = Mock()
        mock_g.current_user.id = 42
        
        with patch('secure_logging_filters.request', mock_request), \
             patch('secure_logging_filters.g', mock_g), \
             patch('secure_logging_filters.has_request_context', return_value=True):
            
            formatter = SecureJSONFormatter()
            
            # Create log record
            record = logging.LogRecord(
                name='test.logger',
                level=logging.INFO,
                pathname='test.py',
                lineno=100,
                msg='User login with password=%s',
                args=('secret123',),
                exc_info=None
            )
            
            # Format with context
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            # Verify context is included
            assert 'user_id' in log_data
            assert log_data['user_id'] == 42
            
            # Verify sensitive data is redacted
            assert 'secret123' not in formatted
    
    def test_circuit_breaker_integration(self):
        """Test integration with circuit breaker logging."""
        # This would test that circuit breaker logs don't expose sensitive data
        from unittest.mock import Mock
        
        # Mock circuit breaker event
        cb_event = {
            'circuit_name': 'database',
            'state': 'OPEN',
            'failure_count': 5,
            'connection_string': 'postgresql://user:password@localhost/db'
        }
        
        sanitizer = SensitiveDataSanitizer()
        
        # Sanitize circuit breaker log data
        log_message = f"Circuit breaker event: {json.dumps(cb_event)}"
        sanitized = sanitizer.sanitize(log_message)
        
        # Verify connection string credentials are redacted
        assert 'password' not in sanitized or ':***@' in sanitized


# Test fixtures and utilities
@pytest.fixture
def temp_config_file():
    """Provide a temporary configuration file."""
    config_data = {
        'settings': {
            'max_history_size': 100,
            'default_redaction_level': 'full'
        },
        'custom_patterns': []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_file = f.name
    
    yield config_file
    
    # Cleanup
    Path(config_file).unlink()


@pytest.fixture
def sample_sensitive_data():
    """Provide sample sensitive data for testing."""
    return [
        'password="mysecret123"',
        'api_key="sk-1234567890abcdef"',
        'token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"',
        'email="user@example.com"',
        'ssn="123-45-6789"',
        'postgresql://user:pass@localhost/db',
        'credit_card="4111111111111111"'
    ]


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, '-v'])