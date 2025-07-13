"""
Comprehensive assertion helpers for NightScan test suite.

This module provides reusable assertion utilities to improve test quality
by moving beyond shallow status code checking to deep validation of:
- Response content and structure
- Business logic correctness
- State changes and side effects
- Error handling and messages
"""

import json
import re
from typing import Dict, Any, List, Optional, Union
from flask import Response
from werkzeug.test import TestResponse


class ResponseAssertions:
    """Comprehensive response validation helpers."""
    
    @staticmethod
    def assert_success_response(response: TestResponse, expected_data: Optional[Dict] = None):
        """
        Assert successful response with optional data validation.
        
        Args:
            response: Flask test response
            expected_data: Optional dict to validate against response data
        """
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.get_json()
        assert data is not None, "Response should contain JSON data"
        
        if expected_data:
            for key, value in expected_data.items():
                assert key in data, f"Missing key '{key}' in response"
                if value is not None:
                    assert data[key] == value, f"Expected {key}={value}, got {data[key]}"
    
    @staticmethod
    def assert_error_response(response: TestResponse, expected_status: int, 
                            error_keywords: Optional[List[str]] = None):
        """
        Assert error response with status and message validation.
        
        Args:
            response: Flask test response
            expected_status: Expected HTTP status code
            error_keywords: Keywords that should appear in error message
        """
        assert response.status_code == expected_status, \
            f"Expected {expected_status}, got {response.status_code}"
        
        data = response.get_json()
        assert data is not None, "Error response should contain JSON"
        assert 'error' in data, "Error response should contain 'error' field"
        
        if error_keywords:
            error_msg = data['error'].lower()
            for keyword in error_keywords:
                assert keyword.lower() in error_msg, \
                    f"Expected '{keyword}' in error message: {data['error']}"
    
    @staticmethod
    def assert_validation_error(response: TestResponse, field_name: str):
        """
        Assert validation error for specific field.
        
        Args:
            response: Flask test response
            field_name: Name of field that should have validation error
        """
        ResponseAssertions.assert_error_response(response, 400, ['validation', field_name])
        
        data = response.get_json()
        # Check for field-specific error details
        error_msg = data['error'].lower()
        assert field_name.lower() in error_msg, \
            f"Field '{field_name}' should be mentioned in validation error"
    
    @staticmethod
    def assert_authentication_error(response: TestResponse):
        """Assert authentication-related error response."""
        assert response.status_code in [401, 403], \
            f"Expected authentication error (401/403), got {response.status_code}"
        
        # Check for redirect to login (common pattern)
        if response.status_code == 302:
            assert 'login' in response.location.lower(), \
                "Redirect should go to login page"
    
    @staticmethod
    def assert_rate_limit_error(response: TestResponse):
        """Assert rate limiting error with proper headers."""
        assert response.status_code == 429, \
            f"Expected 429 (Too Many Requests), got {response.status_code}"
        
        # Validate rate limiting headers
        assert 'X-RateLimit-Remaining' in response.headers or \
               'Retry-After' in response.headers, \
            "Rate limit response should include rate limiting headers"
        
        data = response.get_json()
        if data:
            error_msg = data.get('error', '').lower()
            assert any(keyword in error_msg for keyword in ['rate', 'limit', 'too many']), \
                "Error message should indicate rate limiting"
    
    @staticmethod
    def assert_json_structure(response: TestResponse, required_fields: List[str], 
                            optional_fields: Optional[List[str]] = None):
        """
        Assert JSON response has required structure.
        
        Args:
            response: Flask test response
            required_fields: Fields that must be present
            optional_fields: Fields that may be present
        """
        data = response.get_json()
        assert data is not None, "Response should contain JSON data"
        
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from response"
        
        if optional_fields:
            # Log which optional fields are present (for debugging)
            present_optional = [f for f in optional_fields if f in data]
            print(f"Optional fields present: {present_optional}")
    
    @staticmethod
    def assert_pagination_response(response: TestResponse, expected_page: int = 1, 
                                 expected_per_page: int = 20):
        """Assert paginated response structure."""
        ResponseAssertions.assert_success_response(response)
        
        data = response.get_json()
        required_pagination_fields = ['items', 'total', 'page', 'per_page', 'pages']
        
        for field in required_pagination_fields:
            assert field in data, f"Pagination field '{field}' missing"
        
        assert data['page'] == expected_page, \
            f"Expected page {expected_page}, got {data['page']}"
        assert data['per_page'] == expected_per_page, \
            f"Expected per_page {expected_per_page}, got {data['per_page']}"
        assert isinstance(data['items'], list), "Items should be a list"


class AuthenticationAssertions:
    """Authentication and authorization validation helpers."""
    
    @staticmethod
    def assert_user_logged_in(client, username: str):
        """
        Assert user is properly logged in with session validation.
        
        Args:
            client: Flask test client
            username: Expected logged-in username
        """
        with client.session_transaction() as sess:
            # Check for common session keys used by Flask-Login
            assert any(key in sess for key in ['user_id', '_user_id', 'username']), \
                "No user session found"
            
            # If username is stored in session, validate it
            if 'username' in sess:
                assert sess['username'] == username, \
                    f"Expected username '{username}', got '{sess['username']}'"
    
    @staticmethod
    def assert_user_logged_out(client):
        """Assert user is properly logged out."""
        with client.session_transaction() as sess:
            # Session should not contain user identification
            user_keys = ['user_id', '_user_id', 'username']
            for key in user_keys:
                assert key not in sess, f"User session key '{key}' still present after logout"
    
    @staticmethod
    def assert_csrf_token_present(response: TestResponse):
        """Assert CSRF token is present in response."""
        # Check for CSRF token in form or meta tag
        content = response.get_data(as_text=True)
        csrf_patterns = [
            r'<meta name="csrf-token"',
            r'name="csrf_token"',
            r'csrf_token'
        ]
        
        assert any(re.search(pattern, content, re.IGNORECASE) for pattern in csrf_patterns), \
            "CSRF token not found in response"
    
    @staticmethod
    def assert_protected_route_accessible(client, route: str):
        """Assert authenticated user can access protected route."""
        response = client.get(route)
        assert response.status_code == 200, \
            f"Authenticated user should access {route}, got {response.status_code}"
    
    @staticmethod
    def assert_protected_route_requires_auth(client, route: str):
        """Assert unauthenticated user cannot access protected route."""
        response = client.get(route)
        assert response.status_code in [302, 401, 403], \
            f"Unauthenticated user should not access {route}, got {response.status_code}"


class FileUploadAssertions:
    """File upload and processing validation helpers."""
    
    @staticmethod
    def assert_upload_success(response: TestResponse, filename: str, 
                            expected_file_type: Optional[str] = None):
        """
        Assert successful file upload with metadata validation.
        
        Args:
            response: Upload response
            filename: Expected filename
            expected_file_type: Expected file type (e.g., 'audio', 'image')
        """
        ResponseAssertions.assert_success_response(response)
        
        data = response.get_json()
        assert 'filename' in data, "Upload response should contain filename"
        assert filename in data['filename'], f"Filename should contain '{filename}'"
        
        if expected_file_type:
            assert 'file_type' in data, "Upload response should contain file_type"
            assert data['file_type'] == expected_file_type, \
                f"Expected file_type '{expected_file_type}', got '{data['file_type']}'"
        
        # Validate file processing metadata
        expected_metadata = ['file_size', 'upload_time']
        for field in expected_metadata:
            if field in data:
                assert data[field] is not None, f"Metadata field '{field}' should not be null"
    
    @staticmethod
    def assert_upload_rejected(response: TestResponse, reason_keywords: List[str]):
        """
        Assert file upload was rejected with specific reason.
        
        Args:
            response: Upload response
            reason_keywords: Keywords that should appear in rejection reason
        """
        ResponseAssertions.assert_error_response(response, 400, reason_keywords)
    
    @staticmethod
    def assert_quota_exceeded(response: TestResponse):
        """Assert upload was rejected due to quota exceeded."""
        FileUploadAssertions.assert_upload_rejected(response, ['quota', 'limit', 'exceeded'])
    
    @staticmethod
    def assert_invalid_file_type(response: TestResponse, file_extension: str):
        """Assert upload was rejected due to invalid file type."""
        FileUploadAssertions.assert_upload_rejected(response, ['invalid', 'format', file_extension])


class DatabaseAssertions:
    """Database state validation helpers."""
    
    @staticmethod
    def assert_record_created(model_class, **filter_kwargs):
        """
        Assert database record was created with specified attributes.
        
        Args:
            model_class: SQLAlchemy model class
            **filter_kwargs: Attributes to filter by
        """
        record = model_class.query.filter_by(**filter_kwargs).first()
        assert record is not None, \
            f"No {model_class.__name__} record found with {filter_kwargs}"
        return record
    
    @staticmethod
    def assert_record_updated(model_class, record_id: int, expected_values: Dict[str, Any]):
        """
        Assert database record was updated with expected values.
        
        Args:
            model_class: SQLAlchemy model class
            record_id: ID of record to check
            expected_values: Dict of field_name: expected_value
        """
        record = model_class.query.get(record_id)
        assert record is not None, f"No {model_class.__name__} record found with ID {record_id}"
        
        for field, expected_value in expected_values.items():
            actual_value = getattr(record, field, None)
            assert actual_value == expected_value, \
                f"Expected {field}={expected_value}, got {actual_value}"
    
    @staticmethod
    def assert_record_deleted(model_class, record_id: int):
        """Assert database record was deleted."""
        record = model_class.query.get(record_id)
        assert record is None, \
            f"{model_class.__name__} record with ID {record_id} should be deleted"
    
    @staticmethod
    def assert_record_count(model_class, expected_count: int, **filter_kwargs):
        """
        Assert count of records matching criteria.
        
        Args:
            model_class: SQLAlchemy model class
            expected_count: Expected number of records
            **filter_kwargs: Optional filter criteria
        """
        query = model_class.query
        if filter_kwargs:
            query = query.filter_by(**filter_kwargs)
        
        actual_count = query.count()
        assert actual_count == expected_count, \
            f"Expected {expected_count} {model_class.__name__} records, got {actual_count}"


class PredictionAssertions:
    """ML prediction and processing validation helpers."""
    
    @staticmethod
    def assert_prediction_response(response: TestResponse, expected_species: Optional[str] = None,
                                 min_confidence: float = 0.0):
        """
        Assert valid prediction response structure and content.
        
        Args:
            response: Prediction response
            expected_species: Expected species prediction
            min_confidence: Minimum confidence threshold
        """
        ResponseAssertions.assert_success_response(response)
        
        data = response.get_json()
        required_fields = ['predictions', 'processing_time', 'file_info']
        ResponseAssertions.assert_json_structure(response, required_fields)
        
        # Validate predictions structure
        predictions = data['predictions']
        if isinstance(predictions, dict) and 'species' in predictions:
            # Single prediction format
            assert 'confidence' in predictions, "Prediction should include confidence"
            assert predictions['confidence'] >= min_confidence, \
                f"Confidence {predictions['confidence']} below threshold {min_confidence}"
            
            if expected_species:
                assert predictions['species'] == expected_species, \
                    f"Expected species '{expected_species}', got '{predictions['species']}'"
        
        elif isinstance(predictions, list):
            # Multiple predictions format
            assert len(predictions) > 0, "Should have at least one prediction"
            for pred in predictions:
                assert 'label' in pred and 'probability' in pred, \
                    "Each prediction should have label and probability"
                assert 0 <= pred['probability'] <= 1, \
                    f"Probability {pred['probability']} should be between 0 and 1"
        
        # Validate processing metadata
        assert data['processing_time'] > 0, "Processing time should be positive"
        assert 'filename' in data['file_info'], "File info should include filename"
    
    @staticmethod
    def assert_prediction_cached(response: TestResponse):
        """Assert prediction result was served from cache."""
        data = response.get_json()
        file_info = data.get('file_info', {})
        assert file_info.get('cached') is True, "Prediction should be marked as cached"
    
    @staticmethod
    def assert_prediction_not_cached(response: TestResponse):
        """Assert prediction result was computed fresh (not cached)."""
        data = response.get_json()
        file_info = data.get('file_info', {})
        assert file_info.get('cached') is False, "Prediction should not be cached"


class SecurityAssertions:
    """Security and compliance validation helpers."""
    
    @staticmethod
    def assert_security_headers(response: TestResponse):
        """Assert presence of important security headers."""
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block'
        }
        
        for header, expected_value in security_headers.items():
            assert header in response.headers, f"Missing security header: {header}"
            if expected_value:
                assert response.headers[header] == expected_value, \
                    f"Header {header} should be '{expected_value}', got '{response.headers[header]}'"
    
    @staticmethod
    def assert_no_sensitive_data_in_response(response: TestResponse, 
                                           sensitive_patterns: List[str]):
        """
        Assert response doesn't contain sensitive data.
        
        Args:
            response: Response to check
            sensitive_patterns: List of patterns that shouldn't appear
        """
        content = response.get_data(as_text=True)
        
        for pattern in sensitive_patterns:
            assert pattern not in content, \
                f"Sensitive pattern '{pattern}' found in response"
    
    @staticmethod
    def assert_content_security_policy(response: TestResponse):
        """Assert Content Security Policy header is present and configured."""
        assert 'Content-Security-Policy' in response.headers, \
            "Missing Content-Security-Policy header"
        
        csp = response.headers['Content-Security-Policy']
        required_directives = ['default-src', 'script-src', 'style-src']
        
        for directive in required_directives:
            assert directive in csp, f"CSP missing directive: {directive}"


class PerformanceAssertions:
    """Performance and timing validation helpers."""
    
    @staticmethod
    def assert_response_time_under(response_time: float, max_seconds: float):
        """Assert response time is under specified threshold."""
        assert response_time <= max_seconds, \
            f"Response time {response_time}s exceeds threshold {max_seconds}s"
    
    @staticmethod
    def assert_memory_usage_reasonable(memory_before: int, memory_after: int, 
                                     max_increase_mb: int = 100):
        """Assert memory usage increase is reasonable."""
        increase_mb = (memory_after - memory_before) / (1024 * 1024)
        assert increase_mb <= max_increase_mb, \
            f"Memory usage increased by {increase_mb:.2f}MB, exceeds {max_increase_mb}MB"