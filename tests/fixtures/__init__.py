"""
Test fixtures package for NightScan.

Provides standardized fixtures and test data for consistent testing
across the NightScan application.
"""

from .response_fixtures import (
    ResponseSchemas,
    success_response_factory,
    error_response_factory,
    prediction_response_factory,
    upload_response_factory,
    pagination_response_factory,
    health_response_factory,
    mock_owl_prediction,
    mock_bat_prediction,
    mock_validation_error,
    mock_rate_limit_error,
    mock_auth_error,
    mock_quota_exceeded_error,
    mock_detections_list,
    mock_paginated_detections,
    valid_wav_file_data,
    invalid_file_data,
    test_user_data,
    test_admin_data,
    sensitive_data_patterns,
    expected_security_headers,
    performance_thresholds,
    memory_usage_limits
)

__all__ = [
    'ResponseSchemas',
    'success_response_factory',
    'error_response_factory', 
    'prediction_response_factory',
    'upload_response_factory',
    'pagination_response_factory',
    'health_response_factory',
    'mock_owl_prediction',
    'mock_bat_prediction',
    'mock_validation_error',
    'mock_rate_limit_error',
    'mock_auth_error',
    'mock_quota_exceeded_error',
    'mock_detections_list',
    'mock_paginated_detections',
    'valid_wav_file_data',
    'invalid_file_data',
    'test_user_data',
    'test_admin_data',
    'sensitive_data_patterns',
    'expected_security_headers',
    'performance_thresholds',
    'memory_usage_limits'
]