"""
Response fixtures and schemas for testing.

Provides standardized response structures and validation schemas
for consistent testing across the NightScan application.
"""

import pytest
from typing import Dict, Any, List
from datetime import datetime, timezone


class ResponseSchemas:
    """Standard response schemas for validation."""
    
    @staticmethod
    def success_response(data: Any = None) -> Dict[str, Any]:
        """Standard success response schema."""
        return {
            'status': 'success',
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def error_response(error_message: str, error_code: str = None) -> Dict[str, Any]:
        """Standard error response schema."""
        response = {
            'status': 'error',
            'error': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        if error_code:
            response['code'] = error_code
        return response
    
    @staticmethod
    def validation_error_response(field_errors: Dict[str, str]) -> Dict[str, Any]:
        """Standard validation error response schema."""
        return {
            'status': 'error',
            'error': 'Validation failed',
            'code': 'VALIDATION_ERROR',
            'details': field_errors,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def prediction_response(species: str, confidence: float, 
                          filename: str, processing_time: float,
                          cached: bool = False) -> Dict[str, Any]:
        """Standard prediction response schema."""
        return {
            'status': 'success',
            'predictions': {
                'species': species,
                'confidence': confidence
            },
            'processing_time': processing_time,
            'file_info': {
                'filename': filename,
                'cached': cached,
                'size_bytes': 12345,
                'duration_seconds': 8.0
            },
            'prediction_id': 'pred_12345',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    @staticmethod
    def upload_response(filename: str, file_type: str, 
                       file_size: int) -> Dict[str, Any]:
        """Standard file upload response schema."""
        return {
            'status': 'success',
            'filename': filename,
            'file_type': file_type,
            'file_size': file_size,
            'upload_time': datetime.now(timezone.utc).isoformat(),
            'upload_id': 'upload_12345'
        }
    
    @staticmethod
    def pagination_response(items: List[Any], page: int = 1, 
                          per_page: int = 20, total: int = None) -> Dict[str, Any]:
        """Standard pagination response schema."""
        if total is None:
            total = len(items)
        
        pages = (total + per_page - 1) // per_page  # Ceiling division
        
        return {
            'items': items,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': pages,
                'has_next': page < pages,
                'has_prev': page > 1
            }
        }
    
    @staticmethod
    def health_response(service_name: str = 'nightscan', 
                       checks: Dict[str, str] = None) -> Dict[str, Any]:
        """Standard health check response schema."""
        if checks is None:
            checks = {
                'database': 'healthy',
                'redis': 'healthy', 
                'model': 'healthy'
            }
        
        return {
            'status': 'healthy',
            'service': service_name,
            'version': '1.0.0',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': checks
        }


@pytest.fixture
def success_response_factory():
    """Factory for creating success responses."""
    return ResponseSchemas.success_response


@pytest.fixture
def error_response_factory():
    """Factory for creating error responses."""
    return ResponseSchemas.error_response


@pytest.fixture
def prediction_response_factory():
    """Factory for creating prediction responses."""
    return ResponseSchemas.prediction_response


@pytest.fixture
def upload_response_factory():
    """Factory for creating upload responses."""
    return ResponseSchemas.upload_response


@pytest.fixture
def pagination_response_factory():
    """Factory for creating pagination responses."""
    return ResponseSchemas.pagination_response


@pytest.fixture
def health_response_factory():
    """Factory for creating health check responses."""
    return ResponseSchemas.health_response


# Mock response fixtures for testing
@pytest.fixture
def mock_owl_prediction():
    """Mock successful owl prediction response."""
    return ResponseSchemas.prediction_response(
        species='Great Horned Owl',
        confidence=0.89,
        filename='owl_sound.wav',
        processing_time=1.23
    )


@pytest.fixture
def mock_bat_prediction():
    """Mock successful bat prediction response."""
    return ResponseSchemas.prediction_response(
        species='Big Brown Bat',
        confidence=0.95,
        filename='bat_sound.wav',
        processing_time=0.87
    )


@pytest.fixture
def mock_validation_error():
    """Mock validation error response."""
    return ResponseSchemas.validation_error_response({
        'file': 'File is required',
        'format': 'Only WAV files are supported'
    })


@pytest.fixture
def mock_rate_limit_error():
    """Mock rate limit error response."""
    return ResponseSchemas.error_response(
        'Rate limit exceeded. Try again in 60 seconds.',
        'RATE_LIMIT_EXCEEDED'
    )


@pytest.fixture
def mock_auth_error():
    """Mock authentication error response."""
    return ResponseSchemas.error_response(
        'Authentication required. Please log in.',
        'AUTHENTICATION_REQUIRED'
    )


@pytest.fixture 
def mock_quota_exceeded_error():
    """Mock quota exceeded error response."""
    return ResponseSchemas.error_response(
        'Monthly upload quota exceeded. Upgrade your plan for more uploads.',
        'QUOTA_EXCEEDED'
    )


@pytest.fixture
def mock_detections_list():
    """Mock detections list for pagination testing."""
    detections = []
    for i in range(25):  # More than one page
        detections.append({
            'id': i + 1,
            'species': 'Owl' if i % 2 == 0 else 'Bat',
            'confidence': 0.8 + (i % 3) * 0.1,
            'location': f'Zone {i % 5}',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    return detections


@pytest.fixture
def mock_paginated_detections(mock_detections_list):
    """Mock paginated detections response."""
    return ResponseSchemas.pagination_response(
        items=mock_detections_list[:20],  # First page
        page=1,
        per_page=20,
        total=len(mock_detections_list)
    )


# Test data fixtures
@pytest.fixture
def valid_wav_file_data():
    """Valid WAV file header for testing."""
    return (
        b'RIFF'  # ChunkID
        b'\x24\x00\x00\x00'  # ChunkSize (36 bytes)
        b'WAVE'  # Format
        b'fmt '  # Subchunk1ID
        b'\x10\x00\x00\x00'  # Subchunk1Size (16)
        b'\x01\x00'  # AudioFormat (PCM)
        b'\x01\x00'  # NumChannels (1)
        b'\x44\xAC\x00\x00'  # SampleRate (44100)
        b'\x88\x58\x01\x00'  # ByteRate
        b'\x02\x00'  # BlockAlign
        b'\x10\x00'  # BitsPerSample (16)
        b'data'  # Subchunk2ID
        b'\x00\x00\x00\x00'  # Subchunk2Size (0 - no audio data)
    )


@pytest.fixture
def invalid_file_data():
    """Invalid file data for testing."""
    return b'This is not a valid audio file'


@pytest.fixture
def test_user_data():
    """Standard test user data."""
    return {
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'SecurePassword123!',
        'plan': 'basic'
    }


@pytest.fixture
def test_admin_data():
    """Standard test admin user data."""
    return {
        'username': 'admin',
        'email': 'admin@example.com', 
        'password': 'AdminPassword123!',
        'plan': 'enterprise',
        'is_admin': True
    }


# Security test fixtures
@pytest.fixture
def sensitive_data_patterns():
    """Patterns that should not appear in responses."""
    return [
        'password',
        'secret_key',
        'api_key',
        'private_key',
        'jwt_secret',
        'database_url',
        'redis_url'
    ]


@pytest.fixture
def expected_security_headers():
    """Expected security headers in responses."""
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block'
    }


# Performance test fixtures
@pytest.fixture
def performance_thresholds():
    """Performance thresholds for different operations."""
    return {
        'health_check': 0.1,  # 100ms
        'login': 0.5,  # 500ms
        'file_upload': 2.0,  # 2 seconds
        'prediction': 3.0,  # 3 seconds
        'api_endpoint': 1.0  # 1 second
    }


@pytest.fixture
def memory_usage_limits():
    """Memory usage limits for operations."""
    return {
        'file_upload': 50,  # 50MB increase max
        'prediction': 100,  # 100MB increase max
        'batch_processing': 200  # 200MB increase max
    }