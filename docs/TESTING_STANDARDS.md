# NightScan Testing Standards and Guidelines

## Overview

This document establishes comprehensive testing standards for the NightScan project to ensure high-quality, reliable tests that go beyond shallow status code checking to validate application behavior, state changes, and business logic.

## Testing Philosophy

### Core Principles

1. **Comprehensive Validation**: Tests should validate response content, data structure, state changes, and business logic - not just HTTP status codes
2. **Meaningful Assertions**: Every assertion should verify specific application behavior or requirement
3. **Test Isolation**: Each test should be independent and not rely on external state
4. **Realistic Data**: Use realistic test data that represents actual usage scenarios
5. **Error Scenarios**: Test both success and failure paths comprehensively

### Test Quality Standards

- **Shallow Testing** ❌: Only checking status codes
- **Deep Testing** ✅: Validating response content, state changes, business logic

## Test Structure Guidelines

### Standard Test Pattern

```python
def test_feature_behavior():
    # 1. Setup/Arrange - Prepare test data and environment
    test_data = prepare_test_data()
    
    # 2. Execute/Act - Perform the action being tested
    response = client.post('/endpoint', data=test_data)
    
    # 3. Assert - Multiple layers of validation
    # Status code validation
    assert response.status_code == 200
    
    # Response structure validation
    data = response.get_json()
    assert 'required_field' in data
    
    # Business logic validation
    assert data['result'] == expected_value
    
    # State change validation (if applicable)
    db_record = Model.query.get(test_id)
    assert db_record.field == expected_db_value
    
    # Side effects validation (if applicable)
    assert mock_service.called_with(expected_params)
```

## Assertion Helpers Usage

### Import Standards

```python
from tests.helpers import (
    ResponseAssertions,
    AuthenticationAssertions,
    FileUploadAssertions,
    DatabaseAssertions,
    PredictionAssertions,
    SecurityAssertions,
    PerformanceAssertions
)
```

### Response Validation

#### Success Responses
```python
# Basic success validation
ResponseAssertions.assert_success_response(response)

# Success with expected data
ResponseAssertions.assert_success_response(response, {
    'status': 'success',
    'data': expected_data
})

# JSON structure validation
ResponseAssertions.assert_json_structure(response, 
    required_fields=['id', 'name', 'status'],
    optional_fields=['description', 'metadata']
)
```

#### Error Responses
```python
# Generic error validation
ResponseAssertions.assert_error_response(response, 400, ['validation', 'required'])

# Specific validation errors
ResponseAssertions.assert_validation_error(response, 'email')

# Rate limiting errors
ResponseAssertions.assert_rate_limit_error(response)

# Authentication errors
ResponseAssertions.assert_authentication_error(response)
```

#### Pagination Responses
```python
ResponseAssertions.assert_pagination_response(response, 
    expected_page=1, 
    expected_per_page=20
)
```

### Authentication Testing

```python
# Verify user login state
AuthenticationAssertions.assert_user_logged_in(client, 'username')

# Verify user logout state
AuthenticationAssertions.assert_user_logged_out(client)

# Test protected route access
AuthenticationAssertions.assert_protected_route_accessible(client, '/dashboard')
AuthenticationAssertions.assert_protected_route_requires_auth(client, '/admin')

# CSRF token validation
AuthenticationAssertions.assert_csrf_token_present(response)
```

### File Upload Testing

```python
# Successful upload validation
FileUploadAssertions.assert_upload_success(response, 'test.wav', 'audio')

# Upload rejection validation
FileUploadAssertions.assert_upload_rejected(response, ['invalid', 'format'])

# Quota validation
FileUploadAssertions.assert_quota_exceeded(response)
FileUploadAssertions.assert_invalid_file_type(response, 'txt')
```

### Database State Validation

```python
# Record creation validation
user = DatabaseAssertions.assert_record_created(User, username='testuser')

# Record update validation
DatabaseAssertions.assert_record_updated(User, user_id, {
    'email': 'new@example.com',
    'updated_at': expected_timestamp
})

# Record deletion validation
DatabaseAssertions.assert_record_deleted(User, user_id)

# Record count validation
DatabaseAssertions.assert_record_count(Prediction, 5, user_id=user.id)
```

### ML Prediction Testing

```python
# Prediction response validation
PredictionAssertions.assert_prediction_response(response, 
    expected_species='Great Horned Owl',
    min_confidence=0.8
)

# Cache validation
PredictionAssertions.assert_prediction_cached(response)
PredictionAssertions.assert_prediction_not_cached(response)
```

### Security Testing

```python
# Security headers validation
SecurityAssertions.assert_security_headers(response)

# Content Security Policy validation
SecurityAssertions.assert_content_security_policy(response)

# Sensitive data validation
SecurityAssertions.assert_no_sensitive_data_in_response(response, [
    'password', 'secret_key', 'api_key'
])
```

### Performance Testing

```python
# Response time validation
PerformanceAssertions.assert_response_time_under(response_time, 1.0)

# Memory usage validation
PerformanceAssertions.assert_memory_usage_reasonable(
    memory_before, memory_after, max_increase_mb=50
)
```

## Test Categories and Markers

### Pytest Markers Usage

```python
import pytest

@pytest.mark.unit
def test_model_validation():
    """Unit test for model validation logic."""
    pass

@pytest.mark.integration
def test_auth_flow():
    """Integration test for authentication flow."""
    pass

@pytest.mark.performance_critical
def test_prediction_performance():
    """Performance test with SLA requirements."""
    pass

@pytest.mark.slow
def test_large_file_processing():
    """Test that takes longer than 30 seconds."""
    pass
```

### Test Organization

```
tests/
├── unit/                 # Fast unit tests
├── integration/          # Service integration tests
├── performance/          # Performance and load tests
├── helpers/             # Reusable assertion helpers
├── fixtures/            # Test data and response fixtures
└── conftest.py          # Shared pytest configuration
```

## Common Anti-Patterns to Avoid

### ❌ Shallow Status Code Testing
```python
def test_endpoint():
    response = client.post('/api/data')
    assert response.status_code == 200  # INSUFFICIENT
```

### ✅ Comprehensive Testing
```python
def test_endpoint():
    response = client.post('/api/data', json=test_data)
    
    # Status validation
    ResponseAssertions.assert_success_response(response)
    
    # Content validation
    data = response.get_json()
    assert data['id'] is not None
    assert data['status'] == 'created'
    
    # State validation
    record = DatabaseAssertions.assert_record_created(DataModel, 
        name=test_data['name']
    )
    assert record.status == 'active'
```

### ❌ Missing Error Validation
```python
def test_invalid_input():
    response = client.post('/api/data', json={'invalid': 'data'})
    assert response.status_code == 400  # INSUFFICIENT
```

### ✅ Complete Error Validation
```python
def test_invalid_input():
    response = client.post('/api/data', json={'invalid': 'data'})
    
    ResponseAssertions.assert_validation_error(response, 'name')
    
    data = response.get_json()
    assert 'name is required' in data['error']
    
    # Ensure no side effects
    assert DataModel.query.count() == 0
```

### ❌ No State Change Verification
```python
def test_user_update():
    response = client.put('/api/users/1', json={'email': 'new@test.com'})
    assert response.status_code == 200  # INSUFFICIENT
```

### ✅ State Change Validation
```python
def test_user_update():
    original_user = User.query.get(1)
    original_email = original_user.email
    
    response = client.put('/api/users/1', json={'email': 'new@test.com'})
    
    ResponseAssertions.assert_success_response(response)
    
    # Verify database state changed
    DatabaseAssertions.assert_record_updated(User, 1, {
        'email': 'new@test.com'
    })
    
    # Verify old email is no longer present
    updated_user = User.query.get(1)
    assert updated_user.email != original_email
```

## Test Data Management

### Using Fixtures

```python
from tests.fixtures import (
    test_user_data,
    valid_wav_file_data,
    mock_owl_prediction,
    performance_thresholds
)

def test_user_registration(test_user_data):
    response = client.post('/register', json=test_user_data)
    # Test implementation
```

### Factory Pattern

```python
@pytest.fixture
def user_factory():
    """Factory for creating test users with different attributes."""
    def _create_user(**kwargs):
        default_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'SecurePassword123!'
        }
        default_data.update(kwargs)
        return default_data
    return _create_user

def test_admin_user(user_factory):
    admin_data = user_factory(is_admin=True, username='admin')
    # Test implementation
```

## Performance Testing Guidelines

### Response Time Thresholds

| Endpoint Type | Max Response Time |
|---------------|-------------------|
| Health Check  | 100ms            |
| Login         | 500ms            |
| File Upload   | 2s               |
| ML Prediction | 3s               |
| API Endpoint  | 1s               |

### Memory Usage Limits

| Operation | Max Memory Increase |
|-----------|-------------------|
| File Upload | 50MB |
| ML Prediction | 100MB |
| Batch Processing | 200MB |

### Performance Test Example

```python
import time
import psutil

def test_prediction_performance(performance_thresholds):
    process = psutil.Process()
    memory_before = process.memory_info().rss
    
    start_time = time.time()
    response = client.post('/api/predict', data=audio_file)
    end_time = time.time()
    
    memory_after = process.memory_info().rss
    response_time = end_time - start_time
    
    # Performance assertions
    PerformanceAssertions.assert_response_time_under(
        response_time, 
        performance_thresholds['prediction']
    )
    
    PerformanceAssertions.assert_memory_usage_reasonable(
        memory_before, 
        memory_after, 
        max_increase_mb=100
    )
```

## Security Testing Requirements

### Required Security Validations

1. **Authentication**: Verify login/logout flows and session management
2. **Authorization**: Test access control and permission enforcement
3. **Input Validation**: Validate input sanitization and XSS prevention
4. **CSRF Protection**: Ensure CSRF tokens are present and validated
5. **Security Headers**: Verify security headers are set correctly
6. **Sensitive Data**: Ensure no sensitive data leaks in responses

### Security Test Example

```python
def test_sensitive_data_protection(sensitive_data_patterns):
    response = client.get('/api/user/profile')
    
    SecurityAssertions.assert_security_headers(response)
    SecurityAssertions.assert_no_sensitive_data_in_response(
        response, 
        sensitive_data_patterns
    )
    
    # Verify user data is sanitized
    data = response.get_json()
    assert 'password' not in data
    assert 'api_key' not in data
```

## CI/CD Integration

### Test Execution Strategy

```bash
# Unit tests (fast feedback)
pytest tests/unit/ -m "not slow"

# Integration tests (comprehensive validation)
pytest tests/integration/ --maxfail=5

# Performance tests (SLA validation)
pytest tests/performance/ -m performance_critical

# Full test suite (pre-deployment)
pytest tests/ --cov=. --cov-report=xml
```

### Quality Gates

- **Unit Test Coverage**: > 80%
- **Integration Test Coverage**: > 70%
- **Performance Tests**: All must pass SLA requirements
- **Security Tests**: Zero vulnerabilities detected
- **Test Execution Time**: < 10 minutes for full suite

## Conclusion

These testing standards ensure that the NightScan project maintains high code quality and reliability. By following these guidelines and using the provided assertion helpers, tests will move beyond shallow status code checking to provide comprehensive validation of application behavior.

All team members should:
1. Use the provided assertion helpers for consistent validation
2. Follow the standard test patterns outlined in this document
3. Validate state changes and business logic, not just HTTP responses
4. Include both positive and negative test scenarios
5. Maintain test performance within established thresholds

For questions or updates to these standards, please create an issue in the project repository.