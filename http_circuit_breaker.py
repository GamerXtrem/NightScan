"""
HTTP Circuit Breaker for NightScan

Provides specialized circuit breaker protection for HTTP/API calls
with intelligent retry logic, connection pooling, and fallback mechanisms.

Features:
- HTTP request protection with timeouts
- Exponential backoff retry logic
- Connection pooling with health monitoring
- Request/response validation
- Async and sync operation support
- Detailed HTTP metrics and error categorization
"""

import time
import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
import json

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    from requests.exceptions import (
        RequestException, ConnectionError as RequestsConnectionError,
        Timeout, HTTPError, TooManyRedirects
    )
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None
    RequestException = Exception
    RequestsConnectionError = Exception
    Timeout = Exception
    HTTPError = Exception

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenException,
    register_circuit_breaker
)
from exceptions import ExternalServiceError, PredictionError

logger = logging.getLogger(__name__)


@dataclass
class HTTPCircuitBreakerConfig(CircuitBreakerConfig):
    """Extended configuration for HTTP circuit breaker."""
    # Request configuration
    connect_timeout: float = 3.0          # Connection timeout
    read_timeout: float = 10.0            # Read timeout
    total_timeout: float = 30.0           # Total request timeout
    
    # Retry configuration
    max_retries: int = 3                  # Maximum retry attempts
    backoff_factor: float = 0.3           # Backoff factor for retries
    retry_on_status: List[int] = field(default_factory=lambda: [500, 502, 503, 504])
    
    # Connection pooling
    pool_connections: int = 10            # Number of connection pools
    pool_maxsize: int = 20               # Max connections per pool
    max_redirects: int = 3               # Maximum redirects to follow
    
    # Health checking
    health_check_url: Optional[str] = None    # URL for health checks
    health_check_interval: float = 60.0      # Seconds between health checks
    health_check_timeout: float = 5.0        # Timeout for health checks
    
    # Response validation
    expected_status_codes: List[int] = field(default_factory=lambda: [200, 201, 202])
    validate_response_json: bool = False      # Validate JSON responses
    max_response_size: int = 10 * 1024 * 1024  # 10MB max response size
    
    # Service-specific settings
    service_name: Optional[str] = None        # Name of the service being called
    base_url: Optional[str] = None           # Base URL for relative requests
    
    def __post_init__(self):
        super().__post_init__()
        
        # HTTP-specific exception handling
        if REQUESTS_AVAILABLE:
            self.expected_exception = (
                RequestException, RequestsConnectionError, Timeout, HTTPError,
                TooManyRedirects, ExternalServiceError
            )
        else:
            self.expected_exception = (ExternalServiceError,)


class HTTPMetrics:
    """Detailed HTTP metrics tracking."""
    
    def __init__(self):
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.timeout_count = 0
        self.retry_count = 0
        
        # Response time tracking
        self.total_response_time = 0.0
        self.min_response_time = float('inf')
        self.max_response_time = 0.0
        
        # Status code tracking
        self.status_codes = {}
        
        # Error categorization
        self.connection_errors = 0
        self.timeout_errors = 0
        self.http_errors = 0
        self.validation_errors = 0
        
        # Lock for thread safety
        self.lock = threading.RLock()
    
    def record_request(self, response_time: float, status_code: int = None, 
                      error_type: str = None, retries: int = 0):
        """Record metrics for a request."""
        with self.lock:
            self.request_count += 1
            self.retry_count += retries
            
            if error_type:
                self.error_count += 1
                if error_type == 'connection':
                    self.connection_errors += 1
                elif error_type == 'timeout':
                    self.timeout_errors += 1
                elif error_type == 'http':
                    self.http_errors += 1
                elif error_type == 'validation':
                    self.validation_errors += 1
            else:
                self.success_count += 1
            
            # Response time tracking
            self.total_response_time += response_time
            self.min_response_time = min(self.min_response_time, response_time)
            self.max_response_time = max(self.max_response_time, response_time)
            
            # Status code tracking
            if status_code:
                self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self.lock:
            avg_response_time = (self.total_response_time / self.request_count 
                               if self.request_count > 0 else 0)
            
            success_rate = (self.success_count / self.request_count 
                          if self.request_count > 0 else 0)
            
            return {
                'request_count': self.request_count,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'timeout_count': self.timeout_count,
                'retry_count': self.retry_count,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'min_response_time': self.min_response_time if self.min_response_time != float('inf') else 0,
                'max_response_time': self.max_response_time,
                'status_codes': self.status_codes.copy(),
                'connection_errors': self.connection_errors,
                'timeout_errors': self.timeout_errors,
                'http_errors': self.http_errors,
                'validation_errors': self.validation_errors
            }


class HTTPCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for HTTP/API operations.
    
    Provides protection for HTTP requests with intelligent retry logic,
    connection pooling, and comprehensive error handling.
    """
    
    def __init__(self, config: HTTPCircuitBreakerConfig):
        super().__init__(config)
        self.http_config = config
        self.http_metrics = HTTPMetrics()
        
        # HTTP session with connection pooling
        self._session = None
        self._async_session = None
        self._setup_session()
        
        # Health checking
        self._health_check_task = None
        self._health_status = True
        
        if config.health_check_url and config.health_check_interval > 0:
            self._start_health_monitoring()
        
        logger.info(f"HTTP circuit breaker '{config.name}' initialized for {config.service_name or 'unknown service'}")
    
    def _setup_session(self):
        """Setup HTTP session with connection pooling and retry strategy."""
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available, HTTP circuit breaker will have limited functionality")
            return
        
        self._session = requests.Session()
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=self.http_config.max_retries,
            backoff_factor=self.http_config.backoff_factor,
            status_forcelist=self.http_config.retry_on_status,
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        # Setup adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.http_config.pool_connections,
            pool_maxsize=self.http_config.pool_maxsize
        )
        
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # Set default timeouts
        self._session.timeout = (self.http_config.connect_timeout, self.http_config.read_timeout)
        
        logger.info(f"HTTP session configured with {self.http_config.pool_connections} connection pools")
    
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make HTTP request with circuit breaker protection.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for requests
            
        Returns:
            requests.Response object
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            ExternalServiceError: When request fails
        """
        if self.http_config.base_url and not url.startswith(('http://', 'https://')):
            url = f"{self.http_config.base_url.rstrip('/')}/{url.lstrip('/')}"
        
        def make_request():
            return self._execute_request(method, url, **kwargs)
        
        try:
            return self.call(make_request)
        except CircuitBreakerOpenException as e:
            # Try fallback if available
            fallback_response = self._try_fallback(method, url, **kwargs)
            if fallback_response:
                return fallback_response
            raise e
    
    async def request_async(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Make async HTTP request with circuit breaker protection.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for aiohttp
            
        Returns:
            aiohttp.ClientResponse object
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            ExternalServiceError: When request fails
        """
        if not AIOHTTP_AVAILABLE:
            raise ExternalServiceError("aiohttp not available for async requests")
        
        if self.http_config.base_url and not url.startswith(('http://', 'https://')):
            url = f"{self.http_config.base_url.rstrip('/')}/{url.lstrip('/')}"
        
        async def make_async_request():
            return await self._execute_request_async(method, url, **kwargs)
        
        try:
            return await self.call_async(make_async_request)
        except CircuitBreakerOpenException as e:
            # Try fallback if available
            fallback_response = await self._try_fallback_async(method, url, **kwargs)
            if fallback_response:
                return fallback_response
            raise e
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make GET request."""
        return self.request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Make POST request."""
        return self.request('POST', url, **kwargs)
    
    def put(self, url: str, **kwargs) -> requests.Response:
        """Make PUT request."""
        return self.request('PUT', url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """Make DELETE request."""
        return self.request('DELETE', url, **kwargs)
    
    async def get_async(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make async GET request."""
        return await self.request_async('GET', url, **kwargs)
    
    async def post_async(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make async POST request."""
        return await self.request_async('POST', url, **kwargs)
    
    def _execute_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Execute HTTP request with comprehensive error handling."""
        if not self._session:
            raise ExternalServiceError("HTTP session not available")
        
        start_time = time.time()
        retries = 0
        
        # Set timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = (self.http_config.connect_timeout, self.http_config.read_timeout)
        
        try:
            response = self._session.request(method, url, **kwargs)
            execution_time = time.time() - start_time
            
            # Validate response
            self._validate_response(response)
            
            # Record success metrics
            self.http_metrics.record_request(execution_time, response.status_code, retries=retries)
            
            logger.debug(f"HTTP {method} {url} completed in {execution_time:.2f}s with status {response.status_code}")
            return response
            
        except RequestsConnectionError as e:
            execution_time = time.time() - start_time
            self.http_metrics.record_request(execution_time, error_type='connection', retries=retries)
            logger.error(f"HTTP connection error for {method} {url}: {e}")
            raise ExternalServiceError(f"Connection failed to {url}: {str(e)}")
            
        except Timeout as e:
            execution_time = time.time() - start_time
            self.http_metrics.record_request(execution_time, error_type='timeout', retries=retries)
            logger.error(f"HTTP timeout for {method} {url}: {e}")
            raise ExternalServiceError(f"Request timeout for {url}: {str(e)}")
            
        except HTTPError as e:
            execution_time = time.time() - start_time
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else None
            self.http_metrics.record_request(execution_time, status_code, error_type='http', retries=retries)
            logger.error(f"HTTP error for {method} {url}: {e}")
            raise ExternalServiceError(f"HTTP error for {url}: {str(e)}")
            
        except RequestException as e:
            execution_time = time.time() - start_time
            self.http_metrics.record_request(execution_time, error_type='http', retries=retries)
            logger.error(f"HTTP request error for {method} {url}: {e}")
            raise ExternalServiceError(f"Request failed for {url}: {str(e)}")
    
    async def _execute_request_async(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Execute async HTTP request with comprehensive error handling."""
        if not self._async_session:
            # Create session if not exists
            timeout = aiohttp.ClientTimeout(
                total=self.http_config.total_timeout,
                connect=self.http_config.connect_timeout,
                sock_read=self.http_config.read_timeout
            )
            self._async_session = aiohttp.ClientSession(timeout=timeout)
        
        start_time = time.time()
        retries = 0
        
        try:
            async with self._async_session.request(method, url, **kwargs) as response:
                execution_time = time.time() - start_time
                
                # Validate response
                await self._validate_response_async(response)
                
                # Record success metrics
                self.http_metrics.record_request(execution_time, response.status, retries=retries)
                
                logger.debug(f"Async HTTP {method} {url} completed in {execution_time:.2f}s with status {response.status}")
                return response
                
        except aiohttp.ClientConnectionError as e:
            execution_time = time.time() - start_time
            self.http_metrics.record_request(execution_time, error_type='connection', retries=retries)
            logger.error(f"Async HTTP connection error for {method} {url}: {e}")
            raise ExternalServiceError(f"Async connection failed to {url}: {str(e)}")
            
        except aiohttp.ServerTimeoutError as e:
            execution_time = time.time() - start_time
            self.http_metrics.record_request(execution_time, error_type='timeout', retries=retries)
            logger.error(f"Async HTTP timeout for {method} {url}: {e}")
            raise ExternalServiceError(f"Async request timeout for {url}: {str(e)}")
            
        except aiohttp.ClientError as e:
            execution_time = time.time() - start_time
            self.http_metrics.record_request(execution_time, error_type='http', retries=retries)
            logger.error(f"Async HTTP error for {method} {url}: {e}")
            raise ExternalServiceError(f"Async HTTP error for {url}: {str(e)}")
    
    def _validate_response(self, response: requests.Response):
        """Validate HTTP response."""
        # Check status code
        if response.status_code not in self.http_config.expected_status_codes:
            if response.status_code >= 400:
                response.raise_for_status()
        
        # Check response size
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > self.http_config.max_response_size:
            raise ExternalServiceError(f"Response too large: {content_length} bytes")
        
        # Validate JSON if required
        if self.http_config.validate_response_json:
            try:
                response.json()
            except ValueError as e:
                self.http_metrics.record_request(0, response.status_code, error_type='validation')
                raise ExternalServiceError(f"Invalid JSON response: {str(e)}")
    
    async def _validate_response_async(self, response: aiohttp.ClientResponse):
        """Validate async HTTP response."""
        # Check status code
        if response.status not in self.http_config.expected_status_codes:
            if response.status >= 400:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status
                )
        
        # Check response size
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > self.http_config.max_response_size:
            raise ExternalServiceError(f"Response too large: {content_length} bytes")
        
        # Validate JSON if required
        if self.http_config.validate_response_json:
            try:
                await response.json()
            except ValueError as e:
                self.http_metrics.record_request(0, response.status, error_type='validation')
                raise ExternalServiceError(f"Invalid JSON response: {str(e)}")
    
    def _try_fallback(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Try fallback mechanisms when circuit is open."""
        # This could be implemented to:
        # 1. Return cached responses
        # 2. Use alternative endpoints
        # 3. Return default/mock responses
        # For now, return None (no fallback)
        
        logger.info(f"No fallback available for {method} {url}")
        return None
    
    async def _try_fallback_async(self, method: str, url: str, **kwargs) -> Optional[aiohttp.ClientResponse]:
        """Try async fallback mechanisms when circuit is open."""
        logger.info(f"No async fallback available for {method} {url}")
        return None
    
    def _start_health_monitoring(self):
        """Start background health monitoring."""
        if not self.http_config.health_check_url:
            return
        
        def health_check_loop():
            while self._health_check_task:
                try:
                    start_time = time.time()
                    response = self._session.get(
                        self.http_config.health_check_url,
                        timeout=self.http_config.health_check_timeout
                    )
                    
                    execution_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        self._health_status = True
                        logger.debug(f"Health check passed for {self.config.name} in {execution_time:.2f}s")
                    else:
                        self._health_status = False
                        logger.warning(f"Health check failed for {self.config.name}: HTTP {response.status_code}")
                    
                except Exception as e:
                    self._health_status = False
                    logger.error(f"Health check error for {self.config.name}: {e}")
                
                time.sleep(self.http_config.health_check_interval)
        
        self._health_check_task = threading.Thread(target=health_check_loop, daemon=True)
        self._health_check_task.start()
        
        logger.info(f"Started health monitoring for {self.config.name}")
    
    def _stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._health_check_task:
            self._health_check_task = None
            logger.info(f"Stopped health monitoring for {self.config.name}")
    
    def get_http_metrics(self) -> Dict[str, Any]:
        """Get HTTP-specific metrics."""
        base_metrics = self.get_metrics()
        http_metrics = self.http_metrics.get_metrics()
        
        combined_metrics = {
            **base_metrics,
            **http_metrics,
            'health_status': self._health_status,
            'service_name': self.http_config.service_name,
            'base_url': self.http_config.base_url
        }
        
        return combined_metrics
    
    def is_healthy(self) -> bool:
        """Check if the HTTP service is healthy."""
        return self._health_status and self.is_available()
    
    def cleanup(self):
        """Cleanup HTTP circuit breaker resources."""
        # Close sessions
        if self._session:
            self._session.close()
        
        if self._async_session:
            asyncio.create_task(self._async_session.close())
        
        # Stop health monitoring
        self._stop_health_monitoring()
        self._stop_health_check()
        
        logger.info(f"HTTP circuit breaker '{self.config.name}' cleaned up")


# Convenience functions for HTTP circuit breaker setup

def create_http_circuit_breaker(name: str, **kwargs) -> HTTPCircuitBreaker:
    """
    Create and register an HTTP circuit breaker.
    
    Args:
        name: Circuit breaker name
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured HTTPCircuitBreaker instance
    """
    config = HTTPCircuitBreakerConfig(name=name, **kwargs)
    circuit_breaker = HTTPCircuitBreaker(config)
    register_circuit_breaker(circuit_breaker)
    return circuit_breaker


def http_circuit_breaker(name: str, **kwargs):
    """
    Decorator for HTTP operations with circuit breaker protection.
    
    Usage:
        ml_api_cb = http_circuit_breaker(
            name="ml_api", 
            base_url="http://localhost:8001",
            health_check_url="http://localhost:8001/health"
        )
        
        @ml_api_cb
        def call_prediction_api(data):
            return requests.post("/predict", json=data)
    """
    circuit_breaker = create_http_circuit_breaker(name, **kwargs)
    return circuit_breaker


# Pre-configured circuit breakers for common NightScan services

def create_ml_api_circuit_breaker(base_url: str = "http://localhost:8001") -> HTTPCircuitBreaker:
    """Create circuit breaker for ML prediction API."""
    return create_http_circuit_breaker(
        name="ml_prediction_api",
        service_name="ML Prediction Service",
        base_url=base_url,
        health_check_url=f"{base_url}/health",
        connect_timeout=2.0,
        read_timeout=30.0,  # ML predictions can take time
        total_timeout=60.0,
        expected_status_codes=[200, 202],  # 202 for async predictions
        validate_response_json=True,
        failure_threshold=3,
        timeout=120  # 2 minutes before trying again
    )


def create_notification_api_circuit_breaker() -> HTTPCircuitBreaker:
    """Create circuit breaker for notification services."""
    return create_http_circuit_breaker(
        name="notification_service",
        service_name="Notification Service", 
        connect_timeout=1.0,
        read_timeout=5.0,
        total_timeout=10.0,
        failure_threshold=5,  # More tolerant for notifications
        timeout=60,  # 1 minute before trying again
        max_retries=2  # Fewer retries for notifications
    )


def create_analytics_api_circuit_breaker(base_url: str = "http://localhost:8008") -> HTTPCircuitBreaker:
    """Create circuit breaker for analytics service."""
    return create_http_circuit_breaker(
        name="analytics_service",
        service_name="Analytics Service",
        base_url=base_url,
        health_check_url=f"{base_url}/health",
        connect_timeout=1.0,
        read_timeout=10.0,
        total_timeout=15.0,
        failure_threshold=3,
        timeout=90  # 1.5 minutes before trying again
    )