"""
Circuit Breaker Implementation for NightScan

Provides fault tolerance and resilience patterns to prevent cascade failures
and enable graceful degradation when external services are unavailable.

Key Features:
- Three states: CLOSED, OPEN, HALF_OPEN
- Configurable failure thresholds and timeouts
- Exponential backoff and retry logic
- Metrics collection for monitoring
- Support for sync and async operations
- Graceful degradation and fallback mechanisms
"""

import time
import threading
import asyncio
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states following the standard pattern."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    name: str
    failure_threshold: int = 5          # Number of failures before opening
    success_threshold: int = 3          # Number of successes to close from half-open
    timeout: float = 60.0               # Seconds before trying half-open
    expected_exception: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
    fallback: Optional[Callable] = None # Fallback function when circuit is open
    
    # Advanced configuration
    rolling_window_size: int = 100      # Size of rolling window for failure tracking
    minimum_requests: int = 10          # Minimum requests before considering failures
    slow_call_duration: float = 5.0    # Calls slower than this count as failures
    slow_call_rate_threshold: float = 0.5  # % of slow calls that trigger circuit
    
    # Retry configuration
    max_retries: int = 3
    base_delay: float = 1.0            # Base delay for exponential backoff
    max_delay: float = 60.0            # Maximum delay between retries
    
    # Monitoring
    enable_metrics: bool = True
    health_check_interval: float = 30.0  # Seconds between health checks in OPEN state


@dataclass
class CircuitBreakerMetrics:
    """Metrics collected by the circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    slow_requests: int = 0
    rejected_requests: int = 0        # Requests rejected when circuit is OPEN
    
    state_transitions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    
    # Performance metrics
    average_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Rolling window for recent performance
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=100))


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is in OPEN state."""
    
    def __init__(self, circuit_name: str, last_failure: str = None):
        self.circuit_name = circuit_name
        self.last_failure = last_failure
        message = f"Circuit breaker '{circuit_name}' is OPEN"
        if last_failure:
            message += f" (last failure: {last_failure})"
        super().__init__(message)


class CircuitBreakerTimeoutException(Exception):
    """Exception raised when operation times out in circuit breaker."""
    pass


class CircuitBreaker:
    """
    Implementation of the Circuit Breaker pattern for fault tolerance.
    
    The circuit breaker monitors the failure rate of operations and
    can automatically stop calling a failing service, allowing it time to recover.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        
        # State management
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._last_success_time = None
        self._next_attempt_time = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Health checking
        self._health_check_task = None
        self._health_check_running = False
        
        logger.info(f"Circuit breaker '{config.name}' initialized with {config}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function execution
            
        Raises:
            CircuitBreakerOpenException: When circuit is OPEN
            Original exceptions: When function fails and circuit allows
        """
        with self._lock:
            self._check_and_update_state()
            
            if self.state == CircuitState.OPEN:
                self.metrics.rejected_requests += 1
                if self.config.fallback:
                    logger.info(f"Circuit '{self.config.name}' is OPEN, using fallback")
                    return self.config.fallback(*args, **kwargs)
                else:
                    raise CircuitBreakerOpenException(
                        self.config.name, 
                        str(self.metrics.last_failure_time)
                    )
        
        # Execute with monitoring
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            # Apply timeout if configured
            if hasattr(self.config, 'call_timeout') and self.config.call_timeout:
                result = self._execute_with_timeout(func, self.config.call_timeout, *args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except self.config.expected_exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
        except Exception as e:
            # Unexpected exception - still record as failure but re-raise
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            logger.warning(f"Unexpected exception in circuit '{self.config.name}': {e}")
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function execution
            
        Raises:
            CircuitBreakerOpenException: When circuit is OPEN
            Original exceptions: When function fails and circuit allows
        """
        with self._lock:
            self._check_and_update_state()
            
            if self.state == CircuitState.OPEN:
                self.metrics.rejected_requests += 1
                if self.config.fallback:
                    logger.info(f"Circuit '{self.config.name}' is OPEN, using fallback")
                    if asyncio.iscoroutinefunction(self.config.fallback):
                        return await self.config.fallback(*args, **kwargs)
                    else:
                        return self.config.fallback(*args, **kwargs)
                else:
                    raise CircuitBreakerOpenException(
                        self.config.name,
                        str(self.metrics.last_failure_time)
                    )
        
        # Execute with monitoring
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            # Apply timeout if configured
            if hasattr(self.config, 'call_timeout') and self.config.call_timeout:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.call_timeout)
            else:
                result = await func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except self.config.expected_exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            raise
        except asyncio.TimeoutError as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            self.metrics.timeout_requests += 1
            raise CircuitBreakerTimeoutException(f"Operation timed out in circuit '{self.config.name}'")
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            logger.warning(f"Unexpected exception in async circuit '{self.config.name}': {e}")
            raise
    
    def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with timeout (for sync operations)."""
        import signal
        
        def timeout_handler(signum, frame):
            raise CircuitBreakerTimeoutException(f"Operation timed out after {timeout}s")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _record_success(self, execution_time: float):
        """Record a successful operation."""
        with self._lock:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now()
            self.metrics.recent_response_times.append(execution_time)
            self.metrics.recent_failures.append(False)
            
            # Check for slow calls
            if execution_time > self.config.slow_call_duration:
                self.metrics.slow_requests += 1
            
            # Update response time metrics
            self._update_response_time_metrics()
            
            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    def _record_failure(self, exception: Exception, execution_time: float):
        """Record a failed operation."""
        with self._lock:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()
            self.metrics.recent_response_times.append(execution_time)
            self.metrics.recent_failures.append(True)
            
            # Update response time metrics
            self._update_response_time_metrics()
            
            # State transitions
            if self.state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._should_open_circuit():
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                # Immediate transition back to OPEN on any failure
                self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on current metrics."""
        # Check failure threshold
        if self._failure_count >= self.config.failure_threshold:
            return True
        
        # Check if we have minimum requests to make a decision
        if len(self.metrics.recent_failures) < self.config.minimum_requests:
            return False
        
        # Check failure rate in rolling window
        recent_failures = sum(self.metrics.recent_failures)
        failure_rate = recent_failures / len(self.metrics.recent_failures)
        
        if failure_rate >= 0.5:  # 50% failure rate
            return True
        
        # Check slow call rate
        if self.metrics.recent_response_times:
            slow_calls = sum(1 for t in self.metrics.recent_response_times 
                           if t > self.config.slow_call_duration)
            slow_call_rate = slow_calls / len(self.metrics.recent_response_times)
            
            if slow_call_rate >= self.config.slow_call_rate_threshold:
                return True
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        logger.warning(f"Circuit breaker '{self.config.name}' transitioning to OPEN state")
        self.state = CircuitState.OPEN
        self.metrics.last_state_change = datetime.now()
        self.metrics.state_transitions[f"to_{CircuitState.OPEN.value}"] += 1
        self._next_attempt_time = time.time() + self.config.timeout
        self._failure_count = 0
        self._success_count = 0
        
        # Start health checking if enabled
        if self.config.health_check_interval > 0:
            self._start_health_check()
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        logger.info(f"Circuit breaker '{self.config.name}' transitioning to HALF_OPEN state")
        self.state = CircuitState.HALF_OPEN
        self.metrics.last_state_change = datetime.now()
        self.metrics.state_transitions[f"to_{CircuitState.HALF_OPEN.value}"] += 1
        self._success_count = 0
        self._next_attempt_time = None
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        logger.info(f"Circuit breaker '{self.config.name}' transitioning to CLOSED state")
        self.state = CircuitState.CLOSED
        self.metrics.last_state_change = datetime.now()
        self.metrics.state_transitions[f"to_{CircuitState.CLOSED.value}"] += 1
        self._failure_count = 0
        self._success_count = 0
        self._next_attempt_time = None
        
        # Stop health checking
        self._stop_health_check()
    
    def _check_and_update_state(self):
        """Check if state should be updated based on current conditions."""
        if self.state == CircuitState.OPEN and self._next_attempt_time:
            if time.time() >= self._next_attempt_time:
                self._transition_to_half_open()
    
    def _update_response_time_metrics(self):
        """Update response time percentile metrics."""
        if not self.metrics.recent_response_times:
            return
        
        times = sorted(self.metrics.recent_response_times)
        self.metrics.average_response_time = sum(times) / len(times)
        
        if len(times) >= 5:  # Need minimum data for percentiles
            p95_idx = int(len(times) * 0.95)
            p99_idx = int(len(times) * 0.99)
            self.metrics.p95_response_time = times[p95_idx]
            self.metrics.p99_response_time = times[p99_idx]
    
    def _start_health_check(self):
        """Start background health checking for OPEN circuit."""
        if self._health_check_running:
            return
        
        self._health_check_running = True
        
        def health_check_loop():
            while self._health_check_running and self.state == CircuitState.OPEN:
                try:
                    time.sleep(self.config.health_check_interval)
                    if self.state == CircuitState.OPEN and time.time() >= self._next_attempt_time:
                        logger.info(f"Health check triggered for circuit '{self.config.name}'")
                        self._transition_to_half_open()
                        break
                except Exception as e:
                    logger.error(f"Error in health check for circuit '{self.config.name}': {e}")
        
        self._health_check_task = threading.Thread(target=health_check_loop, daemon=True)
        self._health_check_task.start()
    
    def _stop_health_check(self):
        """Stop background health checking."""
        self._health_check_running = False
        if self._health_check_task and self._health_check_task.is_alive():
            self._health_check_task = None
    
    def reset(self):
        """Reset circuit breaker to CLOSED state and clear metrics."""
        with self._lock:
            logger.info(f"Resetting circuit breaker '{self.config.name}'")
            self.state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._next_attempt_time = None
            
            # Reset metrics
            self.metrics = CircuitBreakerMetrics()
            
            # Stop health checking
            self._stop_health_check()
    
    def force_open(self):
        """Force circuit breaker to OPEN state."""
        with self._lock:
            logger.warning(f"Forcing circuit breaker '{self.config.name}' to OPEN state")
            self._transition_to_open()
    
    def force_close(self):
        """Force circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Forcing circuit breaker '{self.config.name}' to CLOSED state")
            self._transition_to_closed()
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for monitoring."""
        with self._lock:
            return {
                'name': self.config.name,
                'state': self.state.value,
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'rejected_requests': self.metrics.rejected_requests,
                'timeout_requests': self.metrics.timeout_requests,
                'slow_requests': self.metrics.slow_requests,
                'failure_rate': (self.metrics.failed_requests / self.metrics.total_requests 
                               if self.metrics.total_requests > 0 else 0),
                'average_response_time': self.metrics.average_response_time,
                'p95_response_time': self.metrics.p95_response_time,
                'p99_response_time': self.metrics.p99_response_time,
                'last_failure_time': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                'last_success_time': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                'last_state_change': self.metrics.last_state_change.isoformat() if self.metrics.last_state_change else None,
                'state_transitions': dict(self.metrics.state_transitions),
                'next_attempt_time': self._next_attempt_time
            }
    
    def is_available(self) -> bool:
        """Check if circuit breaker is available for requests."""
        with self._lock:
            self._check_and_update_state()
            return self.state != CircuitState.OPEN


# Convenience functions and decorators

def circuit_breaker(name: str, **kwargs) -> Callable:
    """
    Decorator factory for creating circuit breakers.
    
    Usage:
        @circuit_breaker(name="database", failure_threshold=3, timeout=60)
        def query_database():
            return db.query("SELECT 1")
    """
    config = CircuitBreakerConfig(name=name, **kwargs)
    breaker = CircuitBreaker(config)
    return breaker


def with_circuit_breaker(breaker: CircuitBreaker) -> Callable:
    """
    Decorator for using an existing circuit breaker.
    
    Usage:
        db_breaker = CircuitBreaker(CircuitBreakerConfig(name="database"))
        
        @with_circuit_breaker(db_breaker)
        def query_database():
            return db.query("SELECT 1")
    """
    return breaker


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = threading.RLock()


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name from global registry."""
    with _registry_lock:
        return _circuit_breakers.get(name)


def register_circuit_breaker(breaker: CircuitBreaker) -> CircuitBreaker:
    """Register circuit breaker in global registry."""
    with _registry_lock:
        _circuit_breakers[breaker.config.name] = breaker
        logger.info(f"Registered circuit breaker '{breaker.config.name}'")
        return breaker


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    with _registry_lock:
        return _circuit_breakers.copy()


def get_circuit_breaker_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all registered circuit breakers."""
    with _registry_lock:
        return {name: breaker.get_metrics() 
                for name, breaker in _circuit_breakers.items()}


def reset_all_circuit_breakers():
    """Reset all registered circuit breakers."""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()
        logger.info("Reset all circuit breakers")


def cleanup_circuit_breakers():
    """Cleanup all circuit breakers (stop health checks, etc.)."""
    with _registry_lock:
        for breaker in _circuit_breakers.values():
            breaker._stop_health_check()
        _circuit_breakers.clear()
        logger.info("Cleaned up all circuit breakers")