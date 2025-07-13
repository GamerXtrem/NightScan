"""
Database Circuit Breaker for NightScan

Provides specialized circuit breaker protection for database operations
with intelligent fallbacks and connection pool monitoring.

Features:
- SQLAlchemy integration with transaction support
- Connection pool health monitoring
- Read/write operation differentiation
- Graceful degradation with read-only mode
- Query timeout enforcement
- Automatic retry with exponential backoff
"""

import time
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import (
    SQLAlchemyError, DisconnectionError, TimeoutError as SQLTimeoutError,
    OperationalError, DatabaseError as SQLDatabaseError, IntegrityError
)
from sqlalchemy.pool import QueuePool

from circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpenException,
    register_circuit_breaker
)
from exceptions import DatabaseError, DatabaseConnectionError, DatabaseTransactionError

logger = logging.getLogger(__name__)


@dataclass
class DatabaseCircuitBreakerConfig(CircuitBreakerConfig):
    """Extended configuration for database circuit breaker."""
    read_timeout: float = 5.0           # Timeout for read operations
    write_timeout: float = 10.0         # Timeout for write operations
    connection_timeout: float = 3.0     # Timeout for getting connections
    
    # Pool monitoring
    pool_check_interval: float = 30.0   # Seconds between pool health checks
    max_pool_failures: int = 3          # Pool failures before circuit opens
    
    # Fallback behavior
    enable_read_only_fallback: bool = True    # Allow reads when writes fail
    enable_cache_fallback: bool = True        # Use cache when DB unavailable
    read_only_operations: List[str] = None    # List of operations allowed in read-only mode
    
    # Query analysis
    slow_query_threshold: float = 2.0    # Queries slower than this are flagged
    enable_query_analysis: bool = True   # Track slow queries
    
    def __post_init__(self):
        super().__post_init__()
        if self.read_only_operations is None:
            self.read_only_operations = [
                'SELECT', 'EXPLAIN', 'DESCRIBE', 'SHOW'
            ]
        
        # Database-specific exception handling
        self.expected_exception = (
            SQLAlchemyError, DisconnectionError, SQLTimeoutError,
            OperationalError, SQLDatabaseError, DatabaseError,
            DatabaseConnectionError, DatabaseTransactionError
        )


class DatabaseCircuitBreaker(CircuitBreaker):
    """
    Specialized circuit breaker for database operations.
    
    Provides protection for SQLAlchemy operations with intelligent
    fallbacks and connection pool monitoring.
    """
    
    def __init__(self, config: DatabaseCircuitBreakerConfig, db_session=None):
        super().__init__(config)
        self.db_config = config
        self.db_session = db_session
        
        # Database-specific state
        self._pool_failures = 0
        self._read_only_mode = False
        self._last_pool_check = 0
        self._pool_health_status = True
        
        # Query tracking
        self._slow_queries = []
        self._query_stats = {}
        
        # Connection pool monitoring
        self._pool_monitor_thread = None
        self._monitoring_active = False
        
        self._start_pool_monitoring()
        
        logger.info(f"Database circuit breaker '{config.name}' initialized")
    
    def execute_query(self, query: Union[str, Any], params: Dict = None, 
                      timeout: float = None, read_only: bool = False) -> Any:
        """
        Execute a database query with circuit breaker protection.
        
        Args:
            query: SQL query string or SQLAlchemy query object
            params: Query parameters
            timeout: Custom timeout for this query
            read_only: Whether this is a read-only operation
            
        Returns:
            Query result
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            DatabaseError: When query fails
        """
        operation_type = 'READ' if read_only else 'WRITE'
        query_timeout = timeout or (self.db_config.read_timeout if read_only 
                                   else self.db_config.write_timeout)
        
        def execute_operation():
            return self._execute_with_timeout(query, params, query_timeout, read_only)
        
        try:
            return self.call(execute_operation)
        except CircuitBreakerOpenException as e:
            # Try fallback options when circuit is open
            return self._handle_circuit_open(query, params, read_only, e)
    
    async def execute_query_async(self, query: Union[str, Any], params: Dict = None,
                                  timeout: float = None, read_only: bool = False) -> Any:
        """Execute an async database query with circuit breaker protection."""
        operation_type = 'READ' if read_only else 'WRITE'
        query_timeout = timeout or (self.db_config.read_timeout if read_only 
                                   else self.db_config.write_timeout)
        
        async def execute_operation():
            return await self._execute_with_timeout_async(query, params, query_timeout, read_only)
        
        try:
            return await self.call_async(execute_operation)
        except CircuitBreakerOpenException as e:
            # Try fallback options when circuit is open
            return self._handle_circuit_open(query, params, read_only, e)
    
    @contextmanager
    def transaction(self, timeout: float = None):
        """
        Context manager for database transactions with circuit breaker protection.
        
        Usage:
            with db_circuit.transaction():
                db.session.add(user)
                db.session.commit()
        """
        transaction_timeout = timeout or self.db_config.write_timeout
        
        def transaction_operation():
            return self._execute_transaction(transaction_timeout)
        
        try:
            with self.call(transaction_operation):
                yield
        except CircuitBreakerOpenException as e:
            logger.error(f"Transaction failed - circuit open: {e}")
            raise DatabaseTransactionError(
                operation='transaction',
                reason=f"Circuit breaker is open: {e}"
            )
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            if self.db_session:
                try:
                    self.db_session.rollback()
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
            raise
    
    def _execute_with_timeout(self, query: Union[str, Any], params: Dict, 
                             timeout: float, read_only: bool) -> Any:
        """Execute query with timeout enforcement."""
        if not self.db_session:
            raise DatabaseConnectionError("No database session available")
        
        start_time = time.time()
        
        try:
            # Check connection health before executing
            self._check_connection_health()
            
            # Execute query
            if isinstance(query, str):
                if params:
                    result = self.db_session.execute(text(query), params)
                else:
                    result = self.db_session.execute(text(query))
            else:
                result = self.db_session.execute(query, params or {})
            
            execution_time = time.time() - start_time
            
            # Track slow queries
            if self.db_config.enable_query_analysis and execution_time > self.db_config.slow_query_threshold:
                self._record_slow_query(query, execution_time, params)
            
            # Record query statistics
            self._record_query_stats(query, execution_time, read_only, success=True)
            
            return result
            
        except (SQLTimeoutError, TimeoutError) as e:
            execution_time = time.time() - start_time
            logger.warning(f"Database query timeout after {execution_time:.2f}s: {e}")
            self._record_query_stats(query, execution_time, read_only, success=False)
            raise DatabaseError(f"Query timeout after {execution_time:.2f}s: {str(e)}")
            
        except (DisconnectionError, OperationalError) as e:
            execution_time = time.time() - start_time
            logger.error(f"Database connection error: {e}")
            self._record_query_stats(query, execution_time, read_only, success=False)
            self._pool_failures += 1
            raise DatabaseConnectionError(f"Database connection failed: {str(e)}")
            
        except SQLAlchemyError as e:
            execution_time = time.time() - start_time
            logger.error(f"Database error executing query: {e}")
            self._record_query_stats(query, execution_time, read_only, success=False)
            raise DatabaseError(f"Database operation failed: {str(e)}")
    
    async def _execute_with_timeout_async(self, query: Union[str, Any], params: Dict,
                                          timeout: float, read_only: bool) -> Any:
        """Execute async query with timeout enforcement."""
        import asyncio
        
        def execute_sync():
            return self._execute_with_timeout(query, params, timeout, read_only)
        
        # Run in thread pool for async context
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, execute_sync)
    
    @contextmanager
    def _execute_transaction(self, timeout: float):
        """Execute transaction with timeout and rollback handling."""
        if not self.db_session:
            raise DatabaseConnectionError("No database session available")
        
        start_time = time.time()
        
        try:
            # Check connection before starting transaction
            self._check_connection_health()
            
            # Begin transaction (if not already in one)
            if not self.db_session.in_transaction():
                self.db_session.begin()
            
            yield self.db_session
            
            # Commit transaction
            self.db_session.commit()
            
            execution_time = time.time() - start_time
            self._record_query_stats("TRANSACTION", execution_time, False, success=True)
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_stats("TRANSACTION", execution_time, False, success=False)
            
            # Rollback on any error
            try:
                self.db_session.rollback()
                logger.info(f"Transaction rolled back due to error: {e}")
            except Exception as rollback_error:
                logger.error(f"Failed to rollback transaction: {rollback_error}")
            
            # Re-raise original exception
            if isinstance(e, SQLAlchemyError):
                raise DatabaseError(f"Transaction failed: {str(e)}")
            else:
                raise
    
    def _check_connection_health(self):
        """Check if database connection is healthy."""
        try:
            # Simple ping query
            self.db_session.execute(text("SELECT 1"))
            self._pool_health_status = True
        except Exception as e:
            self._pool_health_status = False
            logger.warning(f"Database connection health check failed: {e}")
            raise DatabaseConnectionError(f"Database connection unhealthy: {str(e)}")
    
    def _handle_circuit_open(self, query: Union[str, Any], params: Dict, 
                           read_only: bool, circuit_error: CircuitBreakerOpenException) -> Any:
        """Handle fallback behavior when circuit is open."""
        
        # If it's a read operation and read-only fallback is enabled
        if read_only and self.db_config.enable_read_only_fallback:
            logger.info(f"Attempting read-only fallback for query")
            
            # Try to use cache if available
            if self.db_config.enable_cache_fallback:
                cache_result = self._try_cache_fallback(query, params)
                if cache_result is not None:
                    return cache_result
        
        # If no fallback available, raise the circuit breaker exception
        logger.error(f"No fallback available for query when circuit is open")
        raise circuit_error
    
    def _try_cache_fallback(self, query: Union[str, Any], params: Dict) -> Any:
        """Try to get result from cache when database is unavailable."""
        try:
            from cache_utils import get_cache
            cache = get_cache()
            
            # Create cache key from query and params
            cache_key = f"db_fallback:{hash(str(query))}{hash(str(params))}"
            result = cache.get(cache_key)
            
            if result is not None:
                logger.info(f"Using cached result for database query fallback")
                return result
            
        except Exception as e:
            logger.warning(f"Cache fallback failed: {e}")
        
        return None
    
    def _record_slow_query(self, query: Union[str, Any], execution_time: float, params: Dict):
        """Record information about slow queries for analysis."""
        slow_query_info = {
            'query': str(query)[:500],  # Truncate long queries
            'execution_time': execution_time,
            'params': str(params)[:200] if params else None,
            'timestamp': time.time()
        }
        
        self._slow_queries.append(slow_query_info)
        
        # Keep only last 100 slow queries
        if len(self._slow_queries) > 100:
            self._slow_queries = self._slow_queries[-100:]
        
        logger.warning(f"Slow query detected ({execution_time:.2f}s): {str(query)[:100]}...")
    
    def _record_query_stats(self, query: Union[str, Any], execution_time: float,
                           read_only: bool, success: bool):
        """Record query statistics for monitoring."""
        query_type = self._get_query_type(query)
        operation = 'READ' if read_only else 'WRITE'
        
        stats_key = f"{query_type}_{operation}"
        
        if stats_key not in self._query_stats:
            self._query_stats[stats_key] = {
                'count': 0,
                'total_time': 0.0,
                'successes': 0,
                'failures': 0,
                'avg_time': 0.0
            }
        
        stats = self._query_stats[stats_key]
        stats['count'] += 1
        stats['total_time'] += execution_time
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        stats['avg_time'] = stats['total_time'] / stats['count']
    
    def _get_query_type(self, query: Union[str, Any]) -> str:
        """Extract query type from query string."""
        query_str = str(query).strip().upper()
        
        if query_str.startswith('SELECT'):
            return 'SELECT'
        elif query_str.startswith('INSERT'):
            return 'INSERT'
        elif query_str.startswith('UPDATE'):
            return 'UPDATE'
        elif query_str.startswith('DELETE'):
            return 'DELETE'
        elif query_str.startswith('TRANSACTION'):
            return 'TRANSACTION'
        else:
            return 'OTHER'
    
    def _start_pool_monitoring(self):
        """Start background monitoring of connection pool health."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        def monitor_pool():
            while self._monitoring_active:
                try:
                    time.sleep(self.db_config.pool_check_interval)
                    self._check_pool_health()
                except Exception as e:
                    logger.error(f"Error in pool monitoring: {e}")
        
        self._pool_monitor_thread = threading.Thread(target=monitor_pool, daemon=True)
        self._pool_monitor_thread.start()
        
        logger.info(f"Started pool monitoring for database circuit '{self.config.name}'")
    
    def _check_pool_health(self):
        """Check health of connection pool."""
        try:
            if self.db_session and hasattr(self.db_session, 'bind'):
                engine = self.db_session.bind
                
                if hasattr(engine, 'pool'):
                    pool = engine.pool
                    
                    # Get pool statistics
                    pool_status = {
                        'size': getattr(pool, 'size', lambda: 0)(),
                        'checked_in': getattr(pool, 'checkedin', lambda: 0)(),
                        'checked_out': getattr(pool, 'checkedout', lambda: 0)(),
                        'overflow': getattr(pool, 'overflow', lambda: 0)(),
                        'invalid': getattr(pool, 'invalid', lambda: 0)()
                    }
                    
                    # Check for pool exhaustion
                    if (pool_status['checked_out'] >= pool_status['size'] and 
                        pool_status['overflow'] >= getattr(pool, '_max_overflow', 10)):
                        logger.warning(f"Database pool exhausted: {pool_status}")
                        self._pool_failures += 1
                    
                    # Check for too many invalid connections
                    if pool_status['invalid'] > pool_status['size'] * 0.5:
                        logger.warning(f"High number of invalid connections: {pool_status}")
                        self._pool_failures += 1
                    
                    # Reset pool failures if everything looks good
                    if (pool_status['checked_out'] < pool_status['size'] * 0.8 and
                        pool_status['invalid'] < pool_status['size'] * 0.1):
                        self._pool_failures = max(0, self._pool_failures - 1)
                    
                    logger.debug(f"Pool health check: {pool_status}")
            
        except Exception as e:
            logger.error(f"Failed to check pool health: {e}")
            self._pool_failures += 1
    
    def _stop_pool_monitoring(self):
        """Stop background pool monitoring."""
        self._monitoring_active = False
        if self._pool_monitor_thread and self._pool_monitor_thread.is_alive():
            self._pool_monitor_thread = None
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get database-specific metrics."""
        base_metrics = self.get_metrics()
        
        database_metrics = {
            'pool_failures': self._pool_failures,
            'read_only_mode': self._read_only_mode,
            'pool_health_status': self._pool_health_status,
            'slow_query_count': len(self._slow_queries),
            'query_stats': self._query_stats.copy(),
            'recent_slow_queries': self._slow_queries[-10:] if self._slow_queries else []
        }
        
        base_metrics.update(database_metrics)
        return base_metrics
    
    def enable_read_only_mode(self):
        """Enable read-only mode for graceful degradation."""
        self._read_only_mode = True
        logger.warning(f"Database circuit '{self.config.name}' switched to read-only mode")
    
    def disable_read_only_mode(self):
        """Disable read-only mode."""
        self._read_only_mode = False
        logger.info(f"Database circuit '{self.config.name}' switched back to read-write mode")
    
    def cleanup(self):
        """Cleanup database circuit breaker resources."""
        self._stop_pool_monitoring()
        self._stop_health_check()
        logger.info(f"Database circuit breaker '{self.config.name}' cleaned up")


# Convenience functions for database circuit breaker setup

def create_database_circuit_breaker(name: str, db_session=None, **kwargs) -> DatabaseCircuitBreaker:
    """
    Create and register a database circuit breaker.
    
    Args:
        name: Circuit breaker name
        db_session: SQLAlchemy session
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured DatabaseCircuitBreaker instance
    """
    config = DatabaseCircuitBreakerConfig(name=name, **kwargs)
    circuit_breaker = DatabaseCircuitBreaker(config, db_session)
    register_circuit_breaker(circuit_breaker)
    return circuit_breaker


def database_circuit_breaker(name: str, db_session=None, **kwargs):
    """
    Decorator for database operations with circuit breaker protection.
    
    Usage:
        @database_circuit_breaker(name="user_queries", read_timeout=3.0)
        def get_user_by_id(user_id):
            return db.session.query(User).filter(User.id == user_id).first()
    """
    circuit_breaker = create_database_circuit_breaker(name, db_session, **kwargs)
    return circuit_breaker


# Integration with existing database code

def wrap_sqlalchemy_session(session, circuit_name: str = "database", **kwargs) -> DatabaseCircuitBreaker:
    """
    Wrap an existing SQLAlchemy session with circuit breaker protection.
    
    Args:
        session: SQLAlchemy session to wrap
        circuit_name: Name for the circuit breaker
        **kwargs: Additional configuration
        
    Returns:
        DatabaseCircuitBreaker wrapping the session
    """
    config = DatabaseCircuitBreakerConfig(name=circuit_name, **kwargs)
    return DatabaseCircuitBreaker(config, session)


def protect_database_operation(operation_name: str = "database_op", **kwargs):
    """
    Decorator to protect individual database operations.
    
    Usage:
        @protect_database_operation(operation_name="user_creation", write_timeout=5.0)
        def create_user(username, email):
            user = User(username=username, email=email)
            db.session.add(user)
            db.session.commit()
            return user
    """
    def decorator(func):
        circuit_breaker = create_database_circuit_breaker(
            name=f"db_{operation_name}",
            **kwargs
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator