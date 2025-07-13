# NightScan Connection Pooling Guide

## Overview

Connection pooling is critical for database performance and reliability. NightScan uses SQLAlchemy's connection pooling with optimized settings for different environments.

## Why Connection Pooling Matters

1. **Performance**: Reuses existing connections instead of creating new ones
2. **Resource Management**: Limits the number of database connections
3. **Reliability**: Handles connection failures gracefully
4. **Scalability**: Supports traffic spikes with overflow connections

## Configuration Parameters

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pool_size` | Number of persistent connections maintained | 10 |
| `max_overflow` | Maximum temporary connections above pool_size | 5 |
| `pool_timeout` | Seconds to wait for a connection before timing out | 30 |
| `pool_recycle` | Seconds before recycling a connection | 1800 (30 min) |
| `pool_pre_ping` | Test connections before use | True |
| `echo_pool` | Log pool events (debug only) | False |

### Parameter Details

#### pool_size
- The number of connections to maintain in the pool
- These connections persist and are reused
- Choose based on expected concurrent requests

#### max_overflow
- Temporary connections created when pool is exhausted
- Closed when returned to the pool
- Set to ~50% of pool_size for flexibility

#### pool_timeout
- Maximum time to wait for a connection
- Prevents indefinite blocking
- Increase for high-load environments

#### pool_recycle
- Prevents "MySQL server has gone away" errors
- Should be less than database connection timeout
- 30 minutes is a safe default

#### pool_pre_ping
- Tests connections with "SELECT 1" before use
- Slight overhead but prevents errors
- Essential for production environments

## Environment-Specific Settings

### Development
```python
{
    "pool_size": 5,
    "max_overflow": 0,  # No overflow needed
    "pool_pre_ping": False,  # Not needed for SQLite
    "pool_recycle": 3600,
    "echo_pool": True  # Debug pool events
}
```
- Small pool for local development
- Pool debugging enabled
- SQLite doesn't need pre-ping

### Testing
```python
{
    "pool_size": 1,
    "max_overflow": 0,
    "pool_pre_ping": False,
    "pool_recycle": 3600
}
```
- Minimal pool for fast tests
- In-memory database doesn't need pooling

### Staging
```python
{
    "pool_size": 15,
    "max_overflow": 10,
    "pool_pre_ping": True,
    "pool_recycle": 1800
}
```
- Moderate pool for testing production behavior
- Overflow for handling spikes
- Pre-ping enabled for stability

### Production
```python
{
    "pool_size": 50,
    "max_overflow": 25,
    "pool_pre_ping": True,
    "pool_recycle": 1800,
    "pool_timeout": 60
}
```
- Large pool for high traffic
- Significant overflow capacity
- Longer timeout for peak loads
- Aggressive recycling for stability

### VPS Lite
```python
{
    "pool_size": 20,
    "max_overflow": 10,
    "pool_pre_ping": True,
    "pool_recycle": 1800,
    "pool_timeout": 30
}
```
- Balanced for limited resources
- Still maintains good performance
- Pre-ping for VPS network reliability

### Raspberry Pi
```python
{
    "pool_size": 2,
    "max_overflow": 1,
    "pool_pre_ping": False,
    "pool_recycle": 3600
}
```
- Minimal resources usage
- SQLite local database
- No pre-ping needed

## Calculating Pool Size

### Formula
```
pool_size = (expected_concurrent_requests × average_query_time) / acceptable_wait_time
```

### Example
- 100 concurrent requests
- 50ms average query time
- 100ms acceptable wait time
- pool_size = (100 × 0.05) / 0.1 = 50 connections

### Considerations
1. **Database Limits**: Check max_connections in PostgreSQL/MySQL
2. **Memory**: Each connection uses ~1-5MB
3. **CPU**: More connections = more context switching
4. **Network**: VPS/Cloud may have connection limits

## Monitoring Connection Pool

### SQLAlchemy Pool Status
```python
from sqlalchemy.pool import Pool
from sqlalchemy import event

@event.listens_for(Pool, "connect")
def receive_connect(dbapi_conn, connection_record):
    logger.info(f"Pool connection established: {connection_record}")

@event.listens_for(Pool, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    logger.debug(f"Connection checked out from pool")

@event.listens_for(Pool, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    logger.debug(f"Connection returned to pool")
```

### Metrics to Track
1. **Pool Size**: Current connections in pool
2. **Overflow**: Temporary connections in use
3. **Wait Time**: Time to acquire connection
4. **Timeouts**: Failed connection attempts
5. **Recycles**: Connections recycled

### Database Queries
```sql
-- PostgreSQL: Current connections
SELECT count(*) FROM pg_stat_activity 
WHERE datname = 'nightscan';

-- MySQL: Current connections
SHOW STATUS WHERE Variable_name = 'Threads_connected';
```

## Troubleshooting

### "TimeoutError: QueuePool limit exceeded"
**Cause**: All connections are in use
**Solutions**:
1. Increase `pool_size`
2. Increase `max_overflow`
3. Reduce query execution time
4. Check for connection leaks

### "OperationalError: server closed the connection"
**Cause**: Connection recycled by database
**Solutions**:
1. Enable `pool_pre_ping`
2. Reduce `pool_recycle` time
3. Check database timeout settings

### High Memory Usage
**Cause**: Too many connections
**Solutions**:
1. Reduce `pool_size`
2. Reduce `max_overflow`
3. Enable connection recycling

### Slow Connection Acquisition
**Cause**: Pool exhausted frequently
**Solutions**:
1. Increase `pool_size`
2. Add `max_overflow`
3. Optimize query performance

## Best Practices

1. **Start Conservative**: Begin with smaller pools and increase as needed
2. **Monitor Metrics**: Track pool usage and adjust accordingly
3. **Test Under Load**: Use load testing to find optimal settings
4. **Environment-Specific**: Different settings for dev/staging/prod
5. **Database Limits**: Stay well below database max_connections
6. **Health Checks**: Always enable pool_pre_ping in production
7. **Logging**: Enable echo_pool in development for debugging

## Configuration Examples

### High-Traffic Web Application
```python
{
    "pool_size": 100,
    "max_overflow": 50,
    "pool_timeout": 10,
    "pool_recycle": 1800,
    "pool_pre_ping": True
}
```

### Background Job Processor
```python
{
    "pool_size": 20,
    "max_overflow": 0,  # Jobs can wait
    "pool_timeout": 300,  # Long timeout OK
    "pool_recycle": 3600,
    "pool_pre_ping": True
}
```

### Microservice
```python
{
    "pool_size": 10,
    "max_overflow": 5,
    "pool_timeout": 5,  # Fast failure
    "pool_recycle": 1800,
    "pool_pre_ping": True
}
```

## Database-Specific Considerations

### PostgreSQL
- Default max_connections: 100
- Recommended: Set pool_size to 20-30% of max_connections
- Enable `pool_pre_ping` for network reliability

### MySQL
- Default max_connections: 151
- Watch for wait_timeout (default 8 hours)
- Set pool_recycle < wait_timeout

### SQLite
- No connection limit
- pool_size can be 1 for write operations
- No need for pool_pre_ping

## Performance Impact

### Benchmarks
- Connection creation: ~20-50ms
- Pool checkout: <1ms
- Pre-ping overhead: ~1-2ms
- Memory per connection: 1-5MB

### ROI Calculation
- 1000 requests/minute
- 30ms saved per request (no connection creation)
- 30 seconds saved per minute
- 50% reduction in database load

## Migration Guide

### From Default Settings
1. Identify current connection usage
2. Calculate required pool_size
3. Add max_overflow for flexibility
4. Enable pool_pre_ping
5. Monitor and adjust

### Example Migration
```python
# Before (default)
{
    "pool_size": 10,
    "pool_timeout": 30,
    "pool_recycle": 3600
}

# After (optimized)
{
    "pool_size": 30,
    "max_overflow": 15,
    "pool_timeout": 30,
    "pool_recycle": 1800,
    "pool_pre_ping": True
}
```

## References

- [SQLAlchemy Pool Documentation](https://docs.sqlalchemy.org/en/14/core/pooling.html)
- [PostgreSQL Connection Pooling](https://www.postgresql.org/docs/current/runtime-config-connection.html)
- [MySQL Connection Limits](https://dev.mysql.com/doc/refman/8.0/en/server-system-variables.html#sysvar_max_connections)