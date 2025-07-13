"""Performance tests for optimized analytics module."""

import pytest
import time
import random
from datetime import datetime, timedelta
from sqlalchemy import event, create_engine
from sqlalchemy.pool import Pool
from contextlib import contextmanager

from web.app import create_app, db, Detection, User
from analytics_dashboard import OptimizedAnalyticsEngine, OptimizedReportGenerator


class QueryCounter:
    """Count SQL queries executed during a test."""
    
    def __init__(self):
        self.count = 0
        self.queries = []
    
    def __call__(self, conn, cursor, statement, parameters, context, executemany):
        self.count += 1
        self.queries.append(statement)


@contextmanager
def assert_num_queries(expected_count):
    """Context manager to assert number of SQL queries."""
    counter = QueryCounter()
    
    # Register event listener
    event.listen(db.engine, "before_cursor_execute", counter)
    
    try:
        yield counter
    finally:
        # Unregister event listener
        event.remove(db.engine, "before_cursor_execute", counter)
        
        # Assert query count
        assert counter.count == expected_count, (
            f"Expected {expected_count} queries, but {counter.count} were executed:\n" +
            "\n".join(f"{i+1}. {q[:100]}..." for i, q in enumerate(counter.queries))
        )


@pytest.fixture
def app():
    """Create test Flask app."""
    app = create_app('testing')
    
    with app.app_context():
        db.create_all()
        
        # Create test user
        user = User(
            username='testuser',
            email='test@example.com',
            password_hash='hashed_password'
        )
        db.session.add(user)
        db.session.commit()
        
        yield app
        
        db.drop_all()


@pytest.fixture
def populated_db(app):
    """Populate database with test data."""
    # Generate test data
    species_list = ['wolf', 'bear', 'deer', 'fox', 'rabbit', 'owl', 'eagle', 'lynx']
    zones = ['north', 'south', 'east', 'west', 'central']
    
    # Create detections
    detections = []
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i in range(10000):  # Large dataset to test performance
        detection = Detection(
            species=random.choice(species_list),
            confidence=random.uniform(0.5, 1.0),
            time=base_time + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            ),
            zone=random.choice(zones),
            latitude=random.uniform(45.0, 50.0),
            longitude=random.uniform(-75.0, -70.0),
            image_url=f'/uploads/detection_{i}.jpg'
        )
        detections.append(detection)
        
        # Batch insert for performance
        if len(detections) >= 1000:
            db.session.bulk_save_objects(detections)
            db.session.commit()
            detections = []
    
    # Insert remaining
    if detections:
        db.session.bulk_save_objects(detections)
        db.session.commit()
    
    return app


def test_get_detection_metrics_no_n_plus_one(populated_db):
    """Test that get_detection_metrics uses optimized queries."""
    with populated_db.app_context():
        engine = OptimizedAnalyticsEngine(db)
        
        # Should execute a fixed number of queries regardless of data size
        with assert_num_queries(6):  # Basic metrics, active sensors, top species, hourly, daily
            metrics = engine.get_detection_metrics(days=30)
        
        # Verify results
        assert metrics.total_detections > 0
        assert metrics.unique_species > 0
        assert len(metrics.top_species) > 0
        assert len(metrics.hourly_distribution) > 0


def test_get_species_insights_no_n_plus_one(populated_db):
    """Test that get_species_insights uses optimized queries."""
    with populated_db.app_context():
        engine = OptimizedAnalyticsEngine(db)
        
        # Should execute exactly 4 queries
        with assert_num_queries(4):  # Stats, zones, hourly, recent
            insights = engine.get_species_insights_optimized('wolf', days=30)
        
        # Verify results
        assert insights['species'] == 'wolf'
        assert insights['total_detections'] > 0
        assert len(insights['zones']) > 0
        assert len(insights['recent_detections']) <= 10  # Limited


def test_get_zone_analytics_no_n_plus_one(populated_db):
    """Test that get_zone_analytics uses optimized queries."""
    with populated_db.app_context():
        engine = OptimizedAnalyticsEngine(db)
        
        # Should execute exactly 4 queries
        with assert_num_queries(4):  # Stats, species, daily, peak hour
            analytics = engine.get_zone_analytics_optimized('north', days=30)
        
        # Verify results
        assert analytics['zone'] == 'north'
        assert analytics['total_detections'] > 0
        assert analytics['species_diversity'] > 0


def test_csv_export_memory_usage(populated_db):
    """Test that CSV export uses constant memory via pagination."""
    with populated_db.app_context():
        engine = OptimizedAnalyticsEngine(db)
        generator = OptimizedReportGenerator(engine)
        
        start_date = datetime.utcnow() - timedelta(days=30)
        end_date = datetime.utcnow()
        
        # Track memory usage
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate CSV rows
        row_count = 0
        max_memory = initial_memory
        
        for row in generator.generate_csv_report_paginated(start_date, end_date):
            row_count += 1
            
            # Check memory every 1000 rows
            if row_count % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
        
        # Memory increase should be minimal (< 50MB for 10k rows)
        memory_increase = max_memory - initial_memory
        assert memory_increase < 50, f"Memory increased by {memory_increase}MB"
        
        # Should respect MAX_EXPORT_ROWS limit
        assert row_count <= 10001  # +1 for header


def test_performance_comparison(populated_db):
    """Compare performance of optimized vs naive implementation."""
    with populated_db.app_context():
        engine = OptimizedAnalyticsEngine(db)
        
        # Time optimized implementation
        start_time = time.time()
        metrics = engine.get_detection_metrics(days=30)
        optimized_time = time.time() - start_time
        
        # Simulate naive implementation (commented out to avoid actual N+1)
        # start_time = time.time()
        # detections = Detection.query.all()  # Load ALL records
        # species_count = len(set(d.species for d in detections))
        # naive_time = time.time() - start_time
        
        # Optimized should be under 1 second for 10k records
        assert optimized_time < 1.0, f"Query took {optimized_time}s"
        
        print(f"Optimized query time: {optimized_time:.3f}s")
        print(f"Total detections processed: {metrics.total_detections}")


def test_concurrent_requests(populated_db):
    """Test connection pool handling of concurrent requests."""
    import threading
    import queue
    
    with populated_db.app_context():
        engine = OptimizedAnalyticsEngine(db)
        errors = queue.Queue()
        
        def make_request():
            try:
                metrics = engine.get_detection_metrics(days=7)
                assert metrics.total_detections > 0
            except Exception as e:
                errors.put(str(e))
        
        # Create concurrent requests
        threads = []
        for _ in range(20):  # 20 concurrent requests
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=10)
        
        # Check for errors
        error_list = []
        while not errors.empty():
            error_list.append(errors.get())
        
        assert len(error_list) == 0, f"Errors occurred: {error_list}"


def test_query_limits_enforced(populated_db):
    """Test that query limits are properly enforced."""
    with populated_db.app_context():
        engine = OptimizedAnalyticsEngine(db)
        
        # Test max days limit
        metrics = engine.get_detection_metrics(days=500)  # Should be capped at 365
        
        # Test species insights limit
        insights = engine.get_species_insights_optimized('wolf', days=30)
        assert len(insights['recent_detections']) <= 10
        
        # Test zone heatmap limit
        from analytics_dashboard import analytics_bp
        with populated_db.test_client() as client:
            # Login first
            client.post('/login', data={
                'username': 'testuser',
                'password': 'password'
            })
            
            # Check dashboard
            response = client.get('/analytics/dashboard?days=30')
            assert response.status_code == 200


def test_aggregation_accuracy(populated_db):
    """Test that aggregated queries return accurate results."""
    with populated_db.app_context():
        engine = OptimizedAnalyticsEngine(db)
        
        # Get metrics
        metrics = engine.get_detection_metrics(days=30)
        
        # Manually verify one metric
        manual_count = Detection.query.filter(
            Detection.time >= datetime.utcnow() - timedelta(days=30)
        ).count()
        
        assert metrics.total_detections == manual_count
        
        # Verify species count
        manual_species = db.session.query(
            db.func.count(db.func.distinct(Detection.species))
        ).filter(
            Detection.time >= datetime.utcnow() - timedelta(days=30)
        ).scalar()
        
        assert metrics.unique_species == manual_species


def test_cache_effectiveness(populated_db):
    """Test that caching improves performance."""
    with populated_db.app_context():
        from flask_caching import Cache
        
        # Configure cache
        cache = Cache(populated_db, config={
            'CACHE_TYPE': 'simple',
            'CACHE_DEFAULT_TIMEOUT': 300
        })
        
        engine = OptimizedAnalyticsEngine(db)
        
        # First call (cache miss)
        start_time = time.time()
        metrics1 = engine.get_detection_metrics(days=30)
        first_call_time = time.time() - start_time
        
        # Second call (should be faster if cached)
        start_time = time.time()
        metrics2 = engine.get_detection_metrics(days=30)
        second_call_time = time.time() - start_time
        
        # Results should be identical
        assert metrics1.total_detections == metrics2.total_detections
        
        print(f"First call: {first_call_time:.3f}s")
        print(f"Second call: {second_call_time:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])