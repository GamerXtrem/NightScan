-- Test data for integration tests
-- This script creates test users and sample data for integration testing

-- Create test users with different roles and quotas
INSERT INTO "user" (username, password_hash) VALUES 
    ('test_admin', 'pbkdf2:sha256:260000$test$5a2b6ef50a1234567890abcdef1234567890abcdef1234567890abcdef123456'),
    ('test_user', 'pbkdf2:sha256:260000$test$5a2b6ef50a1234567890abcdef1234567890abcdef1234567890abcdef123456'),
    ('test_premium', 'pbkdf2:sha256:260000$test$5a2b6ef50a1234567890abcdef1234567890abcdef1234567890abcdef123456'),
    ('test_api_user', 'pbkdf2:sha256:260000$test$5a2b6ef50a1234567890abcdef1234567890abcdef1234567890abcdef123456')
ON CONFLICT (username) DO NOTHING;

-- Insert plan features for testing
INSERT INTO plan_features (plan_type, plan_name, monthly_quota, max_file_size_mb, max_concurrent_uploads, api_access, data_retention_days) VALUES 
    ('free', 'Free Plan', 10, 25, 1, false, 30),
    ('premium', 'Premium Plan', 100, 100, 5, true, 90),
    ('enterprise', 'Enterprise Plan', 1000, 500, 10, true, 365)
ON CONFLICT (plan_type) DO UPDATE SET
    plan_name = EXCLUDED.plan_name,
    monthly_quota = EXCLUDED.monthly_quota,
    max_file_size_mb = EXCLUDED.max_file_size_mb,
    max_concurrent_uploads = EXCLUDED.max_concurrent_uploads,
    api_access = EXCLUDED.api_access,
    data_retention_days = EXCLUDED.data_retention_days;

-- Assign users to plans
INSERT INTO user_plans (user_id, plan_type, start_date, end_date, is_active) 
SELECT 
    u.id,
    CASE 
        WHEN u.username = 'test_admin' THEN 'enterprise'
        WHEN u.username = 'test_premium' THEN 'premium'
        ELSE 'free'
    END as plan_type,
    CURRENT_DATE,
    CASE 
        WHEN u.username = 'test_admin' THEN CURRENT_DATE + INTERVAL '1 year'
        WHEN u.username = 'test_premium' THEN CURRENT_DATE + INTERVAL '1 month'
        ELSE NULL
    END as end_date,
    true
FROM "user" u
WHERE u.username IN ('test_admin', 'test_user', 'test_premium', 'test_api_user')
ON CONFLICT (user_id) DO UPDATE SET
    plan_type = EXCLUDED.plan_type,
    start_date = EXCLUDED.start_date,
    end_date = EXCLUDED.end_date,
    is_active = EXCLUDED.is_active;

-- Create sample predictions for testing
INSERT INTO prediction (user_id, filename, result, file_size, created_at)
SELECT 
    u.id,
    'sample_audio_' || generate_series(1, 3) || '.wav',
    '{"species": "owl", "confidence": 0.85, "predictions": [{"class": "owl", "confidence": 0.85}, {"class": "wind", "confidence": 0.15}]}',
    44100 * 2 * generate_series(1, 3), -- 1-3 seconds of audio
    CURRENT_TIMESTAMP - (generate_series(1, 3) || ' days')::interval
FROM "user" u
WHERE u.username = 'test_user'
ON CONFLICT DO NOTHING;

-- Create sample detections for testing
INSERT INTO detection (species, time, latitude, longitude, zone, confidence, user_id, description)
SELECT 
    species_name,
    CURRENT_TIMESTAMP - (random() * 30 || ' days')::interval,
    45.5017 + (random() - 0.5) * 0.1, -- Around Montreal
    -73.5673 + (random() - 0.5) * 0.1,
    'Test Zone ' || zone_num,
    0.5 + random() * 0.5,
    (SELECT id FROM "user" WHERE username = 'test_user'),
    'Test detection for integration testing'
FROM (
    VALUES 
        ('owl', 1),
        ('fox', 1),
        ('deer', 2),
        ('rabbit', 2),
        ('wind', 3)
) AS species_data(species_name, zone_num)
ON CONFLICT DO NOTHING;

-- Initialize quota usage for test users
INSERT INTO quota_usage (user_id, quota_type, usage_amount, reset_date)
SELECT 
    u.id,
    'monthly_predictions',
    CASE 
        WHEN u.username = 'test_premium' THEN 15
        WHEN u.username = 'test_user' THEN 3
        ELSE 0
    END,
    DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'
FROM "user" u
WHERE u.username IN ('test_admin', 'test_user', 'test_premium', 'test_api_user')
ON CONFLICT (user_id, quota_type) DO UPDATE SET
    usage_amount = EXCLUDED.usage_amount,
    reset_date = EXCLUDED.reset_date;

-- Create notification preferences for test users
INSERT INTO notification_preference (user_id, notification_type, is_enabled, delivery_method)
SELECT 
    u.id,
    notification_type,
    true,
    CASE notification_type
        WHEN 'new_detection' THEN 'email'
        WHEN 'prediction_complete' THEN 'push'
        WHEN 'quota_warning' THEN 'email'
        WHEN 'system_maintenance' THEN 'email'
    END
FROM "user" u
CROSS JOIN (
    VALUES ('new_detection'), ('prediction_complete'), ('quota_warning'), ('system_maintenance')
) AS notifications(notification_type)
WHERE u.username IN ('test_user', 'test_premium')
ON CONFLICT (user_id, notification_type) DO UPDATE SET
    is_enabled = EXCLUDED.is_enabled,
    delivery_method = EXCLUDED.delivery_method;

-- Log test data creation
INSERT INTO quota_transactions (
    user_id, 
    transaction_type, 
    quota_type, 
    amount, 
    description,
    created_at
)
SELECT 
    u.id,
    'usage',
    'monthly_predictions',
    3,
    'Test data initialization',
    CURRENT_TIMESTAMP
FROM "user" u
WHERE u.username = 'test_user'
ON CONFLICT DO NOTHING;

-- Create sample daily usage details
INSERT INTO daily_usage_details (
    user_id,
    usage_date,
    quota_type,
    daily_usage,
    peak_hour_usage,
    created_at
)
SELECT 
    u.id,
    CURRENT_DATE - generate_series(0, 6),
    'monthly_predictions',
    floor(random() * 5) + 1,
    floor(random() * 3) + 1,
    CURRENT_TIMESTAMP
FROM "user" u
WHERE u.username IN ('test_user', 'test_premium')
ON CONFLICT (user_id, usage_date, quota_type) DO UPDATE SET
    daily_usage = EXCLUDED.daily_usage,
    peak_hour_usage = EXCLUDED.peak_hour_usage;

-- Ensure we have some test data for data retention
INSERT INTO prediction_archive (
    original_prediction_id,
    user_id,
    filename,
    result_summary,
    file_size,
    original_created_at,
    archived_at,
    retention_tier
)
SELECT 
    p.id,
    p.user_id,
    p.filename,
    SUBSTRING(p.result, 1, 100),
    p.file_size,
    p.created_at,
    CURRENT_TIMESTAMP,
    'cold_storage'
FROM prediction p
JOIN "user" u ON p.user_id = u.id
WHERE u.username = 'test_user'
    AND p.created_at < CURRENT_TIMESTAMP - INTERVAL '60 days'
ON CONFLICT DO NOTHING;

-- Log retention activities
INSERT INTO data_retention_log (
    table_name,
    retention_action,
    records_affected,
    retention_criteria,
    executed_at
) VALUES 
    ('prediction', 'archive', 1, 'older_than_60_days', CURRENT_TIMESTAMP),
    ('detection', 'delete', 0, 'older_than_365_days', CURRENT_TIMESTAMP - INTERVAL '1 day'),
    ('quota_transactions', 'cleanup', 5, 'older_than_90_days', CURRENT_TIMESTAMP - INTERVAL '2 days')
ON CONFLICT DO NOTHING;

-- Create some subscription events for testing
INSERT INTO subscription_events (
    user_id,
    event_type,
    plan_type,
    event_data,
    created_at
)
SELECT 
    u.id,
    'plan_upgrade',
    'premium',
    '{"from_plan": "free", "to_plan": "premium", "reason": "test_upgrade"}',
    CURRENT_TIMESTAMP - INTERVAL '1 day'
FROM "user" u
WHERE u.username = 'test_premium'
ON CONFLICT DO NOTHING;

-- Commit all test data
COMMIT;