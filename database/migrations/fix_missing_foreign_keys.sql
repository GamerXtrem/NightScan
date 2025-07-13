-- NightScan Database Migration: Fix Missing Foreign Keys and Data Integrity
-- This migration adds missing foreign key constraints, check constraints, and indexes
-- to ensure data integrity and improve performance.

-- Start transaction
BEGIN;

-- =====================================================
-- 1. ADD MISSING FOREIGN KEY CONSTRAINTS
-- =====================================================

-- PredictionArchive.user_id -> User.id
-- Note: Cannot use CASCADE DELETE here as User might be deleted but we want to keep archives
ALTER TABLE prediction_archive 
ADD CONSTRAINT fk_prediction_archive_user 
FOREIGN KEY (user_id) REFERENCES "user"(id) ON DELETE SET NULL;

-- SubscriptionEvent.old_plan_type -> PlanFeatures.plan_type
ALTER TABLE subscription_events 
ADD CONSTRAINT fk_subscription_event_old_plan 
FOREIGN KEY (old_plan_type) REFERENCES plan_features(plan_type) ON DELETE SET NULL;

-- SubscriptionEvent.new_plan_type -> PlanFeatures.plan_type
ALTER TABLE subscription_events 
ADD CONSTRAINT fk_subscription_event_new_plan 
FOREIGN KEY (new_plan_type) REFERENCES plan_features(plan_type) ON DELETE SET NULL;

-- DataRetentionLog.plan_type -> PlanFeatures.plan_type
ALTER TABLE data_retention_log 
ADD CONSTRAINT fk_data_retention_log_plan 
FOREIGN KEY (plan_type) REFERENCES plan_features(plan_type) ON DELETE SET NULL;

-- =====================================================
-- 2. ADD CHECK CONSTRAINTS FOR DATA VALIDATION
-- =====================================================

-- QuotaUsage constraints
ALTER TABLE quota_usage 
ADD CONSTRAINT chk_quota_usage_month CHECK (month >= 1 AND month <= 12);

ALTER TABLE quota_usage 
ADD CONSTRAINT chk_quota_usage_year CHECK (year >= 2024 AND year <= 2100);

-- DailyUsageDetails constraints
ALTER TABLE daily_usage_details 
ADD CONSTRAINT chk_daily_usage_peak_hour CHECK (peak_hour IS NULL OR (peak_hour >= 0 AND peak_hour <= 23));

-- Detection constraints
ALTER TABLE detection 
ADD CONSTRAINT chk_detection_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0);

-- Add latitude/longitude constraints
ALTER TABLE detection 
ADD CONSTRAINT chk_detection_latitude CHECK (latitude IS NULL OR (latitude >= -90 AND latitude <= 90));

ALTER TABLE detection 
ADD CONSTRAINT chk_detection_longitude CHECK (longitude IS NULL OR (longitude >= -180 AND longitude <= 180));

-- UserPlan constraints
ALTER TABLE user_plans 
ADD CONSTRAINT chk_user_plan_dates CHECK (
    subscription_end IS NULL OR subscription_end > subscription_start
);

ALTER TABLE user_plans 
ADD CONSTRAINT chk_user_plan_trial CHECK (
    (is_trial = false) OR (is_trial = true AND trial_end IS NOT NULL)
);

-- PlanFeatures constraints
ALTER TABLE plan_features 
ADD CONSTRAINT chk_plan_features_quota CHECK (monthly_quota >= 0);

ALTER TABLE plan_features 
ADD CONSTRAINT chk_plan_features_file_size CHECK (max_file_size_mb > 0);

ALTER TABLE plan_features 
ADD CONSTRAINT chk_plan_features_retention CHECK (data_retention_days > 0);

ALTER TABLE plan_features 
ADD CONSTRAINT chk_plan_features_price CHECK (price_monthly_cents >= 0);

-- Prediction constraints
ALTER TABLE prediction 
ADD CONSTRAINT chk_prediction_file_size CHECK (file_size >= 0);

-- PredictionArchive constraints
ALTER TABLE prediction_archive 
ADD CONSTRAINT chk_prediction_archive_file_size CHECK (file_size >= 0);

-- QuotaTransaction constraints
ALTER TABLE quota_transactions 
ADD CONSTRAINT chk_quota_transaction_type CHECK (
    transaction_type IN ('usage', 'bonus', 'reset', 'adjustment')
);

-- SubscriptionEvent constraints
ALTER TABLE subscription_events 
ADD CONSTRAINT chk_subscription_event_type CHECK (
    event_type IN ('created', 'upgraded', 'downgraded', 'cancelled', 'renewed', 'expired', 'reactivated')
);

-- UserPlan status constraint
ALTER TABLE user_plans 
ADD CONSTRAINT chk_user_plan_status CHECK (
    status IN ('active', 'cancelled', 'suspended', 'expired', 'pending')
);

-- NotificationPreference constraints
ALTER TABLE notification_preference 
ADD CONSTRAINT chk_notification_min_priority CHECK (
    min_priority IN ('low', 'normal', 'high', 'critical')
);

-- Add quiet hours format check
ALTER TABLE notification_preference 
ADD CONSTRAINT chk_notification_quiet_hours CHECK (
    (quiet_hours_start IS NULL AND quiet_hours_end IS NULL) OR 
    (quiet_hours_start IS NOT NULL AND quiet_hours_end IS NOT NULL AND 
     quiet_hours_start ~ '^[0-2][0-9]:[0-5][0-9]$' AND 
     quiet_hours_end ~ '^[0-2][0-9]:[0-5][0-9]$')
);

-- =====================================================
-- 3. CREATE MISSING INDEXES FOR PERFORMANCE
-- =====================================================

-- User table indexes
CREATE INDEX IF NOT EXISTS idx_user_username ON "user"(username);

-- Prediction table indexes
CREATE INDEX IF NOT EXISTS idx_prediction_created_at ON prediction(created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_user_created ON prediction(user_id, created_at DESC);

-- Detection table indexes
CREATE INDEX IF NOT EXISTS idx_detection_species_time ON detection(species, time);
CREATE INDEX IF NOT EXISTS idx_detection_user_time ON detection(user_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_detection_confidence ON detection(confidence);

-- Geospatial index for location queries
CREATE INDEX IF NOT EXISTS idx_detection_location ON detection(latitude, longitude) 
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- PredictionArchive indexes
CREATE INDEX IF NOT EXISTS idx_prediction_archive_original_id ON prediction_archive(original_prediction_id);
CREATE INDEX IF NOT EXISTS idx_prediction_archive_user_id ON prediction_archive(user_id);
CREATE INDEX IF NOT EXISTS idx_prediction_archive_archived_at ON prediction_archive(archived_at);

-- QuotaUsage indexes
CREATE INDEX IF NOT EXISTS idx_quota_usage_user_date ON quota_usage(user_id, year DESC, month DESC);
CREATE INDEX IF NOT EXISTS idx_quota_usage_reset_date ON quota_usage(reset_date);

-- DailyUsageDetails indexes
CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date ON daily_usage_details(user_id, usage_date DESC);

-- SubscriptionEvent indexes
CREATE INDEX IF NOT EXISTS idx_subscription_event_user_date ON subscription_events(user_id, effective_date DESC);
CREATE INDEX IF NOT EXISTS idx_subscription_event_type ON subscription_events(event_type);

-- UserPlan indexes
CREATE INDEX IF NOT EXISTS idx_user_plan_status ON user_plans(status);
CREATE INDEX IF NOT EXISTS idx_user_plan_subscription_end ON user_plans(subscription_end) 
WHERE subscription_end IS NOT NULL;

-- DataRetentionLog indexes
CREATE INDEX IF NOT EXISTS idx_retention_log_user_date ON data_retention_log(user_id, deletion_date DESC);

-- NotificationPreference indexes
CREATE INDEX IF NOT EXISTS idx_notification_pref_user ON notification_preference(user_id);

-- QuotaTransaction indexes
CREATE INDEX IF NOT EXISTS idx_quota_transaction_user_created ON quota_transactions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_quota_transaction_type ON quota_transactions(transaction_type);

-- =====================================================
-- 4. ADD COMMENTS FOR DOCUMENTATION
-- =====================================================

COMMENT ON CONSTRAINT fk_prediction_archive_user ON prediction_archive IS 
'Links archived predictions to users. SET NULL on user deletion to preserve archive history.';

COMMENT ON CONSTRAINT chk_detection_confidence ON detection IS 
'Ensures detection confidence is between 0.0 and 1.0';

COMMENT ON CONSTRAINT chk_user_plan_dates ON user_plans IS 
'Ensures subscription end date is after start date';

COMMENT ON INDEX idx_detection_location ON detection IS 
'Spatial index for location-based queries';

-- =====================================================
-- 5. CREATE INTEGRITY VALIDATION FUNCTION
-- =====================================================

CREATE OR REPLACE FUNCTION validate_data_integrity() RETURNS TABLE (
    issue_type TEXT,
    table_name TEXT,
    record_count BIGINT,
    details TEXT
) AS $$
BEGIN
    -- Check for orphaned predictions
    RETURN QUERY
    SELECT 'orphaned_record'::TEXT, 
           'prediction'::TEXT,
           COUNT(*)::BIGINT,
           'Predictions with non-existent user_id'::TEXT
    FROM prediction p
    WHERE NOT EXISTS (SELECT 1 FROM "user" u WHERE u.id = p.user_id);

    -- Check for orphaned prediction archives
    RETURN QUERY
    SELECT 'orphaned_record'::TEXT,
           'prediction_archive'::TEXT,
           COUNT(*)::BIGINT,
           'Archived predictions with non-existent user_id'::TEXT
    FROM prediction_archive pa
    WHERE pa.user_id IS NOT NULL 
      AND NOT EXISTS (SELECT 1 FROM "user" u WHERE u.id = pa.user_id);

    -- Check for invalid plan references
    RETURN QUERY
    SELECT 'invalid_reference'::TEXT,
           'user_plans'::TEXT,
           COUNT(*)::BIGINT,
           'User plans with non-existent plan_type'::TEXT
    FROM user_plans up
    WHERE NOT EXISTS (SELECT 1 FROM plan_features pf WHERE pf.plan_type = up.plan_type);

    -- Check for invalid subscription events
    RETURN QUERY
    SELECT 'invalid_reference'::TEXT,
           'subscription_events'::TEXT,
           COUNT(*)::BIGINT,
           'Subscription events with invalid plan types'::TEXT
    FROM subscription_events se
    WHERE (se.old_plan_type IS NOT NULL 
           AND NOT EXISTS (SELECT 1 FROM plan_features pf WHERE pf.plan_type = se.old_plan_type))
       OR (se.new_plan_type IS NOT NULL 
           AND NOT EXISTS (SELECT 1 FROM plan_features pf WHERE pf.plan_type = se.new_plan_type));

    -- Check for data constraint violations
    RETURN QUERY
    SELECT 'constraint_violation'::TEXT,
           'quota_usage'::TEXT,
           COUNT(*)::BIGINT,
           'Invalid month values'::TEXT
    FROM quota_usage
    WHERE month < 1 OR month > 12;

    RETURN QUERY
    SELECT 'constraint_violation'::TEXT,
           'detection'::TEXT,
           COUNT(*)::BIGINT,
           'Invalid confidence values'::TEXT
    FROM detection
    WHERE confidence < 0.0 OR confidence > 1.0;

    -- Check for duplicate user plans
    RETURN QUERY
    SELECT 'duplicate_record'::TEXT,
           'user_plans'::TEXT,
           COUNT(*)::BIGINT,
           'Users with multiple active plans'::TEXT
    FROM (
        SELECT user_id, COUNT(*) as plan_count
        FROM user_plans
        WHERE status = 'active'
        GROUP BY user_id
        HAVING COUNT(*) > 1
    ) duplicates;
END;
$$ LANGUAGE plpgsql;

-- Run integrity check
SELECT * FROM validate_data_integrity();

-- Commit transaction
COMMIT;

-- =====================================================
-- ROLLBACK SCRIPT (save separately)
-- =====================================================
/*
-- To rollback this migration, run:

BEGIN;

-- Drop indexes
DROP INDEX IF EXISTS idx_user_username;
DROP INDEX IF EXISTS idx_prediction_created_at;
DROP INDEX IF EXISTS idx_prediction_user_created;
DROP INDEX IF EXISTS idx_detection_species_time;
DROP INDEX IF EXISTS idx_detection_user_time;
DROP INDEX IF EXISTS idx_detection_confidence;
DROP INDEX IF EXISTS idx_detection_location;
DROP INDEX IF EXISTS idx_prediction_archive_original_id;
DROP INDEX IF EXISTS idx_prediction_archive_user_id;
DROP INDEX IF EXISTS idx_prediction_archive_archived_at;
DROP INDEX IF EXISTS idx_quota_usage_user_date;
DROP INDEX IF EXISTS idx_quota_usage_reset_date;
DROP INDEX IF EXISTS idx_daily_usage_user_date;
DROP INDEX IF EXISTS idx_subscription_event_user_date;
DROP INDEX IF EXISTS idx_subscription_event_type;
DROP INDEX IF EXISTS idx_user_plan_status;
DROP INDEX IF EXISTS idx_user_plan_subscription_end;
DROP INDEX IF EXISTS idx_retention_log_user_date;
DROP INDEX IF EXISTS idx_notification_pref_user;
DROP INDEX IF EXISTS idx_quota_transaction_user_created;
DROP INDEX IF EXISTS idx_quota_transaction_type;

-- Drop check constraints
ALTER TABLE quota_usage DROP CONSTRAINT IF EXISTS chk_quota_usage_month;
ALTER TABLE quota_usage DROP CONSTRAINT IF EXISTS chk_quota_usage_year;
ALTER TABLE daily_usage_details DROP CONSTRAINT IF EXISTS chk_daily_usage_peak_hour;
ALTER TABLE detection DROP CONSTRAINT IF EXISTS chk_detection_confidence;
ALTER TABLE detection DROP CONSTRAINT IF EXISTS chk_detection_latitude;
ALTER TABLE detection DROP CONSTRAINT IF EXISTS chk_detection_longitude;
ALTER TABLE user_plans DROP CONSTRAINT IF EXISTS chk_user_plan_dates;
ALTER TABLE user_plans DROP CONSTRAINT IF EXISTS chk_user_plan_trial;
ALTER TABLE user_plans DROP CONSTRAINT IF EXISTS chk_user_plan_status;
ALTER TABLE plan_features DROP CONSTRAINT IF EXISTS chk_plan_features_quota;
ALTER TABLE plan_features DROP CONSTRAINT IF EXISTS chk_plan_features_file_size;
ALTER TABLE plan_features DROP CONSTRAINT IF EXISTS chk_plan_features_retention;
ALTER TABLE plan_features DROP CONSTRAINT IF EXISTS chk_plan_features_price;
ALTER TABLE prediction DROP CONSTRAINT IF EXISTS chk_prediction_file_size;
ALTER TABLE prediction_archive DROP CONSTRAINT IF EXISTS chk_prediction_archive_file_size;
ALTER TABLE quota_transactions DROP CONSTRAINT IF EXISTS chk_quota_transaction_type;
ALTER TABLE subscription_events DROP CONSTRAINT IF EXISTS chk_subscription_event_type;
ALTER TABLE notification_preference DROP CONSTRAINT IF EXISTS chk_notification_min_priority;
ALTER TABLE notification_preference DROP CONSTRAINT IF EXISTS chk_notification_quiet_hours;

-- Drop foreign key constraints
ALTER TABLE prediction_archive DROP CONSTRAINT IF EXISTS fk_prediction_archive_user;
ALTER TABLE subscription_events DROP CONSTRAINT IF EXISTS fk_subscription_event_old_plan;
ALTER TABLE subscription_events DROP CONSTRAINT IF EXISTS fk_subscription_event_new_plan;
ALTER TABLE data_retention_log DROP CONSTRAINT IF EXISTS fk_data_retention_log_plan;

-- Drop validation function
DROP FUNCTION IF EXISTS validate_data_integrity();

COMMIT;
*/