-- =====================================================
-- NightScan Database Schema - Complete Structure
-- =====================================================
-- This script creates the complete database structure with all
-- constraints, indexes, and initial data for NightScan.
-- No migrations needed - this is the final schema.

-- Drop existing tables (in correct order due to foreign keys)
DROP TABLE IF EXISTS prediction_archive CASCADE;
DROP TABLE IF EXISTS data_retention_log CASCADE;
DROP TABLE IF EXISTS subscription_events CASCADE;
DROP TABLE IF EXISTS quota_transactions CASCADE;
DROP TABLE IF EXISTS daily_usage_details CASCADE;
DROP TABLE IF EXISTS quota_usage CASCADE;
DROP TABLE IF EXISTS user_plans CASCADE;
DROP TABLE IF EXISTS notification_preference CASCADE;
DROP TABLE IF EXISTS detection CASCADE;
DROP TABLE IF EXISTS prediction CASCADE;
DROP TABLE IF EXISTS plan_features CASCADE;
DROP TABLE IF EXISTS "user" CASCADE;

-- =====================================================
-- CORE TABLES
-- =====================================================

-- User table
CREATE TABLE "user" (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password_hash VARCHAR(128) NOT NULL,
    CONSTRAINT chk_username_length CHECK (LENGTH(username) >= 3)
);

-- Plan features configuration
CREATE TABLE plan_features (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_type VARCHAR(50) NOT NULL UNIQUE,
    plan_name VARCHAR(100) NOT NULL,
    monthly_quota INTEGER NOT NULL DEFAULT 100,
    max_file_size_mb INTEGER NOT NULL DEFAULT 50,
    max_concurrent_uploads INTEGER NOT NULL DEFAULT 1,
    priority_queue BOOLEAN NOT NULL DEFAULT FALSE,
    advanced_analytics BOOLEAN NOT NULL DEFAULT FALSE,
    api_access BOOLEAN NOT NULL DEFAULT FALSE,
    email_support BOOLEAN NOT NULL DEFAULT FALSE,
    phone_support BOOLEAN NOT NULL DEFAULT FALSE,
    features_json TEXT, -- JSON string for additional features
    price_monthly_cents INTEGER NOT NULL DEFAULT 0,
    price_yearly_cents INTEGER,
    data_retention_days INTEGER NOT NULL DEFAULT 30,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_monthly_quota CHECK (monthly_quota >= 0),
    CONSTRAINT chk_max_file_size CHECK (max_file_size_mb > 0),
    CONSTRAINT chk_retention_days CHECK (data_retention_days > 0),
    CONSTRAINT chk_price CHECK (price_monthly_cents >= 0),
    CONSTRAINT chk_yearly_price CHECK (price_yearly_cents IS NULL OR price_yearly_cents >= 0)
);

-- =====================================================
-- PREDICTION TABLES
-- =====================================================

-- Prediction table
CREATE TABLE prediction (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    filename VARCHAR(200),
    result TEXT,
    file_size INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    CONSTRAINT fk_prediction_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT chk_file_size CHECK (file_size >= 0)
);

-- Prediction archive (soft delete)
CREATE TABLE prediction_archive (
    id SERIAL PRIMARY KEY,
    original_prediction_id INTEGER NOT NULL,
    user_id INTEGER, -- Nullable to preserve archives when user is deleted
    filename VARCHAR(200),
    result TEXT,
    file_size INTEGER,
    created_at TIMESTAMP NOT NULL,
    archived_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    plan_type_at_archive VARCHAR(50),
    retention_days INTEGER,
    archived_by VARCHAR(50) DEFAULT 'system',
    
    -- Foreign keys
    CONSTRAINT fk_archive_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE SET NULL,
    
    -- Constraints
    CONSTRAINT chk_archive_file_size CHECK (file_size >= 0)
);

-- =====================================================
-- DETECTION TABLE
-- =====================================================

CREATE TABLE detection (
    id SERIAL PRIMARY KEY,
    species VARCHAR(100) NOT NULL,
    time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    latitude FLOAT,
    longitude FLOAT,
    zone VARCHAR(100),
    image_url VARCHAR(200),
    confidence FLOAT DEFAULT 0.0,
    user_id INTEGER,
    description TEXT,
    
    -- Foreign keys
    CONSTRAINT fk_detection_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE SET NULL,
    
    -- Constraints
    CONSTRAINT chk_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT chk_latitude CHECK (latitude IS NULL OR (latitude >= -90 AND latitude <= 90)),
    CONSTRAINT chk_longitude CHECK (longitude IS NULL OR (longitude >= -180 AND longitude <= 180))
);

-- =====================================================
-- NOTIFICATION PREFERENCES
-- =====================================================

CREATE TABLE notification_preference (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE,
    email_notifications BOOLEAN DEFAULT TRUE,
    push_notifications BOOLEAN DEFAULT TRUE,
    email_address VARCHAR(200),
    min_priority VARCHAR(20) DEFAULT 'normal',
    species_filter TEXT, -- JSON string
    zone_filter TEXT,    -- JSON string
    quiet_hours_start VARCHAR(5), -- HH:MM
    quiet_hours_end VARCHAR(5),   -- HH:MM
    slack_webhook VARCHAR(500),
    discord_webhook VARCHAR(500),
    
    -- Foreign keys
    CONSTRAINT fk_notification_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT chk_min_priority CHECK (min_priority IN ('low', 'normal', 'high', 'critical')),
    CONSTRAINT chk_quiet_hours CHECK (
        (quiet_hours_start IS NULL AND quiet_hours_end IS NULL) OR 
        (quiet_hours_start IS NOT NULL AND quiet_hours_end IS NOT NULL AND 
         quiet_hours_start ~ '^[0-2][0-9]:[0-5][0-9]$' AND 
         quiet_hours_end ~ '^[0-2][0-9]:[0-5][0-9]$')
    )
);

-- =====================================================
-- SUBSCRIPTION AND QUOTA TABLES
-- =====================================================

-- User subscription plans
CREATE TABLE user_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL UNIQUE,
    plan_type VARCHAR(50) NOT NULL,
    subscription_start TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    subscription_end TIMESTAMP,
    auto_renew BOOLEAN NOT NULL DEFAULT FALSE,
    payment_method VARCHAR(50),
    subscription_id VARCHAR(200), -- For payment provider
    trial_end TIMESTAMP,
    is_trial BOOLEAN NOT NULL DEFAULT FALSE,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    CONSTRAINT fk_user_plan_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE CASCADE,
    CONSTRAINT fk_user_plan_type FOREIGN KEY (plan_type) 
        REFERENCES plan_features(plan_type) ON DELETE RESTRICT,
    
    -- Constraints
    CONSTRAINT chk_subscription_dates CHECK (
        subscription_end IS NULL OR subscription_end > subscription_start
    ),
    CONSTRAINT chk_trial_dates CHECK (
        (is_trial = FALSE) OR (is_trial = TRUE AND trial_end IS NOT NULL)
    ),
    CONSTRAINT chk_plan_status CHECK (
        status IN ('active', 'cancelled', 'suspended', 'expired', 'pending')
    )
);

-- Monthly quota usage tracking
CREATE TABLE quota_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    month INTEGER NOT NULL,
    year INTEGER NOT NULL,
    prediction_count INTEGER NOT NULL DEFAULT 0,
    total_file_size_bytes BIGINT NOT NULL DEFAULT 0,
    successful_predictions INTEGER NOT NULL DEFAULT 0,
    failed_predictions INTEGER NOT NULL DEFAULT 0,
    premium_features_used TEXT, -- JSON string
    reset_date TIMESTAMP NOT NULL,
    last_prediction_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    CONSTRAINT fk_quota_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT chk_month CHECK (month >= 1 AND month <= 12),
    CONSTRAINT chk_year CHECK (year >= 2024 AND year <= 2100),
    CONSTRAINT chk_counts CHECK (
        prediction_count >= 0 AND 
        successful_predictions >= 0 AND 
        failed_predictions >= 0
    ),
    
    -- Unique constraint
    CONSTRAINT unique_user_month_year UNIQUE (user_id, month, year)
);

-- Daily usage details
CREATE TABLE daily_usage_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    usage_date DATE NOT NULL DEFAULT CURRENT_DATE,
    prediction_count INTEGER NOT NULL DEFAULT 0,
    total_file_size_bytes BIGINT NOT NULL DEFAULT 0,
    average_processing_time_ms INTEGER,
    peak_hour INTEGER,
    device_type VARCHAR(50),
    app_version VARCHAR(20),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    CONSTRAINT fk_daily_usage_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT chk_peak_hour CHECK (peak_hour IS NULL OR (peak_hour >= 0 AND peak_hour <= 23)),
    CONSTRAINT chk_daily_counts CHECK (prediction_count >= 0),
    
    -- Unique constraint
    CONSTRAINT unique_user_date UNIQUE (user_id, usage_date)
);

-- Quota transactions audit trail
CREATE TABLE quota_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    amount INTEGER NOT NULL,
    reason VARCHAR(200),
    metadata TEXT, -- JSON string
    prediction_id INTEGER,
    admin_user_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    CONSTRAINT fk_quota_trans_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE CASCADE,
    CONSTRAINT fk_quota_trans_admin FOREIGN KEY (admin_user_id) 
        REFERENCES "user"(id) ON DELETE SET NULL,
    CONSTRAINT fk_quota_trans_prediction FOREIGN KEY (prediction_id) 
        REFERENCES prediction(id) ON DELETE SET NULL,
    
    -- Constraints
    CONSTRAINT chk_transaction_type CHECK (
        transaction_type IN ('usage', 'bonus', 'reset', 'adjustment')
    )
);

-- Subscription lifecycle events
CREATE TABLE subscription_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    old_plan_type VARCHAR(50),
    new_plan_type VARCHAR(50),
    effective_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT, -- JSON string
    created_by INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    CONSTRAINT fk_sub_event_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE CASCADE,
    CONSTRAINT fk_sub_event_created_by FOREIGN KEY (created_by) 
        REFERENCES "user"(id) ON DELETE SET NULL,
    CONSTRAINT fk_sub_event_old_plan FOREIGN KEY (old_plan_type) 
        REFERENCES plan_features(plan_type) ON DELETE SET NULL,
    CONSTRAINT fk_sub_event_new_plan FOREIGN KEY (new_plan_type) 
        REFERENCES plan_features(plan_type) ON DELETE SET NULL,
    
    -- Constraints
    CONSTRAINT chk_event_type CHECK (
        event_type IN ('created', 'upgraded', 'downgraded', 'cancelled', 'renewed', 'expired', 'reactivated')
    )
);

-- =====================================================
-- DATA RETENTION LOG
-- =====================================================

CREATE TABLE data_retention_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    plan_type VARCHAR(50),
    retention_days INTEGER NOT NULL,
    records_deleted INTEGER NOT NULL DEFAULT 0,
    total_size_deleted_bytes BIGINT NOT NULL DEFAULT 0,
    deletion_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    retention_policy_version VARCHAR(50) DEFAULT '1.0',
    admin_override BOOLEAN DEFAULT FALSE,
    admin_reason TEXT,
    metadata TEXT, -- JSON string
    
    -- Foreign keys
    CONSTRAINT fk_retention_user FOREIGN KEY (user_id) 
        REFERENCES "user"(id) ON DELETE CASCADE,
    CONSTRAINT fk_retention_plan FOREIGN KEY (plan_type) 
        REFERENCES plan_features(plan_type) ON DELETE SET NULL,
    
    -- Constraints
    CONSTRAINT chk_retention_counts CHECK (records_deleted >= 0),
    CONSTRAINT chk_retention_size CHECK (total_size_deleted_bytes >= 0)
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- User indexes
CREATE INDEX idx_user_username ON "user"(username);

-- Prediction indexes
CREATE INDEX idx_prediction_created_at ON prediction(created_at);
CREATE INDEX idx_prediction_user_created ON prediction(user_id, created_at DESC);

-- Detection indexes
CREATE INDEX idx_detection_species_time ON detection(species, time);
CREATE INDEX idx_detection_user_time ON detection(user_id, time DESC);
CREATE INDEX idx_detection_confidence ON detection(confidence);
CREATE INDEX idx_detection_location ON detection(latitude, longitude) 
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Archive indexes
CREATE INDEX idx_archive_original_id ON prediction_archive(original_prediction_id);
CREATE INDEX idx_archive_user_id ON prediction_archive(user_id);
CREATE INDEX idx_archive_archived_at ON prediction_archive(archived_at);

-- Quota indexes
CREATE INDEX idx_quota_usage_user_date ON quota_usage(user_id, year DESC, month DESC);
CREATE INDEX idx_quota_usage_reset_date ON quota_usage(reset_date);
CREATE INDEX idx_daily_usage_user_date ON daily_usage_details(user_id, usage_date DESC);

-- Subscription indexes
CREATE INDEX idx_user_plan_status ON user_plans(status);
CREATE INDEX idx_user_plan_subscription_end ON user_plans(subscription_end) 
    WHERE subscription_end IS NOT NULL;
CREATE INDEX idx_sub_event_user_date ON subscription_events(user_id, effective_date DESC);
CREATE INDEX idx_sub_event_type ON subscription_events(event_type);

-- Transaction indexes
CREATE INDEX idx_quota_trans_user_created ON quota_transactions(user_id, created_at DESC);
CREATE INDEX idx_quota_trans_type ON quota_transactions(transaction_type);

-- Retention log indexes
CREATE INDEX idx_retention_log_user_date ON data_retention_log(user_id, deletion_date DESC);

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Insert default plan types
INSERT INTO plan_features (
    plan_type, plan_name, monthly_quota, max_file_size_mb, 
    max_concurrent_uploads, data_retention_days, price_monthly_cents
) VALUES 
    ('free', 'Free Plan', 10, 10, 1, 7, 0),
    ('basic', 'Basic Plan', 100, 50, 2, 30, 999),
    ('pro', 'Professional Plan', 500, 100, 5, 90, 2999),
    ('enterprise', 'Enterprise Plan', 2000, 200, 10, 365, 9999);

-- Update features for plans
UPDATE plan_features SET 
    priority_queue = TRUE,
    email_support = TRUE
WHERE plan_type IN ('basic', 'pro', 'enterprise');

UPDATE plan_features SET 
    advanced_analytics = TRUE,
    api_access = TRUE
WHERE plan_type IN ('pro', 'enterprise');

UPDATE plan_features SET 
    phone_support = TRUE,
    features_json = '{"custom_models": true, "dedicated_support": true, "sla": "99.9%"}'
WHERE plan_type = 'enterprise';

-- =====================================================
-- HELPER FUNCTIONS
-- =====================================================

-- Function to get user's current plan
CREATE OR REPLACE FUNCTION get_user_plan(p_user_id INTEGER) 
RETURNS TABLE (
    plan_type VARCHAR(50),
    plan_name VARCHAR(100),
    monthly_quota INTEGER,
    data_retention_days INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pf.plan_type,
        pf.plan_name,
        pf.monthly_quota,
        pf.data_retention_days
    FROM user_plans up
    JOIN plan_features pf ON up.plan_type = pf.plan_type
    WHERE up.user_id = p_user_id 
      AND up.status = 'active'
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to check quota usage
CREATE OR REPLACE FUNCTION check_user_quota(p_user_id INTEGER) 
RETURNS TABLE (
    used INTEGER,
    limit_quota INTEGER,
    remaining INTEGER
) AS $$
DECLARE
    v_month INTEGER;
    v_year INTEGER;
    v_used INTEGER;
    v_limit INTEGER;
BEGIN
    -- Get current month/year
    v_month := EXTRACT(MONTH FROM CURRENT_DATE);
    v_year := EXTRACT(YEAR FROM CURRENT_DATE);
    
    -- Get usage
    SELECT COALESCE(prediction_count, 0) INTO v_used
    FROM quota_usage
    WHERE user_id = p_user_id 
      AND month = v_month 
      AND year = v_year;
    
    -- Get limit from plan
    SELECT COALESCE(pf.monthly_quota, 0) INTO v_limit
    FROM user_plans up
    JOIN plan_features pf ON up.plan_type = pf.plan_type
    WHERE up.user_id = p_user_id 
      AND up.status = 'active';
    
    -- Return results
    RETURN QUERY
    SELECT 
        COALESCE(v_used, 0) as used,
        COALESCE(v_limit, 10) as limit_quota,
        GREATEST(COALESCE(v_limit, 10) - COALESCE(v_used, 0), 0) as remaining;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- TRIGGERS
-- =====================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to relevant tables
CREATE TRIGGER update_plan_features_updated_at BEFORE UPDATE ON plan_features
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_plans_updated_at BEFORE UPDATE ON user_plans
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_quota_usage_updated_at BEFORE UPDATE ON quota_usage
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- COMMENTS FOR DOCUMENTATION
-- =====================================================

COMMENT ON TABLE "user" IS 'Core user authentication table';
COMMENT ON TABLE plan_features IS 'Available subscription plans and their features';
COMMENT ON TABLE prediction IS 'Audio file predictions and analysis results';
COMMENT ON TABLE prediction_archive IS 'Soft-deleted predictions for data retention compliance';
COMMENT ON TABLE detection IS 'Wildlife detection events with location and confidence';
COMMENT ON TABLE user_plans IS 'User subscription assignments and status';
COMMENT ON TABLE quota_usage IS 'Monthly quota tracking per user';
COMMENT ON TABLE subscription_events IS 'Audit log of all subscription changes';

COMMENT ON CONSTRAINT fk_archive_user ON prediction_archive IS 
    'Allows preserving archives when user is deleted by setting user_id to NULL';
COMMENT ON CONSTRAINT chk_confidence ON detection IS 
    'Ensures detection confidence is a valid probability between 0 and 1';

-- =====================================================
-- VERIFICATION
-- =====================================================

-- List all tables with row counts (run after creation)
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;