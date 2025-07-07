-- NightScan Production Database Initialization (Secure Version)
-- Enhanced security with dynamic admin password, audit tables, and RLS

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- For password hashing
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements"; -- For query monitoring

-- Enable Row Level Security on critical tables
ALTER DATABASE nightscan SET row_security = on;

-- Create audit schema for security logging
CREATE SCHEMA IF NOT EXISTS audit;

-- Audit table for tracking all sensitive operations
CREATE TABLE IF NOT EXISTS audit.security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    user_id UUID,
    user_email VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    additional_info JSONB
);

-- Index for efficient audit queries
CREATE INDEX idx_audit_event_time ON audit.security_events(event_time DESC);
CREATE INDEX idx_audit_user_id ON audit.security_events(user_id);
CREATE INDEX idx_audit_event_type ON audit.security_events(event_type);
CREATE INDEX idx_audit_table_name ON audit.security_events(table_name);

-- Function to log security events
CREATE OR REPLACE FUNCTION audit.log_security_event(
    p_event_type VARCHAR(50),
    p_action VARCHAR(100),
    p_table_name VARCHAR(100) DEFAULT NULL,
    p_record_id UUID DEFAULT NULL,
    p_old_values JSONB DEFAULT NULL,
    p_new_values JSONB DEFAULT NULL,
    p_additional_info JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_event_id UUID;
BEGIN
    INSERT INTO audit.security_events (
        event_type, action, table_name, record_id,
        old_values, new_values, additional_info,
        user_id, user_email
    ) VALUES (
        p_event_type, p_action, p_table_name, p_record_id,
        p_old_values, p_new_values, p_additional_info,
        current_setting('app.current_user_id', true)::UUID,
        current_setting('app.current_user_email', true)
    ) RETURNING id INTO v_event_id;
    
    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Enhanced performance indexes with security considerations
CREATE OR REPLACE FUNCTION create_secure_indexes()
RETURNS void AS $$
BEGIN
    -- User table indexes with partial indexes for active users
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_email_active 
            ON "user"(email) WHERE is_active = true;
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_last_login 
            ON "user"(last_login_at DESC NULLS LAST);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_created 
            ON "user"(created_at DESC);
    END IF;
    
    -- Prediction indexes with partitioning support
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'prediction') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prediction_composite 
            ON prediction(user_id, timestamp DESC);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prediction_species_confidence 
            ON prediction(species, confidence DESC) WHERE confidence > 0.5;
    END IF;
    
    -- Detection indexes optimized for common queries
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'detection') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_composite 
            ON detection(species, time DESC);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_location 
            ON detection(location) WHERE location IS NOT NULL;
    END IF;
    
    RAISE NOTICE 'Secure indexes created successfully';
END;
$$ LANGUAGE plpgsql;

-- Function to create admin user with secure random password
CREATE OR REPLACE FUNCTION create_secure_admin_user()
RETURNS TABLE(admin_email VARCHAR, admin_password VARCHAR) AS $$
DECLARE
    v_password VARCHAR;
    v_password_hash VARCHAR;
    v_admin_id UUID;
BEGIN
    -- Generate secure random password
    v_password := encode(gen_random_bytes(16), 'base64');
    
    -- Generate bcrypt hash (cost factor 12 for production)
    -- Note: In real implementation, use application-level bcrypt
    v_password_hash := crypt(v_password, gen_salt('bf', 12));
    
    -- Insert admin user if not exists
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user') THEN
        INSERT INTO "user" (id, email, username, password_hash, is_active, is_admin, created_at)
        SELECT 
            uuid_generate_v4(),
            'admin@nightscan.local',
            'admin',
            v_password_hash,
            true,
            true,
            CURRENT_TIMESTAMP
        WHERE NOT EXISTS (SELECT 1 FROM "user" WHERE email = 'admin@nightscan.local')
        RETURNING id INTO v_admin_id;
        
        IF v_admin_id IS NOT NULL THEN
            -- Log admin creation
            PERFORM audit.log_security_event(
                'USER_CREATION',
                'ADMIN_USER_CREATED',
                'user',
                v_admin_id,
                NULL,
                jsonb_build_object('email', 'admin@nightscan.local', 'is_admin', true),
                jsonb_build_object('auto_generated', true)
            );
            
            -- Return credentials
            RETURN QUERY SELECT 'admin@nightscan.local'::VARCHAR, v_password;
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Enhanced system settings with encryption support
CREATE OR REPLACE FUNCTION create_secure_system_settings()
RETURNS void AS $$
BEGIN
    -- Create settings table with encryption support
    CREATE TABLE IF NOT EXISTS system_settings (
        key VARCHAR(100) PRIMARY KEY,
        value TEXT,
        encrypted BOOLEAN DEFAULT false,
        description TEXT,
        category VARCHAR(50) DEFAULT 'general',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_by UUID
    );
    
    -- Create audit trigger for settings changes
    CREATE OR REPLACE FUNCTION audit.trigger_settings_change()
    RETURNS TRIGGER AS $trigger$
    BEGIN
        PERFORM audit.log_security_event(
            'SETTINGS_CHANGE',
            TG_OP,
            'system_settings',
            NULL,
            to_jsonb(OLD),
            to_jsonb(NEW),
            jsonb_build_object('key', COALESCE(NEW.key, OLD.key))
        );
        RETURN NEW;
    END;
    $trigger$ LANGUAGE plpgsql;
    
    DROP TRIGGER IF EXISTS audit_settings_changes ON system_settings;
    CREATE TRIGGER audit_settings_changes
        AFTER INSERT OR UPDATE OR DELETE ON system_settings
        FOR EACH ROW EXECUTE FUNCTION audit.trigger_settings_change();
    
    -- Insert secure default settings
    INSERT INTO system_settings (key, value, description, category) VALUES
        ('system_version', '1.0.0', 'NightScan system version', 'system'),
        ('max_upload_size_mb', '50', 'Maximum file upload size in MB', 'limits'),
        ('session_timeout_minutes', '1440', 'User session timeout (24 hours)', 'security'),
        ('password_min_length', '12', 'Minimum password length', 'security'),
        ('password_require_special', 'true', 'Require special characters in password', 'security'),
        ('max_login_attempts', '5', 'Max failed login attempts before lockout', 'security'),
        ('lockout_duration_minutes', '30', 'Account lockout duration', 'security'),
        ('enable_2fa', 'false', 'Enable two-factor authentication', 'security'),
        ('audit_retention_days', '365', 'Days to retain audit logs', 'compliance'),
        ('enable_data_encryption', 'true', 'Enable at-rest data encryption', 'security')
    ON CONFLICT (key) DO NOTHING;
    
    RAISE NOTICE 'Secure system settings initialized';
END;
$$ LANGUAGE plpgsql;

-- Table partitioning for high-volume tables
CREATE OR REPLACE FUNCTION create_partitioned_tables()
RETURNS void AS $$
BEGIN
    -- Create partitioned metrics table (by month)
    CREATE TABLE IF NOT EXISTS system_metrics_partitioned (
        id BIGSERIAL,
        metric_name VARCHAR(100) NOT NULL,
        metric_value FLOAT NOT NULL,
        unit VARCHAR(20),
        tags JSONB DEFAULT '{}',
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id, timestamp)
    ) PARTITION BY RANGE (timestamp);
    
    -- Create initial partitions (current and next 3 months)
    FOR i IN 0..3 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS system_metrics_%s PARTITION OF system_metrics_partitioned
             FOR VALUES FROM (%L) TO (%L)',
            to_char(CURRENT_DATE + (i || ' months')::interval, 'YYYY_MM'),
            date_trunc('month', CURRENT_DATE + (i || ' months')::interval),
            date_trunc('month', CURRENT_DATE + ((i + 1) || ' months')::interval)
        );
    END LOOP;
    
    -- Create partitioned API usage table (by week)
    CREATE TABLE IF NOT EXISTS api_usage_partitioned (
        id BIGSERIAL,
        endpoint VARCHAR(200) NOT NULL,
        method VARCHAR(10) NOT NULL,
        status_code INTEGER NOT NULL,
        response_time_ms INTEGER,
        user_id UUID,
        ip_address INET,
        request_headers JSONB,
        response_size_bytes INTEGER,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (id, timestamp)
    ) PARTITION BY RANGE (timestamp);
    
    -- Create indexes on partitioned tables
    CREATE INDEX idx_metrics_part_timestamp ON system_metrics_partitioned(timestamp DESC);
    CREATE INDEX idx_metrics_part_name ON system_metrics_partitioned(metric_name, timestamp DESC);
    CREATE INDEX idx_api_usage_part_timestamp ON api_usage_partitioned(timestamp DESC);
    CREATE INDEX idx_api_usage_part_endpoint ON api_usage_partitioned(endpoint, timestamp DESC);
    
    RAISE NOTICE 'Partitioned tables created';
END;
$$ LANGUAGE plpgsql;

-- Row Level Security policies
CREATE OR REPLACE FUNCTION setup_row_level_security()
RETURNS void AS $$
BEGIN
    -- Enable RLS on sensitive tables (after they're created by SQLAlchemy)
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user') THEN
        ALTER TABLE "user" ENABLE ROW LEVEL SECURITY;
        
        -- Users can only see their own profile (except admins)
        CREATE POLICY user_isolation ON "user"
            FOR ALL
            USING (
                id = current_setting('app.current_user_id', true)::UUID 
                OR 
                EXISTS (
                    SELECT 1 FROM "user" u 
                    WHERE u.id = current_setting('app.current_user_id', true)::UUID 
                    AND u.is_admin = true
                )
            );
    END IF;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'prediction') THEN
        ALTER TABLE prediction ENABLE ROW LEVEL SECURITY;
        
        -- Users can only see their own predictions
        CREATE POLICY prediction_isolation ON prediction
            FOR ALL
            USING (user_id = current_setting('app.current_user_id', true)::UUID);
    END IF;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'audio_file') THEN
        ALTER TABLE audio_file ENABLE ROW LEVEL SECURITY;
        
        -- Users can only see their own audio files
        CREATE POLICY audio_file_isolation ON audio_file
            FOR ALL
            USING (user_id = current_setting('app.current_user_id', true)::UUID);
    END IF;
    
    RAISE NOTICE 'Row Level Security configured';
END;
$$ LANGUAGE plpgsql;

-- Automated maintenance functions
CREATE OR REPLACE FUNCTION create_maintenance_functions()
RETURNS void AS $$
BEGIN
    -- Function to archive old data
    CREATE OR REPLACE FUNCTION archive_old_data(p_days INTEGER DEFAULT 90)
    RETURNS void AS $func$
    DECLARE
        v_archived_count INTEGER;
    BEGIN
        -- Archive old metrics
        CREATE TABLE IF NOT EXISTS archive.system_metrics AS 
            SELECT * FROM system_metrics WHERE timestamp < NOW() - (p_days || ' days')::interval
            WITH NO DATA;
            
        INSERT INTO archive.system_metrics
        SELECT * FROM system_metrics WHERE timestamp < NOW() - (p_days || ' days')::interval;
        
        GET DIAGNOSTICS v_archived_count = ROW_COUNT;
        
        DELETE FROM system_metrics WHERE timestamp < NOW() - (p_days || ' days')::interval;
        
        PERFORM audit.log_security_event(
            'DATA_ARCHIVAL',
            'ARCHIVE_OLD_METRICS',
            'system_metrics',
            NULL,
            NULL,
            NULL,
            jsonb_build_object('archived_rows', v_archived_count, 'retention_days', p_days)
        );
        
        RAISE NOTICE 'Archived % old metric records', v_archived_count;
    END;
    $func$ LANGUAGE plpgsql SECURITY DEFINER;
    
    -- Function to clean up audit logs
    CREATE OR REPLACE FUNCTION cleanup_audit_logs(p_retention_days INTEGER DEFAULT 365)
    RETURNS void AS $func$
    DECLARE
        v_deleted_count INTEGER;
    BEGIN
        DELETE FROM audit.security_events 
        WHERE event_time < NOW() - (p_retention_days || ' days')::interval
        AND event_type NOT IN ('SECURITY_BREACH', 'ADMIN_ACTION', 'DATA_DELETION');
        
        GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
        
        RAISE NOTICE 'Cleaned up % old audit records', v_deleted_count;
    END;
    $func$ LANGUAGE plpgsql SECURITY DEFINER;
    
    RAISE NOTICE 'Maintenance functions created';
END;
$$ LANGUAGE plpgsql;

-- Enhanced initialization with security logging
CREATE OR REPLACE FUNCTION initialize_secure_nightscan_db()
RETURNS TABLE(message TEXT, detail TEXT) AS $$
DECLARE
    v_admin_creds RECORD;
    v_start_time TIMESTAMP;
    v_duration INTERVAL;
BEGIN
    v_start_time := clock_timestamp();
    
    -- Create schemas
    CREATE SCHEMA IF NOT EXISTS archive;
    
    -- Initialize all components
    PERFORM create_secure_indexes();
    PERFORM create_secure_system_settings();
    PERFORM create_partitioned_tables();
    PERFORM create_monitoring_tables();
    PERFORM create_species_config();
    PERFORM create_maintenance_functions();
    
    -- Wait for SQLAlchemy tables
    PERFORM pg_sleep(3);
    
    -- Setup RLS after tables exist
    PERFORM setup_row_level_security();
    
    -- Create admin with secure password
    FOR v_admin_creds IN SELECT * FROM create_secure_admin_user() LOOP
        RETURN QUERY SELECT 
            'ADMIN_CREATED'::TEXT, 
            format('Email: %s | Password: %s', v_admin_creds.admin_email, v_admin_creds.admin_password)::TEXT;
    END LOOP;
    
    v_duration := clock_timestamp() - v_start_time;
    
    -- Log initialization
    INSERT INTO audit.security_events (
        event_type, action, success, additional_info
    ) VALUES (
        'SYSTEM_INIT', 'DATABASE_INITIALIZED', true,
        jsonb_build_object('duration_ms', extract(milliseconds from v_duration))
    );
    
    RETURN QUERY SELECT 
        'INITIALIZATION_COMPLETE'::TEXT,
        format('Database initialized in %s', v_duration)::TEXT;
    
    RETURN QUERY SELECT
        'SECURITY_ENABLED'::TEXT,
        'Row Level Security, Audit Logging, and Partitioning active'::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Quota Management Tables
CREATE OR REPLACE FUNCTION create_quota_management_tables()
RETURNS void AS $$
BEGIN
    -- Plan Features table - defines what each plan offers
    CREATE TABLE IF NOT EXISTS plan_features (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        plan_type VARCHAR(50) NOT NULL UNIQUE,
        plan_name VARCHAR(100) NOT NULL,
        monthly_quota INTEGER NOT NULL DEFAULT 100,
        max_file_size_mb INTEGER NOT NULL DEFAULT 50,
        max_concurrent_uploads INTEGER NOT NULL DEFAULT 1,
        priority_queue BOOLEAN NOT NULL DEFAULT false,
        advanced_analytics BOOLEAN NOT NULL DEFAULT false,
        api_access BOOLEAN NOT NULL DEFAULT false,
        email_support BOOLEAN NOT NULL DEFAULT false,
        phone_support BOOLEAN NOT NULL DEFAULT false,
        features_json JSONB DEFAULT '{}',
        price_monthly_cents INTEGER NOT NULL DEFAULT 0,
        price_yearly_cents INTEGER DEFAULT NULL,
        is_active BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- User Plans table - assigns plans to users
    CREATE TABLE IF NOT EXISTS user_plans (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        plan_type VARCHAR(50) NOT NULL REFERENCES plan_features(plan_type),
        subscription_start TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        subscription_end TIMESTAMP WITH TIME ZONE,
        auto_renew BOOLEAN NOT NULL DEFAULT false,
        payment_method VARCHAR(50),
        subscription_id VARCHAR(200), -- For payment provider
        trial_end TIMESTAMP WITH TIME ZONE,
        is_trial BOOLEAN NOT NULL DEFAULT false,
        status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'cancelled', 'suspended', 'expired')),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        
        UNIQUE(user_id) -- One active plan per user
    );

    -- Quota Usage table - tracks monthly usage per user
    CREATE TABLE IF NOT EXISTS quota_usage (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        month INTEGER NOT NULL CHECK (month >= 1 AND month <= 12),
        year INTEGER NOT NULL CHECK (year >= 2024),
        prediction_count INTEGER NOT NULL DEFAULT 0,
        total_file_size_bytes BIGINT NOT NULL DEFAULT 0,
        successful_predictions INTEGER NOT NULL DEFAULT 0,
        failed_predictions INTEGER NOT NULL DEFAULT 0,
        premium_features_used JSONB DEFAULT '{}',
        reset_date TIMESTAMP WITH TIME ZONE DEFAULT date_trunc('month', CURRENT_TIMESTAMP + interval '1 month'),
        last_prediction_at TIMESTAMP WITH TIME ZONE,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        
        UNIQUE(user_id, month, year)
    );

    -- Daily Usage Details table - for detailed analytics
    CREATE TABLE IF NOT EXISTS daily_usage_details (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        usage_date DATE NOT NULL DEFAULT CURRENT_DATE,
        prediction_count INTEGER NOT NULL DEFAULT 0,
        total_file_size_bytes BIGINT NOT NULL DEFAULT 0,
        average_processing_time_ms INTEGER,
        peak_hour INTEGER, -- 0-23
        device_type VARCHAR(50),
        app_version VARCHAR(20),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        
        UNIQUE(user_id, usage_date)
    );

    -- Quota Transactions table - for audit trail of quota changes
    CREATE TABLE IF NOT EXISTS quota_transactions (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        transaction_type VARCHAR(50) NOT NULL, -- 'usage', 'bonus', 'reset', 'adjustment'
        amount INTEGER NOT NULL, -- Can be negative for usage
        reason VARCHAR(200),
        metadata JSONB DEFAULT '{}',
        prediction_id INTEGER, -- Link to specific prediction if applicable
        admin_user_id INTEGER, -- If manually adjusted by admin
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Subscription Events table - for billing and lifecycle tracking
    CREATE TABLE IF NOT EXISTS subscription_events (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        event_type VARCHAR(50) NOT NULL, -- 'created', 'upgraded', 'downgraded', 'cancelled', 'renewed', 'expired'
        old_plan_type VARCHAR(50),
        new_plan_type VARCHAR(50),
        effective_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB DEFAULT '{}',
        created_by INTEGER, -- Admin or system
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_user_plans_user_id ON user_plans(user_id);
    CREATE INDEX IF NOT EXISTS idx_user_plans_status ON user_plans(status);
    CREATE INDEX IF NOT EXISTS idx_user_plans_expiry ON user_plans(subscription_end);
    
    CREATE INDEX IF NOT EXISTS idx_quota_usage_user_month ON quota_usage(user_id, year, month);
    CREATE INDEX IF NOT EXISTS idx_quota_usage_reset_date ON quota_usage(reset_date);
    
    CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date ON daily_usage_details(user_id, usage_date DESC);
    CREATE INDEX IF NOT EXISTS idx_daily_usage_date ON daily_usage_details(usage_date DESC);
    
    CREATE INDEX IF NOT EXISTS idx_quota_transactions_user ON quota_transactions(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_quota_transactions_type ON quota_transactions(transaction_type, created_at DESC);
    
    CREATE INDEX IF NOT EXISTS idx_subscription_events_user ON subscription_events(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_subscription_events_type ON subscription_events(event_type, created_at DESC);

    RAISE NOTICE 'Quota management tables created successfully';
END;
$$ LANGUAGE plpgsql;

-- Default plan configurations
CREATE OR REPLACE FUNCTION setup_default_plans()
RETURNS void AS $$
BEGIN
    -- Insert default plans
    INSERT INTO plan_features (plan_type, plan_name, monthly_quota, max_file_size_mb, 
                              max_concurrent_uploads, priority_queue, advanced_analytics, 
                              api_access, email_support, phone_support, price_monthly_cents)
    VALUES 
        ('free', 'Plan Gratuit', 600, 50, 1, false, false, false, false, false, 0),
        ('premium', 'Plan Premium', 3000, 50, 1, false, false, false, false, false, 1990),
        ('enterprise', 'Plan Entreprise', 100000, 50, 1, false, false, false, false, false, 9990)
    ON CONFLICT (plan_type) DO UPDATE SET
        updated_at = CURRENT_TIMESTAMP;

    RAISE NOTICE 'Default plans configured';
END;
$$ LANGUAGE plpgsql;

-- Function to initialize user with free plan
CREATE OR REPLACE FUNCTION assign_free_plan_to_user(p_user_id INTEGER)
RETURNS void AS $$
BEGIN
    -- Assign free plan to new user
    INSERT INTO user_plans (user_id, plan_type, status)
    VALUES (p_user_id, 'free', 'active')
    ON CONFLICT (user_id) DO NOTHING;
    
    -- Initialize current month quota usage
    INSERT INTO quota_usage (user_id, month, year)
    VALUES (p_user_id, EXTRACT(month FROM CURRENT_TIMESTAMP)::INTEGER, 
            EXTRACT(year FROM CURRENT_TIMESTAMP)::INTEGER)
    ON CONFLICT (user_id, month, year) DO NOTHING;
    
    RAISE NOTICE 'Free plan assigned to user %', p_user_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check and update quota
CREATE OR REPLACE FUNCTION check_and_update_quota(
    p_user_id INTEGER,
    p_prediction_count INTEGER DEFAULT 1,
    p_file_size_bytes BIGINT DEFAULT 0
) RETURNS JSONB AS $$
DECLARE
    v_current_month INTEGER := EXTRACT(month FROM CURRENT_TIMESTAMP);
    v_current_year INTEGER := EXTRACT(year FROM CURRENT_TIMESTAMP);
    v_plan plan_features%ROWTYPE;
    v_usage quota_usage%ROWTYPE;
    v_result JSONB;
BEGIN
    -- Get user's current plan
    SELECT pf.* INTO v_plan
    FROM plan_features pf
    JOIN user_plans up ON pf.plan_type = up.plan_type
    WHERE up.user_id = p_user_id AND up.status = 'active';
    
    IF NOT FOUND THEN
        -- No plan found, assign free plan
        PERFORM assign_free_plan_to_user(p_user_id);
        SELECT * INTO v_plan FROM plan_features WHERE plan_type = 'free';
    END IF;
    
    -- Get or create current month usage
    INSERT INTO quota_usage (user_id, month, year)
    VALUES (p_user_id, v_current_month, v_current_year)
    ON CONFLICT (user_id, month, year) DO NOTHING;
    
    SELECT * INTO v_usage FROM quota_usage 
    WHERE user_id = p_user_id AND month = v_current_month AND year = v_current_year;
    
    -- Check if quota would be exceeded
    IF v_usage.prediction_count + p_prediction_count > v_plan.monthly_quota THEN
        v_result := jsonb_build_object(
            'allowed', false,
            'reason', 'quota_exceeded',
            'current_usage', v_usage.prediction_count,
            'monthly_quota', v_plan.monthly_quota,
            'plan_type', v_plan.plan_type,
            'remaining', v_plan.monthly_quota - v_usage.prediction_count
        );
    ELSE
        -- Update usage
        UPDATE quota_usage 
        SET prediction_count = prediction_count + p_prediction_count,
            total_file_size_bytes = total_file_size_bytes + p_file_size_bytes,
            last_prediction_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = p_user_id AND month = v_current_month AND year = v_current_year;
        
        -- Record transaction
        INSERT INTO quota_transactions (user_id, transaction_type, amount, reason)
        VALUES (p_user_id, 'usage', p_prediction_count, 'prediction_processed');
        
        v_result := jsonb_build_object(
            'allowed', true,
            'current_usage', v_usage.prediction_count + p_prediction_count,
            'monthly_quota', v_plan.monthly_quota,
            'plan_type', v_plan.plan_type,
            'remaining', v_plan.monthly_quota - (v_usage.prediction_count + p_prediction_count)
        );
    END IF;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically assign free plan to new users
CREATE OR REPLACE FUNCTION trigger_assign_free_plan()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM assign_free_plan_to_user(NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Only create trigger if user table exists
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user') THEN
        DROP TRIGGER IF EXISTS after_user_insert_assign_plan ON "user";
        CREATE TRIGGER after_user_insert_assign_plan
            AFTER INSERT ON "user"
            FOR EACH ROW
            EXECUTE FUNCTION trigger_assign_free_plan();
    END IF;
END $$;

-- Execute quota system setup
SELECT create_quota_management_tables();
SELECT setup_default_plans();

-- Execute secure initialization
SELECT * FROM initialize_secure_nightscan_db();