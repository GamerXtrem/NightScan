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

-- Execute secure initialization
SELECT * FROM initialize_secure_nightscan_db();