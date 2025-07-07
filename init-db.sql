-- NightScan Production Database Initialization
-- This file is loaded during PostgreSQL container startup

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable PostGIS for future location features (optional)
-- CREATE EXTENSION IF NOT EXISTS "postgis";

-- Performance indexes (tables created by SQLAlchemy models)
-- These will be created AFTER SQLAlchemy creates the base tables

-- Function to create indexes if tables exist
CREATE OR REPLACE FUNCTION create_nightscan_indexes()
RETURNS void AS $$
BEGIN
    -- Check if tables exist before creating indexes
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'prediction') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prediction_user_id ON prediction(user_id);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prediction_timestamp ON prediction(timestamp DESC);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prediction_confidence ON prediction(confidence DESC);
    END IF;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'detection') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_time ON detection(time DESC);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_species ON detection(species);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detection_confidence ON detection(confidence DESC);
    END IF;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_email ON "user"(email);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_active ON "user"(is_active);
    END IF;
    
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'audio_file') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audio_file_user ON audio_file(user_id);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audio_file_timestamp ON audio_file(timestamp DESC);
    END IF;
    
    RAISE NOTICE 'NightScan performance indexes created successfully';
END;
$$ LANGUAGE plpgsql;

-- Initial configuration data
-- This will be inserted after SQLAlchemy creates tables

-- Create admin user insert function
CREATE OR REPLACE FUNCTION create_admin_user()
RETURNS void AS $$
BEGIN
    -- Insert default admin user if users table exists and is empty
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user') THEN
        INSERT INTO "user" (id, email, username, password_hash, is_active, is_admin, created_at)
        SELECT 
            uuid_generate_v4(),
            'admin@nightscan.local',
            'admin',
            '$2b$12$LQv3c1yqBw2aUtfYWJm.qOL6xFyOjXYBhWjBZPw9FYrD8Z8o4Q4Qa', -- password: 'nightscan_admin_2024'
            true,
            true,
            CURRENT_TIMESTAMP
        WHERE NOT EXISTS (SELECT 1 FROM "user" WHERE email = 'admin@nightscan.local');
        
        RAISE NOTICE 'Default admin user created (admin@nightscan.local / nightscan_admin_2024)';
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create system settings function
CREATE OR REPLACE FUNCTION create_system_settings()
RETURNS void AS $$
BEGIN
    -- Create settings table if it doesn't exist
    CREATE TABLE IF NOT EXISTS system_settings (
        key VARCHAR(100) PRIMARY KEY,
        value TEXT,
        description TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Insert default settings
    INSERT INTO system_settings (key, value, description) VALUES
        ('system_version', '1.0.0', 'NightScan system version'),
        ('max_upload_size_mb', '50', 'Maximum file upload size in MB'),
        ('supported_audio_formats', 'wav,mp3,flac,m4a', 'Supported audio file formats'),
        ('supported_image_formats', 'jpg,jpeg,png,tiff', 'Supported image file formats'),
        ('prediction_confidence_threshold', '0.7', 'Minimum confidence for valid predictions'),
        ('max_predictions_per_user_per_day', '1000', 'Rate limiting for predictions'),
        ('backup_retention_days', '30', 'Days to keep backup files'),
        ('session_timeout_hours', '24', 'User session timeout'),
        ('enable_notifications', 'true', 'Enable email notifications'),
        ('maintenance_mode', 'false', 'System maintenance mode')
    ON CONFLICT (key) DO NOTHING;
    
    RAISE NOTICE 'System settings initialized';
END;
$$ LANGUAGE plpgsql;

-- Species/labels configuration
CREATE OR REPLACE FUNCTION create_species_config()
RETURNS void AS $$
BEGIN
    -- Create species table for ML model labels
    CREATE TABLE IF NOT EXISTS species (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) UNIQUE NOT NULL,
        scientific_name VARCHAR(100),
        category VARCHAR(50) DEFAULT 'wildlife',
        confidence_threshold FLOAT DEFAULT 0.7,
        description TEXT,
        is_active BOOLEAN DEFAULT true,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Insert common wildlife species (placeholder data)
    INSERT INTO species (name, scientific_name, category, description) VALUES
        ('Bird Song', 'Aves spp.', 'bird', 'General bird vocalizations'),
        ('Mammal Call', 'Mammalia spp.', 'mammal', 'General mammal vocalizations'),
        ('Insect Sound', 'Insecta spp.', 'insect', 'General insect sounds'),
        ('Amphibian Call', 'Amphibia spp.', 'amphibian', 'General amphibian vocalizations'),
        ('Environmental Sound', 'N/A', 'environment', 'Non-animal environmental sounds'),
        ('Unknown Species', 'Unknown', 'unknown', 'Unidentified species or sounds')
    ON CONFLICT (name) DO NOTHING;
    
    RAISE NOTICE 'Species configuration created';
END;
$$ LANGUAGE plpgsql;

-- Statistics and monitoring tables
CREATE OR REPLACE FUNCTION create_monitoring_tables()
RETURNS void AS $$
BEGIN
    -- System performance metrics
    CREATE TABLE IF NOT EXISTS system_metrics (
        id SERIAL PRIMARY KEY,
        metric_name VARCHAR(100) NOT NULL,
        metric_value FLOAT NOT NULL,
        unit VARCHAR(20),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- API usage statistics
    CREATE TABLE IF NOT EXISTS api_usage (
        id SERIAL PRIMARY KEY,
        endpoint VARCHAR(200) NOT NULL,
        method VARCHAR(10) NOT NULL,
        status_code INTEGER NOT NULL,
        response_time_ms INTEGER,
        user_id UUID,
        ip_address INET,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Create indexes for monitoring tables
    CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
    CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint);
    CREATE INDEX IF NOT EXISTS idx_api_usage_user ON api_usage(user_id);
    
    RAISE NOTICE 'Monitoring tables created';
END;
$$ LANGUAGE plpgsql;

-- Cleanup function for maintenance
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Clean up old system metrics (keep 30 days)
    DELETE FROM system_metrics 
    WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Clean up old API usage logs (keep 90 days)
    DELETE FROM api_usage 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Vacuum and analyze for performance
    VACUUM ANALYZE;
    
    RAISE NOTICE 'Database cleanup completed';
END;
$$ LANGUAGE plpgsql;

-- Main initialization function
CREATE OR REPLACE FUNCTION initialize_nightscan_db()
RETURNS void AS $$
BEGIN
    RAISE NOTICE 'Starting NightScan database initialization...';
    
    -- Wait a bit for SQLAlchemy to create tables
    PERFORM pg_sleep(2);
    
    -- Execute initialization functions
    PERFORM create_nightscan_indexes();
    PERFORM create_admin_user();
    PERFORM create_system_settings();
    PERFORM create_species_config();
    PERFORM create_monitoring_tables();
    
    RAISE NOTICE 'NightScan database initialization completed successfully!';
    RAISE NOTICE 'Default admin credentials: admin@nightscan.local / nightscan_admin_2024';
    RAISE NOTICE 'Please change the admin password after first login!';
END;
$$ LANGUAGE plpgsql;

-- Execute initialization (will run when this file is loaded)
SELECT initialize_nightscan_db();