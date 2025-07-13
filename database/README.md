# NightScan Database Setup

## Overview

NightScan uses PostgreSQL with a complete schema that includes all constraints, foreign keys, and indexes from the start. No migration framework is needed since the database structure is created correctly from the beginning.

## Database Structure

The database includes the following main components:

### Core Tables
- **user**: User authentication and accounts
- **plan_features**: Available subscription plans (free, basic, pro, enterprise)
- **prediction**: Audio file analysis results
- **detection**: Wildlife detection events with geolocation

### Subscription & Quota Management
- **user_plans**: User subscription assignments
- **quota_usage**: Monthly usage tracking
- **daily_usage_details**: Daily analytics
- **quota_transactions**: Audit trail for quota changes
- **subscription_events**: Subscription lifecycle events

### Data Retention
- **prediction_archive**: Soft-deleted predictions
- **data_retention_log**: Retention operation audit trail

### Other
- **notification_preference**: User notification settings

## Key Features

### 1. Foreign Key Constraints
All relationships are properly enforced:
- Cascading deletes where appropriate (e.g., user deletion)
- SET NULL for audit trails (preserves history)
- RESTRICT for critical references (prevents accidental deletion)

### 2. Data Validation
Check constraints ensure data integrity:
- Numeric ranges (confidence: 0.0-1.0, month: 1-12)
- Geographic coordinates (latitude: -90 to 90)
- Date logic (subscription_end > subscription_start)
- Enum values (status, event_type, priority levels)

### 3. Performance Indexes
Strategic indexes for common queries:
- User lookups by username
- Time-based queries (created_at, usage_date)
- Geospatial queries (latitude, longitude)
- Composite indexes for filtering

### 4. Initial Data
Pre-configured subscription plans:
- **Free**: 10 predictions/month, 7-day retention
- **Basic**: 100 predictions/month, 30-day retention ($9.99)
- **Pro**: 500 predictions/month, 90-day retention ($29.99)
- **Enterprise**: 2000 predictions/month, 365-day retention ($99.99)

## Setup Instructions

### Prerequisites
- PostgreSQL 12 or higher
- Database user with CREATE DATABASE privileges

### Quick Setup

1. **Set environment variables**:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=nightscan
export DB_USER=nightscan_user
export DB_PASSWORD=your_secure_password
```

2. **Run the initialization script**:
```bash
cd database
./init_database.sh
```

This script will:
- Create the database
- Apply the complete schema
- Insert initial data
- Verify the installation

### Manual Setup

If you prefer to set up manually:

```bash
# Create database
psql -U postgres -c "CREATE DATABASE nightscan;"

# Apply schema
psql -U postgres -d nightscan -f create_database.sql
```

## Verification

After setup, verify the database:

```sql
-- Check tables
\dt

-- Check foreign keys
\d prediction

-- Check initial data
SELECT * FROM plan_features;

-- Test helper functions
SELECT * FROM check_user_quota(1);
```

## Application Configuration

The application no longer uses `db.create_all()`. Instead, it verifies that the database schema exists on startup. If tables are missing, it will log an error and provide instructions.

## Maintenance

### Adding New Tables
1. Add the table definition to `create_database.sql`
2. Include appropriate constraints and indexes
3. Re-run the script on a fresh database

### Modifying Existing Tables
Since we're not using migrations:
1. For development: Drop and recreate the database
2. For production: Create separate ALTER scripts as needed

## Why No Migration Framework?

For a project in development with no production data:
- **Simpler**: One script creates everything correctly
- **Cleaner**: No migration history to manage
- **Faster**: No need to replay migrations
- **Safer**: All constraints enforced from the start

When you eventually go to production, you can either:
1. Continue with SQL scripts for schema changes
2. Adopt a migration framework at that point

## Troubleshooting

### "Database initialization error" on app startup
Run the initialization script:
```bash
./database/init_database.sh
```

### "Permission denied" errors
Ensure your database user has appropriate privileges:
```sql
GRANT ALL PRIVILEGES ON DATABASE nightscan TO nightscan_user;
```

### Foreign key violations
The schema prevents orphaned data. If you get FK violations:
1. Check the referenced record exists
2. Delete in the correct order (or use CASCADE)

## Database Schema Diagram

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    user     │────<│   prediction     │>────│ prediction_     │
│             │     │                  │     │    archive      │
└─────────────┘     └──────────────────┘     └─────────────────┘
       │                                              
       │            ┌──────────────────┐              
       ├───────────<│    detection     │              
       │            └──────────────────┘              
       │                                              
       │            ┌──────────────────┐     ┌─────────────────┐
       ├───────────<│   user_plans     │>────│ plan_features   │
       │            └──────────────────┘     └─────────────────┘
       │                                              
       │            ┌──────────────────┐              
       ├───────────<│   quota_usage    │              
       │            └──────────────────┘              
       │                                              
       │            ┌──────────────────┐              
       └───────────<│ notification_    │              
                    │   preference     │              
                    └──────────────────┘              
```