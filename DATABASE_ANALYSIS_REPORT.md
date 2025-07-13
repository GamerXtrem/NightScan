# NightScan Database Schema Analysis Report

## Executive Summary

This report provides a comprehensive analysis of the NightScan project's database schema and data models. The analysis reveals several critical issues including missing foreign key constraints, inconsistent data types, normalization problems, and lack of proper migration management.

## Database Architecture Overview

### Database Type
- **Primary Database**: PostgreSQL (with TimescaleDB extension in optimized configuration)
- **Cache Layer**: Redis
- **Secondary Database**: SQLite (used in Raspberry Pi location_manager)
- **Connection Pooling**: pgBouncer (in production)

### ORM/Database Access
- **Primary ORM**: SQLAlchemy with Flask-SQLAlchemy
- **Direct SQL**: psycopg2 for quota management
- **Query Builder**: Custom SecureQueryBuilder class for SQL injection prevention

## Database Tables and Models

### 1. Core User Management Tables

#### `User` (web/app.py:258-268)
```python
- id: Integer (Primary Key)
- username: String(80), unique, not null
- password_hash: String(128), not null
```
**Issues Found:**
- Missing email field despite references in init-db.sql
- No created_at/updated_at timestamps
- No is_active or is_admin fields as referenced in SQL scripts
- No indexes defined in model

#### `NotificationPreference` (web/app.py:320-351)
```python
- id: Integer (Primary Key)
- user_id: Integer, Foreign Key(user.id), unique
- email_notifications: Boolean
- push_notifications: Boolean
- email_address: String(200)
- min_priority: String(20)
- species_filter: Text (JSON string)
- zone_filter: Text (JSON string)
- quiet_hours_start: String(5)
- quiet_hours_end: String(5)
- slack_webhook: String(500)
- discord_webhook: String(500)
```
**Issues Found:**
- JSON stored as Text instead of native JSON/JSONB type
- No validation on JSON structure
- webhook URLs stored without encryption

### 2. Wildlife Detection Tables

#### `Detection` (web/app.py:291-318)
```python
- id: Integer (Primary Key)
- species: String(100), not null
- time: DateTime, not null, server_default=now()
- latitude: Float
- longitude: Float
- zone: String(100)
- image_url: String(200)
- confidence: Float, default=0.0
- user_id: Integer, Foreign Key(user.id)
- description: Text
```
**Issues Found:**
- No composite index on (species, time) for efficient queries
- Missing geospatial index despite lat/lon fields
- No validation on confidence range (0-1)
- image_url limited to 200 chars (too short for some URLs)

#### `Prediction` (web/app.py:270-289)
```python
- id: Integer (Primary Key)
- user_id: Integer, Foreign Key(user.id), not null
- filename: String(200)
- result: Text
- file_size: Integer
- created_at: DateTime, not null, default=utcnow
```
**Issues Found:**
- `created_at` field exists but migration script suggests it was added later
- No index on created_at for retention queries
- file_size as Integer may overflow for large files (should be BigInteger)

### 3. Quota Management Tables

#### `PlanFeatures` (web/app.py:354-387)
```python
- id: String(36), Primary Key (UUID)
- plan_type: String(50), unique, not null
- plan_name: String(100), not null
- monthly_quota: Integer, not null
- max_file_size_mb: Integer, not null
- max_concurrent_uploads: Integer, not null
- priority_queue: Boolean, not null
- advanced_analytics: Boolean, not null
- api_access: Boolean, not null
- email_support: Boolean, not null
- phone_support: Boolean, not null
- features_json: Text
- price_monthly_cents: Integer, not null
- price_yearly_cents: Integer
- data_retention_days: Integer, not null
- is_active: Boolean, not null
- created_at: DateTime, not null
- updated_at: DateTime, not null, onupdate
```
**Issues Found:**
- UUID stored as String(36) instead of native UUID type
- features_json as Text instead of JSON/JSONB
- No foreign key relationship defined with subscription plans

#### `UserPlan` (web/app.py:389-422)
```python
- id: String(36), Primary Key (UUID)
- user_id: Integer, Foreign Key(user.id), unique
- plan_type: String(50), Foreign Key(plan_features.plan_type)
- subscription_start: DateTime, not null
- subscription_end: DateTime
- auto_renew: Boolean, not null
- payment_method: String(50)
- subscription_id: String(200)
- trial_end: DateTime
- is_trial: Boolean, not null
- status: String(20), not null
- created_at: DateTime, not null
- updated_at: DateTime, not null, onupdate
```
**Issues Found:**
- Missing enum validation for status field
- payment_method stored in plain text
- No check constraint for subscription_end > subscription_start

#### `QuotaUsage` (web/app.py:424-462)
```python
- id: String(36), Primary Key (UUID)
- user_id: Integer, Foreign Key(user.id)
- month: Integer, not null
- year: Integer, not null
- prediction_count: Integer, not null
- total_file_size_bytes: BigInteger, not null
- successful_predictions: Integer, not null
- failed_predictions: Integer, not null
- premium_features_used: Text
- reset_date: DateTime, not null
- last_prediction_at: DateTime
- created_at: DateTime, not null
- updated_at: DateTime, not null, onupdate
- UniqueConstraint: (user_id, month, year)
```
**Issues Found:**
- No check constraints for month (1-12) and year ranges
- premium_features_used as Text instead of JSON

#### `DailyUsageDetails` (web/app.py:464-496)
```python
- id: String(36), Primary Key (UUID)
- user_id: Integer, Foreign Key(user.id)
- usage_date: Date, not null
- prediction_count: Integer, not null
- total_file_size_bytes: BigInteger, not null
- average_processing_time_ms: Integer
- peak_hour: Integer
- device_type: String(50)
- app_version: String(20)
- created_at: DateTime, not null
- UniqueConstraint: (user_id, usage_date)
```
**Issues Found:**
- No check constraint for peak_hour (0-23)
- No index on usage_date for time-series queries

#### `QuotaTransaction` (web/app.py:498-530)
```python
- id: String(36), Primary Key (UUID)
- user_id: Integer, Foreign Key(user.id)
- transaction_type: String(50), not null
- amount: Integer, not null
- reason: String(200)
- metadata: Text
- prediction_id: Integer, Foreign Key(prediction.id)
- admin_user_id: Integer, Foreign Key(user.id)
- created_at: DateTime, not null
```
**Issues Found:**
- No enum validation for transaction_type
- Two foreign keys to user table may cause confusion
- metadata as Text instead of JSON

#### `SubscriptionEvent` (web/app.py:532-563)
```python
- id: String(36), Primary Key (UUID)
- user_id: Integer, Foreign Key(user.id)
- event_type: String(50), not null
- old_plan_type: String(50)
- new_plan_type: String(50)
- effective_date: DateTime, not null
- metadata: Text
- created_by: Integer, Foreign Key(user.id)
- created_at: DateTime, not null
```
**Issues Found:**
- No enum validation for event_type
- metadata as Text instead of JSON
- No foreign key to plan_features for plan types

### 4. Data Retention Tables

#### `DataRetentionLog` (web/app.py:567-601)
```python
- id: Integer (Primary Key)
- user_id: Integer, Foreign Key(user.id)
- plan_type: String(50), not null
- retention_days: Integer, not null
- records_deleted: Integer, not null
- total_size_deleted_bytes: BigInteger, not null
- deletion_date: DateTime, not null
- retention_policy_version: String(50)
- admin_override: Boolean
- admin_reason: Text
- metadata: Text
```
**Issues Found:**
- metadata as Text instead of JSON
- No index on deletion_date for audit queries

#### `PredictionArchive` (web/app.py:603-632)
```python
- id: Integer (Primary Key)
- original_prediction_id: Integer, not null
- user_id: Integer, not null
- filename: String(200)
- result: Text
- file_size: Integer
- created_at: DateTime, not null
- archived_at: DateTime, not null
- plan_type_at_archive: String(50)
- retention_days: Integer
- archived_by: String(50)
```
**Issues Found:**
- user_id not defined as foreign key (orphaned records possible)
- No index on original_prediction_id for lookups

## Missing Foreign Key Constraints

1. **PredictionArchive.user_id** → User.id (missing)
2. **SubscriptionEvent plan types** → PlanFeatures.plan_type (missing)
3. **No cascading deletes defined** for any foreign keys

## Data Type Inconsistencies

1. **UUID Fields**: Stored as String(36) instead of native UUID type
2. **JSON Fields**: Stored as Text instead of JSON/JSONB (PostgreSQL)
3. **Timestamp Fields**: Inconsistent use of DateTime vs server_default
4. **Boolean Fields**: Some have defaults, others don't

## Normalization Issues

1. **Denormalized Data**:
   - Species information embedded in Detection table
   - Plan features duplicated in multiple tables
   - User email in NotificationPreference instead of User table

2. **Missing Lookup Tables**:
   - No species/wildlife catalog table
   - No transaction_types table
   - No subscription_event_types table

## Schema Migration Issues

1. **No Migration Framework**: No Alembic or similar migration tool configured
2. **Manual SQL Scripts**: Found migration script (add_data_retention_system.sql) indicates ad-hoc migrations
3. **Schema Drift Risk**: SQLAlchemy models may drift from actual database schema
4. **No Version Tracking**: No schema version table

## Missing Indexes

### Critical Missing Indexes:
1. **User.username** - needed for login queries
2. **Prediction.created_at** - needed for retention queries
3. **Detection.(species, zone)** - composite for filtering
4. **QuotaUsage.(user_id, month, year)** - despite unique constraint
5. **DailyUsageDetails.usage_date** - for time-series queries

### Existing Index Attempts:
- init-db.sql attempts to create indexes but may fail if tables don't exist
- web/app.py:1036-1062 creates indexes programmatically but error handling suppresses failures

## Database Connection Issues

1. **Multiple Connection Methods**:
   - SQLAlchemy in web app
   - Direct psycopg2 in quota_manager
   - SQLite in location_manager (data isolation)

2. **Connection Pool Configuration**:
   - Default pool_size: 10
   - No connection validation
   - No proper connection recycling for long-running processes

3. **Missing Connection Parameters**:
   - No SSL/TLS configuration
   - No connection timeout handling
   - No retry logic

## Security Concerns

1. **Sensitive Data Storage**:
   - Webhook URLs stored unencrypted
   - Payment methods stored as plain text
   - No field-level encryption

2. **Audit Trail**:
   - Limited audit logging
   - No trigger-based audit trail
   - Security events table only in init-db-secure.sql

3. **Access Control**:
   - No Row Level Security (RLS) implemented
   - No database-level user separation
   - Single database user for application

## Recommended Actions

### Immediate (Critical):

1. **Add Missing Foreign Keys**:
   ```sql
   ALTER TABLE prediction_archive 
   ADD CONSTRAINT fk_archive_user 
   FOREIGN KEY (user_id) REFERENCES "user"(id);
   ```

2. **Fix Data Types**:
   - Convert String(36) UUIDs to native UUID type
   - Convert Text JSON fields to JSONB

3. **Add Critical Indexes**:
   ```sql
   CREATE INDEX idx_user_username ON "user"(username);
   CREATE INDEX idx_prediction_created_at ON prediction(created_at);
   ```

### Short-term (Important):

1. **Implement Migration Framework**:
   - Set up Alembic for schema versioning
   - Create initial migration from current state
   - Document migration procedures

2. **Normalize Schema**:
   - Create species lookup table
   - Create enum tables for status fields
   - Move user email to User table

3. **Improve Connection Management**:
   - Implement connection validation
   - Add SSL/TLS configuration
   - Configure proper timeouts

### Long-term (Enhancement):

1. **Security Hardening**:
   - Implement field-level encryption
   - Add Row Level Security
   - Create database audit triggers

2. **Performance Optimization**:
   - Implement table partitioning for time-series data
   - Add materialized views for analytics
   - Configure query optimization

3. **Data Integrity**:
   - Add check constraints
   - Implement database-level validation
   - Create data quality monitoring

## Conclusion

The NightScan database schema shows signs of organic growth without proper planning. Critical issues include missing foreign keys, inconsistent data types, and lack of migration management. Immediate attention should be given to data integrity constraints and index creation to prevent data corruption and performance degradation.