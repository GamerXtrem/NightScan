# NightScan Database Integrity Guide

## Overview

This guide documents the database integrity improvements implemented to address the "Missing foreign keys: Data integrity at risk" issue. The improvements include foreign key constraints, data validation, performance indexes, and integrity checking tools.

## Issues Addressed

### 1. Missing Foreign Key Constraints
- **PredictionArchive.user_id** had no foreign key to User table
- **SubscriptionEvent.old_plan_type** and **new_plan_type** had no foreign keys to PlanFeatures
- **DataRetentionLog.plan_type** had no foreign key to PlanFeatures

### 2. Missing Data Validation
- No constraints on numeric ranges (confidence, months, hours)
- No validation of date relationships
- No enforcement of enumerated values

### 3. Missing Performance Indexes
- No indexes on frequently queried columns
- Missing geospatial indexes for location queries
- No composite indexes for common query patterns

## Implementation Steps

### Step 1: Run Data Integrity Check

Before applying the migration, check for existing data issues:

```bash
cd database/scripts
python data_integrity_cleanup.py --check
```

This will generate a report showing:
- Orphaned records
- Invalid references
- Constraint violations
- Duplicate data

### Step 2: Fix Data Issues

If issues are found, create a backup and fix them:

```bash
python data_integrity_cleanup.py --fix
```

This will:
- Create a timestamped backup
- Fix all identified issues
- Generate a detailed report

### Step 3: Apply Database Migration

Run the migration to add constraints and indexes:

```bash
psql -U $DB_USER -d $DB_NAME -f database/migrations/fix_missing_foreign_keys.sql
```

### Step 4: Update Application Code

Apply the model updates from `web/models_update.py` to add SQLAlchemy validations.

## Foreign Key Constraints Added

### 1. PredictionArchive → User
```sql
ALTER TABLE prediction_archive 
ADD CONSTRAINT fk_prediction_archive_user 
FOREIGN KEY (user_id) REFERENCES "user"(id) ON DELETE SET NULL;
```
- **Behavior**: When a user is deleted, their archived predictions remain but user_id becomes NULL

### 2. SubscriptionEvent → PlanFeatures
```sql
ALTER TABLE subscription_events 
ADD CONSTRAINT fk_subscription_event_old_plan 
FOREIGN KEY (old_plan_type) REFERENCES plan_features(plan_type) ON DELETE SET NULL;

ALTER TABLE subscription_events 
ADD CONSTRAINT fk_subscription_event_new_plan 
FOREIGN KEY (new_plan_type) REFERENCES plan_features(plan_type) ON DELETE SET NULL;
```
- **Behavior**: If a plan type is deleted, historical events remain but plan references become NULL

### 3. DataRetentionLog → PlanFeatures
```sql
ALTER TABLE data_retention_log 
ADD CONSTRAINT fk_data_retention_log_plan 
FOREIGN KEY (plan_type) REFERENCES plan_features(plan_type) ON DELETE SET NULL;
```
- **Behavior**: Audit logs remain intact even if plan types are removed

## Check Constraints Added

### Numeric Range Validations
- **QuotaUsage.month**: 1-12
- **QuotaUsage.year**: 2024-2100  
- **DailyUsageDetails.peak_hour**: 0-23
- **Detection.confidence**: 0.0-1.0
- **Detection.latitude**: -90 to 90
- **Detection.longitude**: -180 to 180

### Date Validations
- **UserPlan**: subscription_end > subscription_start
- **UserPlan**: if is_trial=true, trial_end must not be NULL

### Enum Validations
- **QuotaTransaction.transaction_type**: 'usage', 'bonus', 'reset', 'adjustment'
- **SubscriptionEvent.event_type**: 'created', 'upgraded', 'downgraded', 'cancelled', 'renewed', 'expired', 'reactivated'
- **UserPlan.status**: 'active', 'cancelled', 'suspended', 'expired', 'pending'
- **NotificationPreference.min_priority**: 'low', 'normal', 'high', 'critical'

### Business Rule Validations
- **PlanFeatures**: All numeric values must be positive
- **Prediction/PredictionArchive**: file_size >= 0
- **NotificationPreference**: quiet hours in HH:MM format

## Performance Indexes Added

### Single Column Indexes
- `idx_user_username` - Login performance
- `idx_prediction_created_at` - Time-based queries
- `idx_detection_confidence` - Confidence filtering
- `idx_subscription_event_type` - Event type queries

### Composite Indexes
- `idx_prediction_user_created` - User's predictions by date
- `idx_detection_species_time` - Species tracking over time
- `idx_detection_user_time` - User's detections timeline
- `idx_quota_usage_user_date` - Monthly quota lookups

### Geospatial Index
- `idx_detection_location` - Location-based wildlife queries

### Conditional Indexes
- `idx_user_plan_subscription_end` - Only for non-NULL end dates
- `idx_detection_location` - Only for records with coordinates

## Data Integrity Validation

### Built-in Validation Function

The migration includes a validation function to check integrity:

```sql
SELECT * FROM validate_data_integrity();
```

This returns:
- Orphaned records count
- Invalid references
- Constraint violations
- Duplicate records

### Python Integrity Checker

The `data_integrity_cleanup.py` script provides:
- Comprehensive data validation
- Automatic fixes for common issues
- Backup creation before modifications
- Detailed JSON reports

## SQLAlchemy Model Updates

Add these validations to your models:

```python
from sqlalchemy.orm import validates

class Detection(db.Model):
    @validates('confidence')
    def validate_confidence(self, key, value):
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return value
```

## Rollback Procedure

If needed, the migration can be rolled back:

```bash
psql -U $DB_USER -d $DB_NAME << EOF
BEGIN;
-- Drop all indexes
DROP INDEX IF EXISTS idx_user_username;
-- ... (all other indexes)

-- Drop all check constraints  
ALTER TABLE quota_usage DROP CONSTRAINT IF EXISTS chk_quota_usage_month;
-- ... (all other constraints)

-- Drop all foreign keys
ALTER TABLE prediction_archive DROP CONSTRAINT IF EXISTS fk_prediction_archive_user;
-- ... (all other foreign keys)

COMMIT;
EOF
```

## Monitoring

After implementation, monitor for:
1. Foreign key violations in application logs
2. Performance improvements from new indexes
3. Data quality improvements from constraints

Use the validation function periodically:
```sql
SELECT * FROM validate_data_integrity();
```

## Best Practices

1. **Always backup** before running migrations
2. **Test in staging** environment first
3. **Run integrity check** before and after migration
4. **Monitor logs** for constraint violations
5. **Update application code** to handle new validations

## Troubleshooting

### Common Issues

1. **Migration fails due to existing bad data**
   - Run `data_integrity_cleanup.py --fix` first

2. **Application errors after migration**
   - Check for code attempting invalid inserts
   - Update application validation to match constraints

3. **Performance degradation**
   - Analyze query plans with new indexes
   - Consider partial indexes for large tables

### Support

For issues or questions:
1. Check application logs for constraint violation details
2. Run integrity validation function
3. Review this documentation
4. Contact the database administration team