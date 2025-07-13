"""
Updated SQLAlchemy Model Relationships for NightScan

This file contains the updates needed for the SQLAlchemy models to properly
define all foreign key relationships and add validation.
"""

# Add these relationship updates to the existing models in web/app.py

# =====================================================
# PredictionArchive Model Update
# =====================================================
# In the PredictionArchive class, add:
"""
class PredictionArchive(db.Model):
    # ... existing fields ...
    
    # Update user_id to be nullable (for when users are deleted)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="SET NULL"), nullable=True)
    
    # Add relationship
    user = db.relationship("User", backref=db.backref("prediction_archives", lazy=True))
    
    # Add validation
    @validates('file_size')
    def validate_file_size(self, key, value):
        if value is not None and value < 0:
            raise ValueError("File size cannot be negative")
        return value
"""

# =====================================================
# SubscriptionEvent Model Update
# =====================================================
# In the SubscriptionEvent class, add:
"""
class SubscriptionEvent(db.Model):
    # ... existing fields ...
    
    # Update plan type columns with foreign keys
    old_plan_type = db.Column(db.String(50), db.ForeignKey("plan_features.plan_type", ondelete="SET NULL"))
    new_plan_type = db.Column(db.String(50), db.ForeignKey("plan_features.plan_type", ondelete="SET NULL"))
    
    # Add relationships
    old_plan = db.relationship("PlanFeatures", foreign_keys=[old_plan_type], backref="events_as_old_plan")
    new_plan = db.relationship("PlanFeatures", foreign_keys=[new_plan_type], backref="events_as_new_plan")
    
    # Add validation
    @validates('event_type')
    def validate_event_type(self, key, value):
        valid_types = ['created', 'upgraded', 'downgraded', 'cancelled', 'renewed', 'expired', 'reactivated']
        if value not in valid_types:
            raise ValueError(f"Invalid event type. Must be one of: {', '.join(valid_types)}")
        return value
"""

# =====================================================
# DataRetentionLog Model Update
# =====================================================
# In the DataRetentionLog class, add:
"""
class DataRetentionLog(db.Model):
    # ... existing fields ...
    
    # Update plan_type with foreign key
    plan_type = db.Column(db.String(50), db.ForeignKey("plan_features.plan_type", ondelete="SET NULL"))
    
    # Add relationship
    plan = db.relationship("PlanFeatures", backref=db.backref("retention_logs", lazy=True))
"""

# =====================================================
# Detection Model Update
# =====================================================
# In the Detection class, add validation:
"""
from sqlalchemy.orm import validates

class Detection(db.Model):
    # ... existing fields ...
    
    @validates('confidence')
    def validate_confidence(self, key, value):
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return value
    
    @validates('latitude')
    def validate_latitude(self, key, value):
        if value is not None and (value < -90 or value > 90):
            raise ValueError("Latitude must be between -90 and 90")
        return value
    
    @validates('longitude')
    def validate_longitude(self, key, value):
        if value is not None and (value < -180 or value > 180):
            raise ValueError("Longitude must be between -180 and 180")
        return value
"""

# =====================================================
# QuotaUsage Model Update
# =====================================================
# In the QuotaUsage class, add validation:
"""
class QuotaUsage(db.Model):
    # ... existing fields ...
    
    @validates('month')
    def validate_month(self, key, value):
        if value < 1 or value > 12:
            raise ValueError("Month must be between 1 and 12")
        return value
    
    @validates('year')
    def validate_year(self, key, value):
        if value < 2024 or value > 2100:
            raise ValueError("Year must be between 2024 and 2100")
        return value
"""

# =====================================================
# DailyUsageDetails Model Update
# =====================================================
# In the DailyUsageDetails class, add validation:
"""
class DailyUsageDetails(db.Model):
    # ... existing fields ...
    
    @validates('peak_hour')
    def validate_peak_hour(self, key, value):
        if value is not None and (value < 0 or value > 23):
            raise ValueError("Peak hour must be between 0 and 23")
        return value
"""

# =====================================================
# UserPlan Model Update
# =====================================================
# In the UserPlan class, add validation:
"""
class UserPlan(db.Model):
    # ... existing fields ...
    
    @validates('status')
    def validate_status(self, key, value):
        valid_statuses = ['active', 'cancelled', 'suspended', 'expired', 'pending']
        if value not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")
        return value
    
    @validates('subscription_end')
    def validate_subscription_end(self, key, value):
        if value is not None and self.subscription_start and value <= self.subscription_start:
            raise ValueError("Subscription end must be after subscription start")
        return value
"""

# =====================================================
# QuotaTransaction Model Update
# =====================================================
# In the QuotaTransaction class, add validation:
"""
class QuotaTransaction(db.Model):
    # ... existing fields ...
    
    @validates('transaction_type')
    def validate_transaction_type(self, key, value):
        valid_types = ['usage', 'bonus', 'reset', 'adjustment']
        if value not in valid_types:
            raise ValueError(f"Invalid transaction type. Must be one of: {', '.join(valid_types)}")
        return value
"""

# =====================================================
# NotificationPreference Model Update
# =====================================================
# In the NotificationPreference class, add validation:
"""
import re

class NotificationPreference(db.Model):
    # ... existing fields ...
    
    @validates('min_priority')
    def validate_min_priority(self, key, value):
        valid_priorities = ['low', 'normal', 'high', 'critical']
        if value not in valid_priorities:
            raise ValueError(f"Invalid priority. Must be one of: {', '.join(valid_priorities)}")
        return value
    
    @validates('quiet_hours_start', 'quiet_hours_end')
    def validate_quiet_hours(self, key, value):
        if value is not None:
            if not re.match(r'^[0-2][0-9]:[0-5][0-9]$', value):
                raise ValueError("Quiet hours must be in HH:MM format (24-hour)")
            # Additional validation to ensure valid hour
            hour = int(value.split(':')[0])
            if hour > 23:
                raise ValueError("Hour must be between 00 and 23")
        return value
"""

# =====================================================
# Prediction Model Update
# =====================================================
# In the Prediction class, add validation:
"""
class Prediction(db.Model):
    # ... existing fields ...
    
    @validates('file_size')
    def validate_file_size(self, key, value):
        if value is not None and value < 0:
            raise ValueError("File size cannot be negative")
        return value
"""

# =====================================================
# PlanFeatures Model Update
# =====================================================
# In the PlanFeatures class, add validation:
"""
class PlanFeatures(db.Model):
    # ... existing fields ...
    
    @validates('monthly_quota')
    def validate_monthly_quota(self, key, value):
        if value < 0:
            raise ValueError("Monthly quota cannot be negative")
        return value
    
    @validates('max_file_size_mb')
    def validate_max_file_size(self, key, value):
        if value <= 0:
            raise ValueError("Max file size must be positive")
        return value
    
    @validates('data_retention_days')
    def validate_retention_days(self, key, value):
        if value <= 0:
            raise ValueError("Data retention days must be positive")
        return value
    
    @validates('price_monthly_cents')
    def validate_price(self, key, value):
        if value < 0:
            raise ValueError("Price cannot be negative")
        return value
"""

# =====================================================
# Import Statement Update
# =====================================================
# Add this import at the top of web/app.py:
"""
from sqlalchemy.orm import validates
"""