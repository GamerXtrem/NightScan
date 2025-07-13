#!/usr/bin/env python3
"""
NightScan Quota Management System
Manages user quotas, plan features, and usage tracking with premium tiers.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import psycopg2
import psycopg2.extras
from flask import current_app
from sqlalchemy import text
from cache_manager import cache_user_data, invalidate_user_cache

logger = logging.getLogger(__name__)

class PlanType(Enum):
    """Available subscription plans"""
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

@dataclass
class PlanFeatures:
    """Plan features configuration"""
    plan_type: str
    plan_name: str
    monthly_quota: int
    max_file_size_mb: int
    max_concurrent_uploads: int
    priority_queue: bool
    advanced_analytics: bool
    api_access: bool
    email_support: bool
    phone_support: bool
    price_monthly_cents: int
    data_retention_days: int
    features_json: Dict[str, Any]

@dataclass
class QuotaStatus:
    """Current quota status for a user"""
    user_id: int
    plan_type: str
    current_usage: int
    monthly_quota: int
    remaining: int
    reset_date: datetime
    last_prediction_at: Optional[datetime]
    
    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage"""
        if self.monthly_quota == 0:
            return 100.0
        return (self.current_usage / self.monthly_quota) * 100
    
    @property
    def is_quota_exceeded(self) -> bool:
        """Check if quota is exceeded"""
        return self.current_usage >= self.monthly_quota
    
    @property
    def days_until_reset(self) -> int:
        """Days until quota resets"""
        return (self.reset_date - datetime.now()).days

class QuotaManager:
    """Manages user quotas and plan features"""
    
    def __init__(self, db_connection=None):
        """Initialize quota manager with database connection"""
        self.db = db_connection
        self._plans_cache = {}
        self._cache_expiry = datetime.now()
        self.cache_duration = timedelta(minutes=15)
        
    def _get_db_connection(self):
        """Get database connection"""
        if self.db:
            return self.db
        
        # Try to get from Flask app context
        try:
            from flask import current_app
            return current_app.extensions['sqlalchemy'].db.engine
        except:
            # Fallback to environment variables
            return psycopg2.connect(
                host=os.environ.get('DB_HOST', 'localhost'),
                port=os.environ.get('DB_PORT', '5432'),
                database=os.environ.get('DB_NAME', 'nightscan'),
                user=os.environ.get('DB_USER', 'nightscan'),
                password=os.environ.get('DB_PASSWORD', '')
            )
    
    def _execute_query(self, query: str, params: Tuple = None, fetch_one: bool = False, fetch_all: bool = False):
        """Execute database query with error handling"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(query, params or ())
                    
                    if fetch_one:
                        return cur.fetchone()
                    elif fetch_all:
                        return cur.fetchall()
                    else:
                        conn.commit()
                        return cur.rowcount
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    def get_plan_features(self, plan_type: str) -> Optional[PlanFeatures]:
        """Get plan features with caching"""
        # Check cache
        if (datetime.now() < self._cache_expiry and 
            plan_type in self._plans_cache):
            return self._plans_cache[plan_type]
        
        # Refresh cache if expired
        if datetime.now() >= self._cache_expiry:
            self._refresh_plans_cache()
        
        return self._plans_cache.get(plan_type)
    
    def _refresh_plans_cache(self):
        """Refresh plans cache from database"""
        query = """
        SELECT plan_type, plan_name, monthly_quota, max_file_size_mb,
               max_concurrent_uploads, priority_queue, advanced_analytics,
               api_access, email_support, phone_support, price_monthly_cents,
               data_retention_days, features_json
        FROM plan_features 
        WHERE is_active = true
        """
        
        plans = self._execute_query(query, fetch_all=True)
        self._plans_cache = {}
        
        for plan in plans:
            self._plans_cache[plan['plan_type']] = PlanFeatures(
                plan_type=plan['plan_type'],
                plan_name=plan['plan_name'],
                monthly_quota=plan['monthly_quota'],
                max_file_size_mb=plan['max_file_size_mb'],
                max_concurrent_uploads=plan['max_concurrent_uploads'],
                priority_queue=plan['priority_queue'],
                advanced_analytics=plan['advanced_analytics'],
                api_access=plan['api_access'],
                email_support=plan['email_support'],
                phone_support=plan['phone_support'],
                price_monthly_cents=plan['price_monthly_cents'],
                data_retention_days=plan['data_retention_days'],
                features_json=plan['features_json'] or {}
            )
        
        self._cache_expiry = datetime.now() + self.cache_duration
        logger.info(f"Refreshed plans cache: {list(self._plans_cache.keys())}")
    
    @cache_user_data(ttl=60)  # Cache for 1 minute
    def get_user_quota_status(self, user_id: int) -> QuotaStatus:
        """Get current quota status for user"""
        query = """
        SELECT 
            up.plan_type,
            qu.prediction_count as current_usage,
            pf.monthly_quota,
            qu.reset_date,
            qu.last_prediction_at
        FROM user_plans up
        JOIN plan_features pf ON up.plan_type = pf.plan_type
        LEFT JOIN quota_usage qu ON up.user_id = qu.user_id 
            AND qu.month = EXTRACT(month FROM CURRENT_TIMESTAMP)
            AND qu.year = EXTRACT(year FROM CURRENT_TIMESTAMP)
        WHERE up.user_id = %s AND up.status = 'active'
        """
        
        result = self._execute_query(query, (user_id,), fetch_one=True)
        
        if not result:
            # User has no plan, assign free plan
            self.assign_plan_to_user(user_id, PlanType.FREE.value)
            return self.get_user_quota_status(user_id)
        
        current_usage = result['current_usage'] or 0
        monthly_quota = result['monthly_quota']
        
        return QuotaStatus(
            user_id=user_id,
            plan_type=result['plan_type'],
            current_usage=current_usage,
            monthly_quota=monthly_quota,
            remaining=max(0, monthly_quota - current_usage),
            reset_date=result['reset_date'] or self._get_next_reset_date(),
            last_prediction_at=result['last_prediction_at']
        )
    
    def check_quota_before_prediction(self, user_id: int, file_size_bytes: int = 0) -> Dict[str, Any]:
        """Check if user can make a prediction within quota limits"""
        query = "SELECT check_and_update_quota(%s, 0, %s)"  # Check only, don't update yet
        
        # Get current status first
        status = self.get_user_quota_status(user_id)
        plan = self.get_plan_features(status.plan_type)
        
        if not plan:
            return {
                'allowed': False,
                'reason': 'invalid_plan',
                'message': 'Plan de l\'utilisateur invalide'
            }
        
        # Check quota
        if status.is_quota_exceeded:
            return {
                'allowed': False,
                'reason': 'quota_exceeded',
                'message': f'Quota mensuel dépassé ({status.current_usage}/{status.monthly_quota})',
                'current_usage': status.current_usage,
                'monthly_quota': status.monthly_quota,
                'plan_type': status.plan_type,
                'upgrade_required': True,
                'recommended_plan': self._get_recommended_upgrade(status.plan_type)
            }
        
        # Note: File size check removed - all plans have same limits
        
        return {
            'allowed': True,
            'current_usage': status.current_usage,
            'monthly_quota': status.monthly_quota,
            'remaining': status.remaining,
            'plan_type': status.plan_type
        }
    
    def consume_quota(self, user_id: int, file_size_bytes: int = 0, prediction_id: int = None) -> Dict[str, Any]:
        """Consume quota after successful prediction"""
        query = "SELECT check_and_update_quota(%s, 1, %s)"
        
        try:
            result = self._execute_query(query, (user_id, file_size_bytes), fetch_one=True)
            quota_result = list(result.values())[0]  # Get the JSON result
            
            # Invalidate user cache after quota update
            invalidate_user_cache(user_id)
            
            # Record detailed transaction if prediction_id provided
            if prediction_id and quota_result.get('allowed'):
                self._record_prediction_transaction(user_id, prediction_id, file_size_bytes)
                
            return quota_result
            
        except Exception as e:
            logger.error(f"Failed to consume quota for user {user_id}: {e}")
            return {
                'allowed': False,
                'reason': 'system_error',
                'message': 'Erreur système lors de la vérification du quota'
            }
    
    def _record_prediction_transaction(self, user_id: int, prediction_id: int, file_size_bytes: int):
        """Record detailed prediction transaction"""
        query = """
        INSERT INTO quota_transactions 
        (user_id, transaction_type, amount, reason, metadata, prediction_id)
        VALUES (%s, 'usage', 1, 'prediction_completed', %s, %s)
        """
        
        metadata = {
            'file_size_bytes': file_size_bytes,
            'timestamp': datetime.now().isoformat()
        }
        
        self._execute_query(query, (user_id, json.dumps(metadata), prediction_id))
    
    def assign_plan_to_user(self, user_id: int, plan_type: str, trial_days: int = 0) -> bool:
        """Assign a plan to user"""
        try:
            # Check if plan exists
            plan = self.get_plan_features(plan_type)
            if not plan:
                logger.error(f"Plan {plan_type} not found")
                return False
            
            # Calculate trial end if applicable
            trial_end = None
            if trial_days > 0:
                trial_end = datetime.now() + timedelta(days=trial_days)
            
            # Insert or update user plan
            query = """
            INSERT INTO user_plans (user_id, plan_type, is_trial, trial_end, status)
            VALUES (%s, %s, %s, %s, 'active')
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                plan_type = EXCLUDED.plan_type,
                is_trial = EXCLUDED.is_trial,
                trial_end = EXCLUDED.trial_end,
                updated_at = CURRENT_TIMESTAMP
            """
            
            self._execute_query(query, (user_id, plan_type, trial_days > 0, trial_end))
            
            # Initialize current month quota
            self._initialize_monthly_quota(user_id)
            
            # Record subscription event
            self._record_subscription_event(user_id, 'created', None, plan_type)
            
            logger.info(f"Assigned plan {plan_type} to user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign plan {plan_type} to user {user_id}: {e}")
            return False
    
    def upgrade_user_plan(self, user_id: int, new_plan_type: str) -> Dict[str, Any]:
        """Upgrade user to a new plan"""
        current_status = self.get_user_quota_status(user_id)
        current_plan = current_status.plan_type
        
        new_plan = self.get_plan_features(new_plan_type)
        if not new_plan:
            return {
                'success': False,
                'message': f'Plan {new_plan_type} non trouvé'
            }
        
        # Check if it's actually an upgrade
        plan_hierarchy = [PlanType.FREE.value, PlanType.PREMIUM.value, PlanType.ENTERPRISE.value]
        
        if (current_plan in plan_hierarchy and new_plan_type in plan_hierarchy and
            plan_hierarchy.index(new_plan_type) <= plan_hierarchy.index(current_plan)):
            return {
                'success': False,
                'message': 'Ce n\'est pas une mise à niveau'
            }
        
        # Assign new plan
        if self.assign_plan_to_user(user_id, new_plan_type):
            # Record upgrade event
            self._record_subscription_event(user_id, 'upgraded', current_plan, new_plan_type)
            
            return {
                'success': True,
                'message': f'Mis à niveau vers {new_plan.plan_name}',
                'new_quota': new_plan.monthly_quota,
                'old_quota': self.get_plan_features(current_plan).monthly_quota if current_plan else 0
            }
        
        return {
            'success': False,
            'message': 'Erreur lors de la mise à niveau'
        }
    
    def get_usage_analytics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get detailed usage analytics for user"""
        query = """
        SELECT 
            usage_date,
            prediction_count,
            total_file_size_bytes,
            average_processing_time_ms
        FROM daily_usage_details 
        WHERE user_id = %s 
        AND usage_date >= CURRENT_DATE - INTERVAL '%s days'
        ORDER BY usage_date DESC
        """
        
        daily_usage = self._execute_query(query, (user_id, days), fetch_all=True)
        
        # Calculate summary statistics
        total_predictions = sum(day['prediction_count'] for day in daily_usage)
        total_size_mb = sum(day['total_file_size_bytes'] for day in daily_usage) / (1024 * 1024)
        avg_processing_time = sum(day['average_processing_time_ms'] or 0 for day in daily_usage) / len(daily_usage) if daily_usage else 0
        
        return {
            'period_days': days,
            'total_predictions': total_predictions,
            'total_file_size_mb': round(total_size_mb, 2),
            'average_processing_time_ms': round(avg_processing_time, 2),
            'daily_breakdown': [
                {
                    'date': day['usage_date'].isoformat(),
                    'predictions': day['prediction_count'],
                    'size_mb': round(day['total_file_size_bytes'] / (1024 * 1024), 2),
                    'processing_time_ms': day['average_processing_time_ms']
                }
                for day in daily_usage
            ]
        }
    
    def get_available_plans(self) -> List[Dict[str, Any]]:
        """Get all available plans for user selection"""
        self._refresh_plans_cache()
        
        plans = []
        for plan_type, plan in self._plans_cache.items():
            plans.append({
                'plan_type': plan.plan_type,
                'plan_name': plan.plan_name,
                'monthly_quota': plan.monthly_quota,
                'max_file_size_mb': plan.max_file_size_mb,
                'price_monthly': plan.price_monthly_cents / 100,
                'features': {
                    'max_file_size_mb': plan.max_file_size_mb
                }
            })
        
        # Sort by price
        return sorted(plans, key=lambda x: x['price_monthly'])
    
    def _get_recommended_upgrade(self, current_plan: str) -> str:
        """Get recommended upgrade plan"""
        if current_plan == PlanType.FREE.value:
            return PlanType.PREMIUM.value
        elif current_plan == PlanType.PREMIUM.value:
            return PlanType.ENTERPRISE.value
        else:
            return PlanType.ENTERPRISE.value
    
    def _initialize_monthly_quota(self, user_id: int):
        """Initialize quota for current month"""
        query = """
        INSERT INTO quota_usage (user_id, month, year)
        VALUES (%s, EXTRACT(month FROM CURRENT_TIMESTAMP), EXTRACT(year FROM CURRENT_TIMESTAMP))
        ON CONFLICT (user_id, month, year) DO NOTHING
        """
        self._execute_query(query, (user_id,))
    
    def _record_subscription_event(self, user_id: int, event_type: str, old_plan: str, new_plan: str):
        """Record subscription change event"""
        query = """
        INSERT INTO subscription_events 
        (user_id, event_type, old_plan_type, new_plan_type, metadata)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'user_agent': 'system'
        }
        
        self._execute_query(query, (user_id, event_type, old_plan, new_plan, json.dumps(metadata)))
    
    def _get_next_reset_date(self) -> datetime:
        """Calculate next quota reset date (first day of next month)"""
        now = datetime.now()
        if now.month == 12:
            return datetime(now.year + 1, 1, 1)
        else:
            return datetime(now.year, now.month + 1, 1)
    
    def reset_user_quota(self, user_id: int, admin_reason: str = None) -> bool:
        """Manually reset user quota (admin function)"""
        try:
            # Reset current month usage
            query = """
            UPDATE quota_usage 
            SET prediction_count = 0, 
                total_file_size_bytes = 0,
                successful_predictions = 0,
                failed_predictions = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s 
            AND month = EXTRACT(month FROM CURRENT_TIMESTAMP)
            AND year = EXTRACT(year FROM CURRENT_TIMESTAMP)
            """
            
            rows_affected = self._execute_query(query, (user_id,))
            
            # Invalidate user cache after quota reset
            invalidate_user_cache(user_id)
            
            # Record admin transaction
            if rows_affected > 0:
                transaction_query = """
                INSERT INTO quota_transactions 
                (user_id, transaction_type, amount, reason, metadata)
                VALUES (%s, 'reset', 0, %s, %s)
                """
                
                metadata = {
                    'admin_reset': True,
                    'timestamp': datetime.now().isoformat(),
                    'reason': admin_reason or 'Manual admin reset'
                }
                
                self._execute_query(transaction_query, (user_id, 'admin_reset', json.dumps(metadata)))
                
                logger.info(f"Reset quota for user {user_id}: {admin_reason}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to reset quota for user {user_id}: {e}")
            return False
    
    # ===== DATA RETENTION METHODS =====
    
    def get_user_retention_policy(self, user_id: int) -> Dict[str, Any]:
        """Get the data retention policy for a user"""
        try:
            query = """
            SELECT 
                up.plan_type,
                pf.data_retention_days,
                pf.plan_name
            FROM user_plans up
            JOIN plan_features pf ON up.plan_type = pf.plan_type
            WHERE up.user_id = %s AND up.status = 'active' AND pf.is_active = true
            """
            
            result = self._execute_query(query, (user_id,), fetch_one=True)
            
            if not result:
                return {
                    'error': 'User has no active plan',
                    'plan_type': None,
                    'retention_days': 0
                }
            
            return {
                'user_id': user_id,
                'plan_type': result['plan_type'],
                'plan_name': result['plan_name'],
                'retention_days': result['data_retention_days'],
                'retention_description': self._get_retention_description(result['data_retention_days'])
            }
            
        except Exception as e:
            logger.error(f"Error getting retention policy for user {user_id}: {e}")
            return {
                'error': str(e),
                'retention_days': 0
            }
    
    def get_expired_predictions_count(self, user_id: int) -> Dict[str, Any]:
        """Get count of predictions that should be deleted according to retention policy"""
        try:
            query = """
            SELECT 
                up.plan_type,
                pf.data_retention_days,
                COUNT(p.id) as total_predictions,
                COUNT(CASE WHEN EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > pf.data_retention_days THEN 1 END) as expired_predictions,
                COUNT(CASE WHEN EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > (pf.data_retention_days - 7) THEN 1 END) as expiring_soon,
                SUM(CASE WHEN EXTRACT(DAY FROM (CURRENT_TIMESTAMP - p.created_at)) > pf.data_retention_days THEN p.file_size ELSE 0 END) as expired_size_bytes
            FROM user_plans up
            JOIN plan_features pf ON up.plan_type = pf.plan_type
            LEFT JOIN prediction p ON up.user_id = p.user_id
            WHERE up.user_id = %s AND up.status = 'active' AND pf.is_active = true
            GROUP BY up.plan_type, pf.data_retention_days
            """
            
            result = self._execute_query(query, (user_id,), fetch_one=True)
            
            if not result:
                return {
                    'error': 'User not found or no predictions',
                    'expired_predictions': 0,
                    'expiring_soon': 0
                }
            
            return {
                'user_id': user_id,
                'plan_type': result['plan_type'],
                'retention_days': result['data_retention_days'],
                'total_predictions': result['total_predictions'] or 0,
                'expired_predictions': result['expired_predictions'] or 0,
                'expiring_soon': result['expiring_soon'] or 0,
                'expired_size_bytes': result['expired_size_bytes'] or 0,
                'expired_size_mb': round((result['expired_size_bytes'] or 0) / 1024 / 1024, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting expired predictions for user {user_id}: {e}")
            return {
                'error': str(e),
                'expired_predictions': 0
            }
    
    def cleanup_expired_predictions(self, user_id: int = None, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up expired predictions according to retention policies"""
        try:
            # Use PostgreSQL function for cleanup
            query = "SELECT cleanup_expired_predictions(%s, %s)"
            result = self._execute_query(query, (user_id, dry_run), fetch_one=True)
            
            if result:
                cleanup_result = list(result.values())[0]  # Get the JSON result
                
                # Log the cleanup operation
                if not dry_run and cleanup_result.get('deleted_count', 0) > 0:
                    logger.info(f"Data retention cleanup completed: {cleanup_result}")
                
                return cleanup_result
            
            return {
                'success': False,
                'error': 'No result from cleanup function',
                'deleted_count': 0
            }
            
        except Exception as e:
            logger.error(f"Error during retention cleanup: {e}")
            return {
                'success': False,
                'error': str(e),
                'deleted_count': 0
            }
    
    def get_retention_stats_for_user(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive retention statistics for a user"""
        try:
            # Use PostgreSQL function for stats
            query = "SELECT get_user_retention_stats(%s)"
            result = self._execute_query(query, (user_id,), fetch_one=True)
            
            if result:
                stats = list(result.values())[0]  # Get the JSON result
                return stats
            
            return {
                'error': 'No statistics available',
                'user_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Error getting retention stats for user {user_id}: {e}")
            return {
                'error': str(e),
                'user_id': user_id
            }
    
    def notify_expiring_data(self, user_id: int, days_before: int = 7) -> Dict[str, Any]:
        """Check if user has data expiring soon and needs notification"""
        try:
            expired_info = self.get_expired_predictions_count(user_id)
            
            if expired_info.get('error'):
                return expired_info
            
            expiring_count = expired_info.get('expiring_soon', 0) - expired_info.get('expired_predictions', 0)
            
            if expiring_count > 0:
                retention_policy = self.get_user_retention_policy(user_id)
                
                return {
                    'should_notify': True,
                    'user_id': user_id,
                    'expiring_count': expiring_count,
                    'days_until_deletion': days_before,
                    'plan_type': retention_policy.get('plan_type'),
                    'retention_days': retention_policy.get('retention_days'),
                    'upgrade_options': self._get_upgrade_options_for_retention(retention_policy.get('plan_type'))
                }
            
            return {
                'should_notify': False,
                'user_id': user_id,
                'expiring_count': 0
            }
            
        except Exception as e:
            logger.error(f"Error checking expiring data for user {user_id}: {e}")
            return {
                'error': str(e),
                'should_notify': False
            }
    
    def get_all_plans_with_retention(self) -> List[Dict[str, Any]]:
        """Get all plans with retention information for user comparison"""
        plans = self.get_available_plans()
        
        # Add retention information to each plan
        for plan in plans:
            plan_details = self.get_plan_features(plan['plan_type'])
            if plan_details:
                # Add retention info from the enhanced plan features
                plan['data_retention_days'] = getattr(plan_details, 'data_retention_days', 30)
                plan['retention_description'] = self._get_retention_description(plan['data_retention_days'])
                plan['features']['data_retention_days'] = plan['data_retention_days']
        
        return plans
    
    def _get_retention_description(self, days: int) -> str:
        """Get human-readable description of retention period"""
        if days <= 7:
            return f"{days} jour{'s' if days > 1 else ''}"
        elif days <= 31:
            return f"{days} jours"
        elif days <= 180:
            months = days // 30
            return f"{months} mois"
        else:
            years = days // 365
            return f"{years} an{'s' if years > 1 else ''}"
    
    def _get_upgrade_options_for_retention(self, current_plan: str) -> List[Dict[str, Any]]:
        """Get upgrade options that provide longer retention"""
        if not current_plan:
            return []
        
        all_plans = self.get_all_plans_with_retention()
        current_retention = None
        
        # Find current plan retention
        for plan in all_plans:
            if plan['plan_type'] == current_plan:
                current_retention = plan.get('data_retention_days', 0)
                break
        
        if current_retention is None:
            return []
        
        # Find plans with longer retention
        upgrade_options = []
        for plan in all_plans:
            if plan.get('data_retention_days', 0) > current_retention:
                upgrade_options.append({
                    'plan_type': plan['plan_type'],
                    'plan_name': plan['plan_name'],
                    'retention_days': plan['data_retention_days'],
                    'retention_description': plan['retention_description'],
                    'price_monthly': plan['price_monthly'],
                    'retention_improvement': plan['data_retention_days'] - current_retention
                })
        
        return sorted(upgrade_options, key=lambda x: x['price_monthly'])

# Global quota manager instance
_quota_manager = None

def get_quota_manager() -> QuotaManager:
    """Get or create global quota manager instance"""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager

def check_user_quota_decorator(f):
    """Decorator to check quota before executing function"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Extract user_id from function arguments or current_user
        user_id = kwargs.get('user_id')
        if not user_id:
            try:
                from flask_login import current_user
                user_id = current_user.id if current_user.is_authenticated else None
            except:
                pass
        
        if not user_id:
            return {'error': 'User not authenticated'}, 401
        
        # Check quota
        quota_manager = get_quota_manager()
        file_size = kwargs.get('file_size', 0)
        quota_check = quota_manager.check_quota_before_prediction(user_id, file_size)
        
        if not quota_check['allowed']:
            return {'error': quota_check['message'], 'quota_status': quota_check}, 429
        
        # Execute original function
        result = f(*args, **kwargs)
        
        # Consume quota if successful
        if isinstance(result, tuple) and len(result) == 2:
            response, status_code = result
            if status_code == 200 and isinstance(response, dict):
                prediction_id = response.get('prediction_id')
                quota_manager.consume_quota(user_id, file_size, prediction_id)
        
        return result
    
    return decorated_function