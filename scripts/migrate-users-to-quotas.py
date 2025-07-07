#!/usr/bin/env python3
"""
Migration script for NightScan quota system
Migrates existing users to free plan and initializes quota tracking.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import app, db
from quota_manager import get_quota_manager, PlanType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_existing_users():
    """Migrate all existing users to the quota system"""
    with app.app_context():
        try:
            # Import models
            from web.app import User, UserPlan, PlanFeatures, QuotaUsage
            
            # Create all tables
            db.create_all()
            
            # Check if default plans exist, create if not
            free_plan = PlanFeatures.query.filter_by(plan_type='free').first()
            if not free_plan:
                logger.info("Creating default plans...")
                
                # Create default plans
                plans = [
                    PlanFeatures(
                        plan_type='free',
                        plan_name='Plan Gratuit',
                        monthly_quota=600,
                        max_file_size_mb=50,
                        max_concurrent_uploads=1,
                        priority_queue=False,
                        advanced_analytics=False,
                        api_access=False,
                        email_support=False,
                        phone_support=False,
                        price_monthly_cents=0
                    ),
                    PlanFeatures(
                        plan_type='premium',
                        plan_name='Plan Premium',
                        monthly_quota=3000,
                        max_file_size_mb=50,
                        max_concurrent_uploads=1,
                        priority_queue=False,
                        advanced_analytics=False,
                        api_access=False,
                        email_support=False,
                        phone_support=False,
                        price_monthly_cents=1990
                    ),
                    PlanFeatures(
                        plan_type='enterprise',
                        plan_name='Plan Entreprise',
                        monthly_quota=100000,
                        max_file_size_mb=50,
                        max_concurrent_uploads=1,
                        priority_queue=False,
                        advanced_analytics=False,
                        api_access=False,
                        email_support=False,
                        phone_support=False,
                        price_monthly_cents=9990
                    )
                ]
                
                for plan in plans:
                    db.session.add(plan)
                
                db.session.commit()
                logger.info("Created default plans")
            
            # Get all users without a plan
            users_without_plan = db.session.query(User).outerjoin(UserPlan).filter(UserPlan.user_id.is_(None)).all()
            
            logger.info(f"Found {len(users_without_plan)} users without plans")
            
            migrated_count = 0
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            for user in users_without_plan:
                try:
                    # Create user plan
                    user_plan = UserPlan(
                        user_id=user.id,
                        plan_type='free',
                        status='active'
                    )
                    db.session.add(user_plan)
                    
                    # Initialize current month quota
                    quota_usage = QuotaUsage(
                        user_id=user.id,
                        month=current_month,
                        year=current_year,
                        reset_date=datetime(current_year + (1 if current_month == 12 else 0),
                                          (current_month % 12) + 1, 1)
                    )
                    db.session.add(quota_usage)
                    
                    migrated_count += 1
                    
                    if migrated_count % 100 == 0:
                        db.session.commit()
                        logger.info(f"Migrated {migrated_count} users...")
                
                except Exception as e:
                    logger.error(f"Failed to migrate user {user.id}: {e}")
                    db.session.rollback()
                    continue
            
            # Final commit
            db.session.commit()
            
            logger.info(f"Successfully migrated {migrated_count} users to quota system")
            
            # Update existing predictions count for current month
            update_existing_predictions_count()
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            db.session.rollback()
            raise

def update_existing_predictions_count():
    """Update quota usage based on existing predictions for current month"""
    with app.app_context():
        try:
            from web.app import User, Prediction, QuotaUsage
            
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            # Get predictions for current month
            month_start = datetime(current_year, current_month, 1)
            if current_month == 12:
                month_end = datetime(current_year + 1, 1, 1)
            else:
                month_end = datetime(current_year, current_month + 1, 1)
            
            logger.info(f"Updating quota usage for period {month_start} to {month_end}")
            
            # Count predictions per user for current month
            prediction_counts = db.session.query(
                Prediction.user_id,
                db.func.count(Prediction.id).label('count'),
                db.func.sum(Prediction.file_size).label('total_size'),
                db.func.count(db.case((Prediction.result.isnot(None), 1))).label('successful'),
                db.func.count(db.case((Prediction.result.is_(None), 1))).label('failed')
            ).filter(
                Prediction.created_at >= month_start,
                Prediction.created_at < month_end
            ).group_by(Prediction.user_id).all()
            
            logger.info(f"Found prediction data for {len(prediction_counts)} users")
            
            updated_count = 0
            for user_id, count, total_size, successful, failed in prediction_counts:
                quota_usage = QuotaUsage.query.filter_by(
                    user_id=user_id, 
                    month=current_month, 
                    year=current_year
                ).first()
                
                if quota_usage:
                    quota_usage.prediction_count = count
                    quota_usage.total_file_size_bytes = total_size or 0
                    quota_usage.successful_predictions = successful or 0
                    quota_usage.failed_predictions = failed or 0
                    quota_usage.updated_at = datetime.utcnow()
                    
                    updated_count += 1
                    
                    if updated_count % 50 == 0:
                        db.session.commit()
                        logger.info(f"Updated {updated_count} quota records...")
            
            db.session.commit()
            logger.info(f"Updated quota usage for {updated_count} users")
            
        except Exception as e:
            logger.error(f"Failed to update prediction counts: {e}")
            db.session.rollback()
            raise

def verify_migration():
    """Verify the migration was successful"""
    with app.app_context():
        try:
            from web.app import User, UserPlan, QuotaUsage, PlanFeatures
            
            total_users = User.query.count()
            users_with_plans = UserPlan.query.count()
            quota_records = QuotaUsage.query.count()
            plan_count = PlanFeatures.query.count()
            
            logger.info("Migration verification:")
            logger.info(f"  Total users: {total_users}")
            logger.info(f"  Users with plans: {users_with_plans}")
            logger.info(f"  Quota records: {quota_records}")
            logger.info(f"  Available plans: {plan_count}")
            
            if users_with_plans != total_users:
                logger.warning(f"Missing plans for {total_users - users_with_plans} users")
                return False
            
            if quota_records < users_with_plans:
                logger.warning(f"Missing quota records for {users_with_plans - quota_records} users")
                return False
            
            logger.info("✅ Migration verification successful!")
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

def show_stats():
    """Show current quota system statistics"""
    with app.app_context():
        try:
            from web.app import User, UserPlan, QuotaUsage, PlanFeatures
            
            # Plan distribution
            plan_stats = db.session.query(
                UserPlan.plan_type,
                db.func.count(UserPlan.user_id).label('count')
            ).group_by(UserPlan.plan_type).all()
            
            logger.info("Plan distribution:")
            for plan_type, count in plan_stats:
                logger.info(f"  {plan_type}: {count} users")
            
            # Usage statistics for current month
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            usage_stats = db.session.query(
                db.func.sum(QuotaUsage.prediction_count).label('total_predictions'),
                db.func.avg(QuotaUsage.prediction_count).label('avg_predictions'),
                db.func.max(QuotaUsage.prediction_count).label('max_predictions'),
                db.func.sum(QuotaUsage.total_file_size_bytes).label('total_size')
            ).filter(
                QuotaUsage.month == current_month,
                QuotaUsage.year == current_year
            ).first()
            
            if usage_stats and usage_stats.total_predictions:
                logger.info(f"Current month usage:")
                logger.info(f"  Total predictions: {usage_stats.total_predictions}")
                logger.info(f"  Average per user: {usage_stats.avg_predictions:.1f}")
                logger.info(f"  Maximum by one user: {usage_stats.max_predictions}")
                logger.info(f"  Total data processed: {(usage_stats.total_size or 0) / (1024*1024):.1f} MB")
            else:
                logger.info("No usage data for current month yet")
            
        except Exception as e:
            logger.error(f"Failed to show stats: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting NightScan quota system migration...")
    
    try:
        # Run migration
        migrate_existing_users()
        
        # Verify migration
        if verify_migration():
            logger.info("✅ Migration completed successfully!")
            
            # Show statistics
            show_stats()
            
            logger.info("\n📋 Next steps:")
            logger.info("1. Test quota system with new predictions")
            logger.info("2. Monitor quota usage in database")
            logger.info("3. Set up plan upgrade workflows")
            logger.info("4. Configure payment integration (future)")
        else:
            logger.error("❌ Migration verification failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        sys.exit(1)