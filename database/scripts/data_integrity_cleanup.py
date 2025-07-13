#!/usr/bin/env python3
"""
NightScan Data Integrity Cleanup Script

This script identifies and optionally fixes data integrity issues before applying
foreign key constraints. It should be run before executing the migration.

Usage:
    python data_integrity_cleanup.py --check        # Dry run, only report issues
    python data_integrity_cleanup.py --fix          # Fix issues (creates backup first)
    python data_integrity_cleanup.py --fix --force  # Fix without confirmation
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from web.app import db, app, User, Prediction, PredictionArchive, Detection
from web.app import PlanFeatures, UserPlan, QuotaUsage, DailyUsageDetails
from web.app import QuotaTransaction, SubscriptionEvent, DataRetentionLog
from web.app import NotificationPreference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIntegrityChecker:
    """Check and fix data integrity issues in the NightScan database."""
    
    def __init__(self, fix_mode: bool = False):
        self.fix_mode = fix_mode
        self.issues_found = []
        self.fixes_applied = []
        
    def check_all(self) -> Dict[str, Any]:
        """Run all integrity checks."""
        logger.info("Starting data integrity check...")
        
        with app.app_context():
            results = {
                'timestamp': datetime.utcnow().isoformat(),
                'mode': 'fix' if self.fix_mode else 'check',
                'issues': [],
                'fixes': [],
                'summary': {}
            }
            
            # Run each check
            self._check_orphaned_predictions()
            self._check_orphaned_archives()
            self._check_invalid_plan_references()
            self._check_invalid_subscription_events()
            self._check_constraint_violations()
            self._check_duplicate_user_plans()
            self._check_missing_user_plans()
            self._check_data_consistency()
            
            results['issues'] = self.issues_found
            results['fixes'] = self.fixes_applied
            results['summary'] = {
                'total_issues': len(self.issues_found),
                'total_fixes': len(self.fixes_applied),
                'categories': self._categorize_issues()
            }
            
            return results
    
    def _categorize_issues(self) -> Dict[str, int]:
        """Categorize issues by type."""
        categories = {}
        for issue in self.issues_found:
            category = issue['type']
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _check_orphaned_predictions(self):
        """Check for predictions with non-existent user_id."""
        logger.info("Checking for orphaned predictions...")
        
        orphaned = db.session.query(Prediction).filter(
            ~Prediction.user_id.in_(db.session.query(User.id))
        ).all()
        
        if orphaned:
            issue = {
                'type': 'orphaned_predictions',
                'count': len(orphaned),
                'description': f'Found {len(orphaned)} predictions with non-existent user_id',
                'records': [{'id': p.id, 'user_id': p.user_id, 'filename': p.filename} for p in orphaned[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                logger.info(f"Archiving {len(orphaned)} orphaned predictions...")
                for pred in orphaned:
                    # Move to archive before deletion
                    archive = PredictionArchive(
                        original_prediction_id=pred.id,
                        user_id=None,  # Set to NULL since user doesn't exist
                        filename=pred.filename,
                        result=pred.result,
                        file_size=pred.file_size,
                        created_at=pred.created_at,
                        archived_by='integrity_cleanup',
                        plan_type_at_archive='unknown'
                    )
                    db.session.add(archive)
                    db.session.delete(pred)
                
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'orphaned_predictions',
                    'action': 'archived_and_deleted',
                    'count': len(orphaned)
                })
    
    def _check_orphaned_archives(self):
        """Check for prediction archives with non-existent user_id."""
        logger.info("Checking for orphaned prediction archives...")
        
        orphaned = db.session.query(PredictionArchive).filter(
            PredictionArchive.user_id.isnot(None),
            ~PredictionArchive.user_id.in_(db.session.query(User.id))
        ).all()
        
        if orphaned:
            issue = {
                'type': 'orphaned_archives',
                'count': len(orphaned),
                'description': f'Found {len(orphaned)} prediction archives with non-existent user_id',
                'records': [{'id': a.id, 'user_id': a.user_id, 'original_id': a.original_prediction_id} for a in orphaned[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                logger.info(f"Updating {len(orphaned)} orphaned archives to NULL user_id...")
                for archive in orphaned:
                    archive.user_id = None
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'orphaned_archives',
                    'action': 'set_user_id_null',
                    'count': len(orphaned)
                })
    
    def _check_invalid_plan_references(self):
        """Check for references to non-existent plan types."""
        logger.info("Checking for invalid plan references...")
        
        # Get all valid plan types
        valid_plans = [p.plan_type for p in db.session.query(PlanFeatures.plan_type).all()]
        
        # Check UserPlan
        invalid_user_plans = db.session.query(UserPlan).filter(
            ~UserPlan.plan_type.in_(valid_plans)
        ).all()
        
        if invalid_user_plans:
            issue = {
                'type': 'invalid_user_plans',
                'count': len(invalid_user_plans),
                'description': f'Found {len(invalid_user_plans)} user plans with invalid plan_type',
                'records': [{'user_id': up.user_id, 'plan_type': up.plan_type} for up in invalid_user_plans[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                # Create a default 'free' plan if it doesn't exist
                free_plan = db.session.query(PlanFeatures).filter_by(plan_type='free').first()
                if not free_plan:
                    free_plan = PlanFeatures(
                        plan_type='free',
                        plan_name='Free Plan',
                        monthly_quota=10,
                        max_file_size_mb=10,
                        data_retention_days=7,
                        price_monthly_cents=0
                    )
                    db.session.add(free_plan)
                    db.session.commit()
                
                logger.info(f"Updating {len(invalid_user_plans)} user plans to free plan...")
                for up in invalid_user_plans:
                    up.plan_type = 'free'
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'invalid_user_plans',
                    'action': 'set_to_free_plan',
                    'count': len(invalid_user_plans)
                })
    
    def _check_invalid_subscription_events(self):
        """Check for subscription events with invalid plan types."""
        logger.info("Checking for invalid subscription events...")
        
        valid_plans = [p.plan_type for p in db.session.query(PlanFeatures.plan_type).all()]
        
        invalid_events = db.session.query(SubscriptionEvent).filter(
            db.or_(
                db.and_(SubscriptionEvent.old_plan_type.isnot(None), 
                       ~SubscriptionEvent.old_plan_type.in_(valid_plans)),
                db.and_(SubscriptionEvent.new_plan_type.isnot(None), 
                       ~SubscriptionEvent.new_plan_type.in_(valid_plans))
            )
        ).all()
        
        if invalid_events:
            issue = {
                'type': 'invalid_subscription_events',
                'count': len(invalid_events),
                'description': f'Found {len(invalid_events)} subscription events with invalid plan types',
                'records': [{'id': e.id, 'old_plan': e.old_plan_type, 'new_plan': e.new_plan_type} for e in invalid_events[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                logger.info(f"Updating {len(invalid_events)} subscription events...")
                for event in invalid_events:
                    if event.old_plan_type not in valid_plans:
                        event.old_plan_type = None
                    if event.new_plan_type not in valid_plans:
                        event.new_plan_type = None
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'invalid_subscription_events',
                    'action': 'set_invalid_plans_null',
                    'count': len(invalid_events)
                })
    
    def _check_constraint_violations(self):
        """Check for data that violates the constraints we want to add."""
        logger.info("Checking for constraint violations...")
        
        # Check QuotaUsage month values
        invalid_months = db.session.query(QuotaUsage).filter(
            db.or_(QuotaUsage.month < 1, QuotaUsage.month > 12)
        ).all()
        
        if invalid_months:
            issue = {
                'type': 'invalid_quota_month',
                'count': len(invalid_months),
                'description': f'Found {len(invalid_months)} quota records with invalid month',
                'records': [{'id': q.id, 'month': q.month} for q in invalid_months[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                for quota in invalid_months:
                    # Set to current month if invalid
                    quota.month = datetime.utcnow().month
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'invalid_quota_month',
                    'action': 'set_to_current_month',
                    'count': len(invalid_months)
                })
        
        # Check Detection confidence values
        invalid_confidence = db.session.query(Detection).filter(
            db.or_(Detection.confidence < 0.0, Detection.confidence > 1.0)
        ).all()
        
        if invalid_confidence:
            issue = {
                'type': 'invalid_detection_confidence',
                'count': len(invalid_confidence),
                'description': f'Found {len(invalid_confidence)} detections with invalid confidence',
                'records': [{'id': d.id, 'confidence': d.confidence} for d in invalid_confidence[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                for detection in invalid_confidence:
                    # Clamp to valid range
                    detection.confidence = max(0.0, min(1.0, detection.confidence))
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'invalid_detection_confidence',
                    'action': 'clamped_to_valid_range',
                    'count': len(invalid_confidence)
                })
        
        # Check negative file sizes
        negative_sizes = db.session.query(Prediction).filter(
            Prediction.file_size < 0
        ).all()
        
        if negative_sizes:
            issue = {
                'type': 'negative_file_sizes',
                'count': len(negative_sizes),
                'description': f'Found {len(negative_sizes)} predictions with negative file size',
                'records': [{'id': p.id, 'file_size': p.file_size} for p in negative_sizes[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                for pred in negative_sizes:
                    pred.file_size = 0
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'negative_file_sizes',
                    'action': 'set_to_zero',
                    'count': len(negative_sizes)
                })
    
    def _check_duplicate_user_plans(self):
        """Check for users with multiple active plans."""
        logger.info("Checking for duplicate user plans...")
        
        # Find users with multiple active plans
        duplicates = db.session.query(
            UserPlan.user_id,
            db.func.count(UserPlan.id).label('count')
        ).filter(
            UserPlan.status == 'active'
        ).group_by(UserPlan.user_id).having(
            db.func.count(UserPlan.id) > 1
        ).all()
        
        if duplicates:
            issue = {
                'type': 'duplicate_active_plans',
                'count': len(duplicates),
                'description': f'Found {len(duplicates)} users with multiple active plans',
                'records': [{'user_id': d.user_id, 'count': d.count} for d in duplicates[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                for dup in duplicates:
                    # Keep the most recent plan, deactivate others
                    user_plans = db.session.query(UserPlan).filter(
                        UserPlan.user_id == dup.user_id,
                        UserPlan.status == 'active'
                    ).order_by(UserPlan.created_at.desc()).all()
                    
                    # Keep the first (most recent), deactivate others
                    for plan in user_plans[1:]:
                        plan.status = 'cancelled'
                        plan.subscription_end = datetime.utcnow()
                
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'duplicate_active_plans',
                    'action': 'kept_most_recent_deactivated_others',
                    'count': len(duplicates)
                })
    
    def _check_missing_user_plans(self):
        """Check for users without any plan assignment."""
        logger.info("Checking for users without plans...")
        
        users_without_plans = db.session.query(User).filter(
            ~User.id.in_(db.session.query(UserPlan.user_id))
        ).all()
        
        if users_without_plans:
            issue = {
                'type': 'users_without_plans',
                'count': len(users_without_plans),
                'description': f'Found {len(users_without_plans)} users without any plan',
                'records': [{'id': u.id, 'username': u.username} for u in users_without_plans[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                # Ensure free plan exists
                free_plan = db.session.query(PlanFeatures).filter_by(plan_type='free').first()
                if not free_plan:
                    free_plan = PlanFeatures(
                        plan_type='free',
                        plan_name='Free Plan',
                        monthly_quota=10,
                        max_file_size_mb=10,
                        data_retention_days=7,
                        price_monthly_cents=0
                    )
                    db.session.add(free_plan)
                    db.session.commit()
                
                # Assign free plan to users
                for user in users_without_plans:
                    user_plan = UserPlan(
                        user_id=user.id,
                        plan_type='free',
                        status='active',
                        auto_renew=True
                    )
                    db.session.add(user_plan)
                
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'users_without_plans',
                    'action': 'assigned_free_plan',
                    'count': len(users_without_plans)
                })
    
    def _check_data_consistency(self):
        """Check for general data consistency issues."""
        logger.info("Checking data consistency...")
        
        # Check for UserPlan dates inconsistency
        invalid_dates = db.session.query(UserPlan).filter(
            UserPlan.subscription_end.isnot(None),
            UserPlan.subscription_end <= UserPlan.subscription_start
        ).all()
        
        if invalid_dates:
            issue = {
                'type': 'invalid_subscription_dates',
                'count': len(invalid_dates),
                'description': f'Found {len(invalid_dates)} plans with end date before start date',
                'records': [{'user_id': p.user_id, 'start': p.subscription_start, 'end': p.subscription_end} 
                           for p in invalid_dates[:10]]
            }
            self.issues_found.append(issue)
            
            if self.fix_mode:
                for plan in invalid_dates:
                    # Set end date to None for active plans
                    if plan.status == 'active':
                        plan.subscription_end = None
                    else:
                        # Set end date to 30 days after start for inactive
                        plan.subscription_end = plan.subscription_start + timedelta(days=30)
                db.session.commit()
                self.fixes_applied.append({
                    'type': 'invalid_subscription_dates',
                    'action': 'fixed_date_logic',
                    'count': len(invalid_dates)
                })


def create_backup(output_file: str = None):
    """Create a backup of current data before fixes."""
    if not output_file:
        output_file = f"nightscan_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    
    logger.info(f"Creating backup to {output_file}...")
    
    with app.app_context():
        backup_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'tables': {}
        }
        
        # Backup relevant tables
        tables = [
            (User, 'users'),
            (Prediction, 'predictions'),
            (PredictionArchive, 'prediction_archives'),
            (UserPlan, 'user_plans'),
            (SubscriptionEvent, 'subscription_events'),
            (QuotaUsage, 'quota_usage'),
            (Detection, 'detections')
        ]
        
        for model, name in tables:
            records = db.session.query(model).all()
            backup_data['tables'][name] = [r.to_dict() for r in records]
        
        with open(output_file, 'w') as f:
            json.dump(backup_data, f, indent=2, default=str)
        
        logger.info(f"Backup completed: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='NightScan Data Integrity Checker')
    parser.add_argument('--check', action='store_true', help='Check for issues (dry run)')
    parser.add_argument('--fix', action='store_true', help='Fix issues (creates backup first)')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--output', help='Output file for report (JSON)')
    parser.add_argument('--backup', help='Backup file name')
    
    args = parser.parse_args()
    
    if not args.check and not args.fix:
        parser.error('Either --check or --fix must be specified')
    
    if args.fix and not args.force:
        response = input("This will modify the database. Create backup and continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Operation cancelled.")
            return
    
    # Create backup if fixing
    if args.fix:
        backup_file = create_backup(args.backup)
        logger.info(f"Backup created: {backup_file}")
    
    # Run integrity check
    checker = DataIntegrityChecker(fix_mode=args.fix)
    results = checker.check_all()
    
    # Display results
    print("\n" + "="*60)
    print("DATA INTEGRITY REPORT")
    print("="*60)
    print(f"Mode: {'FIX' if args.fix else 'CHECK ONLY'}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Issues Found: {results['summary']['total_issues']}")
    if args.fix:
        print(f"Total Fixes Applied: {results['summary']['total_fixes']}")
    
    print("\nIssues by Category:")
    for category, count in results['summary']['categories'].items():
        print(f"  - {category}: {count}")
    
    # Save detailed report
    output_file = args.output or f"integrity_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {output_file}")
    
    if results['summary']['total_issues'] > 0 and not args.fix:
        print("\nTo fix these issues, run with --fix flag")
    elif args.fix:
        print("\nFixes have been applied. Please review the report for details.")


if __name__ == '__main__':
    main()