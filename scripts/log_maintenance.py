#!/usr/bin/env python3
"""
Log Maintenance Script for NightScan

This script provides automated log maintenance operations including:
- Scheduled log rotation and cleanup
- Disk space monitoring with alerts
- Health checks and status reports
- Emergency cleanup procedures
- Cron-job integration

Usage:
    # Daily maintenance (recommended cron job)
    python scripts/log_maintenance.py --daily

    # Check health status
    python scripts/log_maintenance.py --status

    # Emergency cleanup
    python scripts/log_maintenance.py --emergency

    # Validate configuration
    python scripts/log_maintenance.py --validate

    # Monitor continuously
    python scripts/log_maintenance.py --monitor --interval 300

    # Custom retention period
    python scripts/log_maintenance.py --cleanup --retention-days 14

Cron job examples:
    # Daily maintenance at 2 AM
    0 2 * * * /usr/bin/python3 /path/to/scripts/log_maintenance.py --daily --quiet

    # Hourly health check
    0 * * * * /usr/bin/python3 /path/to/scripts/log_maintenance.py --status --quiet

    # Emergency check every 15 minutes
    */15 * * * * /usr/bin/python3 /path/to/scripts/log_maintenance.py --emergency-check --quiet
"""

import os
import sys
import time
import json
import argparse
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from log_rotation_manager import get_log_rotation_manager, LogRotationManager
from log_utils import setup_environment_logging


class LogMaintenanceScript:
    """Automated log maintenance operations."""
    
    def __init__(self, quiet: bool = False, log_file: Optional[str] = None):
        """Initialize maintenance script.
        
        Args:
            quiet: Suppress console output except errors
            log_file: Optional log file for script operations
        """
        self.quiet = quiet
        self.log_file = log_file
        self.manager = get_log_rotation_manager()
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup logging if log_file provided
        if log_file:
            setup_environment_logging()
            
        self._log("Log maintenance script initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self._log(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def _log(self, message: str, level: str = "info"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        if not self.quiet or level == "error":
            print(formatted_message)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted_message + "\n")
    
    def daily_maintenance(self) -> int:
        """Perform daily maintenance tasks.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        self._log("Starting daily maintenance")
        
        try:
            # 1. Setup environment logging
            self._log("Setting up environment logging")
            self.manager.setup_environment_logging()
            
            # 2. Perform maintenance
            self._log("Performing log rotation maintenance")
            status = self.manager.perform_maintenance(force=True)
            
            if status.success:
                self._log(f"✅ Daily maintenance completed successfully")
                self._log(f"   Files processed: {status.files_processed}")
                self._log(f"   Space freed: {status.space_freed_mb:.2f}MB")
                self._log(f"   Duration: {status.duration_seconds:.2f}s")
                
                if status.warnings:
                    for warning in status.warnings:
                        self._log(f"⚠️  Warning: {warning}")
                
                return 0
            else:
                self._log("❌ Daily maintenance completed with errors", "error")
                for error in status.errors:
                    self._log(f"   Error: {error}", "error")
                return 1
                
        except Exception as e:
            self._log(f"❌ Daily maintenance failed: {e}", "error")
            return 1
    
    def check_status(self) -> int:
        """Check system health status.
        
        Returns:
            Exit code (0 for healthy, 1 for degraded, 2 for unhealthy)
        """
        self._log("Checking log rotation system health")
        
        try:
            health = self.manager.get_health_status()
            
            status = health.get('overall_status', 'unknown')
            
            if status == 'healthy':
                self._log("✅ System is healthy")
                
                # Show summary if not quiet
                if not self.quiet:
                    disk_status = health.get('disk_status', {})
                    self._log(f"   Disk usage: {disk_status.get('disk_usage_percent', 0):.1f}%")
                    self._log(f"   Free space: {disk_status.get('disk_free_gb', 0):.1f}GB")
                    self._log(f"   Log dir size: {disk_status.get('log_dir_size_mb', 0):.1f}MB")
                
                return 0
                
            elif status == 'degraded':
                self._log("⚠️  System is degraded")
                
                # Show issues
                if health.get('disk_status', {}).get('alert'):
                    alert = health['disk_status']['alert']
                    self._log(f"   Disk alert: {alert['severity']} - {alert['current_usage_percent']:.1f}% used")
                
                return 1
                
            else:
                self._log("❌ System is unhealthy", "error")
                
                if 'error' in health:
                    self._log(f"   Error: {health['error']}", "error")
                
                return 2
                
        except Exception as e:
            self._log(f"❌ Health check failed: {e}", "error")
            return 2
    
    def emergency_cleanup(self) -> int:
        """Perform emergency cleanup.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        self._log("Performing emergency cleanup")
        
        try:
            status = self.manager.force_emergency_cleanup()
            
            self._log(f"✅ Emergency cleanup completed")
            self._log(f"   Files processed: {status.get('files_processed', 0)}")
            self._log(f"   Space freed: {status.get('space_freed_mb', 0):.2f}MB")
            self._log(f"   Duration: {status.get('duration_seconds', 0):.2f}s")
            
            return 0
            
        except Exception as e:
            self._log(f"❌ Emergency cleanup failed: {e}", "error")
            return 1
    
    def emergency_check(self) -> int:
        """Check if emergency cleanup is needed.
        
        Returns:
            Exit code (0 for no action needed, 1 for cleanup recommended, 2 for immediate action required)
        """
        try:
            health = self.manager.get_health_status()
            disk_status = health.get('disk_status', {})
            
            if disk_status.get('error'):
                self._log(f"❌ Cannot check disk status: {disk_status['error']}", "error")
                return 2
            
            usage_percent = disk_status.get('disk_usage_percent', 0)
            
            if usage_percent >= 95:
                self._log(f"🚨 CRITICAL: Disk usage at {usage_percent:.1f}% - immediate action required!", "error")
                # Automatically perform emergency cleanup
                return self.emergency_cleanup()
                
            elif usage_percent >= 90:
                self._log(f"⚠️  WARNING: Disk usage at {usage_percent:.1f}% - cleanup recommended")
                return 1
                
            elif usage_percent >= 80:
                self._log(f"ℹ️  INFO: Disk usage at {usage_percent:.1f}% - monitoring")
                return 0
                
            else:
                self._log(f"✅ Disk usage normal: {usage_percent:.1f}%")
                return 0
                
        except Exception as e:
            self._log(f"❌ Emergency check failed: {e}", "error")
            return 2
    
    def validate_configuration(self) -> int:
        """Validate log rotation configuration.
        
        Returns:
            Exit code (0 for valid, 1 for warnings, 2 for errors)
        """
        self._log("Validating log rotation configuration")
        
        try:
            validation = self.manager.validate_configuration()
            
            if validation['valid']:
                self._log("✅ Configuration is valid")
                
                if validation['warnings']:
                    for warning in validation['warnings']:
                        self._log(f"⚠️  Warning: {warning}")
                    return 1
                
                return 0
                
            else:
                self._log("❌ Configuration has errors", "error")
                
                for error in validation['errors']:
                    self._log(f"   Error: {error}", "error")
                
                return 2
                
        except Exception as e:
            self._log(f"❌ Configuration validation failed: {e}", "error")
            return 2
    
    def cleanup_with_retention(self, retention_days: int) -> int:
        """Perform cleanup with custom retention period.
        
        Args:
            retention_days: Number of days to retain logs
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        self._log(f"Performing cleanup with {retention_days} day retention")
        
        try:
            from log_utils import cleanup_old_logs
            
            log_dir = str(self.manager.log_config.log_dir)
            result = cleanup_old_logs(log_dir, retention_days)
            
            if result.get('error'):
                self._log(f"❌ Cleanup failed: {result['error']}", "error")
                return 1
            
            files_removed = result.get('removed_files', 0)
            space_freed = result.get('total_size_freed_mb', 0)
            
            self._log(f"✅ Cleanup completed")
            self._log(f"   Files removed: {files_removed}")
            self._log(f"   Space freed: {space_freed:.2f}MB")
            
            return 0
            
        except Exception as e:
            self._log(f"❌ Cleanup failed: {e}", "error")
            return 1
    
    def monitor_continuously(self, interval: int = 300) -> int:
        """Monitor log system continuously.
        
        Args:
            interval: Check interval in seconds
            
        Returns:
            Exit code (0 for normal shutdown, 1 for error)
        """
        self._log(f"Starting continuous monitoring (interval: {interval}s)")
        
        try:
            while self.running:
                # Perform health check
                health = self.manager.get_health_status()
                status = health.get('overall_status', 'unknown')
                
                if status != 'healthy':
                    self._log(f"⚠️  Health status: {status}")
                    
                    # Check for disk space alerts
                    disk_status = health.get('disk_status', {})
                    if disk_status.get('alert'):
                        alert = disk_status['alert']
                        usage = alert['current_usage_percent']
                        
                        if alert['severity'] == 'emergency':
                            self._log(f"🚨 EMERGENCY: Disk {usage:.1f}% full - performing cleanup!", "error")
                            self.emergency_cleanup()
                        elif alert['severity'] == 'critical':
                            self._log(f"🔥 CRITICAL: Disk {usage:.1f}% full - cleanup needed!", "error")
                        else:
                            self._log(f"⚠️  WARNING: Disk {usage:.1f}% full")
                
                # Wait for next check
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)
            
            self._log("Monitoring stopped")
            return 0
            
        except Exception as e:
            self._log(f"❌ Monitoring failed: {e}", "error")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NightScan Log Maintenance Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Operation modes
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument('--daily', action='store_true',
                                help='Perform daily maintenance')
    operation_group.add_argument('--status', action='store_true',
                                help='Check system health status')
    operation_group.add_argument('--emergency', action='store_true',
                                help='Perform emergency cleanup')
    operation_group.add_argument('--emergency-check', action='store_true',
                                help='Check if emergency cleanup is needed')
    operation_group.add_argument('--validate', action='store_true',
                                help='Validate configuration')
    operation_group.add_argument('--cleanup', action='store_true',
                                help='Perform cleanup with custom retention')
    operation_group.add_argument('--monitor', action='store_true',
                                help='Monitor continuously')
    
    # Options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress console output except errors')
    parser.add_argument('--log-file', 
                       help='Log script operations to file')
    parser.add_argument('--retention-days', type=int, default=30,
                       help='Retention period in days (for --cleanup)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Monitoring interval in seconds (for --monitor)')
    parser.add_argument('--environment',
                       help='Override environment (dev/staging/production/raspberry_pi)')
    
    args = parser.parse_args()
    
    # Set environment if specified
    if args.environment:
        os.environ['NIGHTSCAN_ENV'] = args.environment
    
    # Initialize script
    script = LogMaintenanceScript(quiet=args.quiet, log_file=args.log_file)
    
    # Execute requested operation
    try:
        if args.daily:
            exit_code = script.daily_maintenance()
        elif args.status:
            exit_code = script.check_status()
        elif args.emergency:
            exit_code = script.emergency_cleanup()
        elif args.emergency_check:
            exit_code = script.emergency_check()
        elif args.validate:
            exit_code = script.validate_configuration()
        elif args.cleanup:
            exit_code = script.cleanup_with_retention(args.retention_days)
        elif args.monitor:
            exit_code = script.monitor_continuously(args.interval)
        else:
            parser.error("No operation specified")
            
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        script._log("Interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        script._log(f"❌ Unexpected error: {e}", "error")
        sys.exit(1)


if __name__ == "__main__":
    main()