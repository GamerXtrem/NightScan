#!/usr/bin/env python3
"""
Centralized Log Rotation Manager for NightScan

This module provides comprehensive log rotation management including:
- Automated log rotation and cleanup
- Disk space monitoring and alerts
- Configuration management for multiple environments
- Health checks and status reporting
- Emergency cleanup procedures

Usage:
    from log_rotation_manager import LogRotationManager
    
    # Initialize manager
    manager = LogRotationManager()
    
    # Setup logging for current environment
    manager.setup_environment_logging()
    
    # Perform maintenance
    manager.perform_maintenance()
    
    # Check health
    status = manager.get_health_status()
"""

import os
import sys
import time
import json
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# Import our enhanced logging utilities
from log_utils import (
    LogRotationConfig, setup_logging, setup_specialized_loggers,
    setup_environment_logging, cleanup_old_logs, get_disk_usage,
    get_log_file_info
)
from config import get_config, initialize_logging_from_config


@dataclass
class LogRotationStatus:
    """Status information for log rotation operations."""
    timestamp: str
    operation: str
    success: bool
    files_processed: int
    space_freed_mb: float
    errors: List[str]
    warnings: List[str]
    duration_seconds: float


@dataclass
class DiskSpaceAlert:
    """Disk space alert information."""
    timestamp: str
    severity: str  # warning, critical, emergency
    current_usage_percent: float
    available_space_gb: float
    threshold_percent: float
    log_dir_size_mb: float
    recommended_action: str


class LogRotationManager:
    """Centralized log rotation and management system."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the log rotation manager.
        
        Args:
            config_file: Optional path to configuration file
        """
        # Load NightScan configuration
        self.config = get_config()
        self.environment = self.config.environment
        
        # Get logging configuration for current environment
        self.log_config = self.config.logging.get_log_rotation_config(self.environment)
        
        # Status tracking
        self.rotation_history: List[LogRotationStatus] = []
        self.disk_alerts: List[DiskSpaceAlert] = []
        self.last_maintenance_time: Optional[datetime] = None
        self.specialized_loggers: Dict[str, logging.Logger] = {}
        
        # Configuration
        self.disk_warning_threshold = 80.0  # percent
        self.disk_critical_threshold = 90.0  # percent
        self.disk_emergency_threshold = 95.0  # percent
        self.maintenance_interval_hours = 24
        self.max_history_entries = 100
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize logger for this manager
        self.logger = logging.getLogger('nightscan.log_rotation_manager')
    
    def setup_environment_logging(self) -> Dict[str, logging.Logger]:
        """Setup logging for the current environment using centralized configuration."""
        try:
            with self._lock:
                self.logger.info(f"Setting up logging for environment: {self.environment}")
                
                # Use config-based initialization
                self.specialized_loggers = initialize_logging_from_config(self.config)
                
                self.logger.info("Environment logging setup completed successfully")
                return self.specialized_loggers
                
        except Exception as e:
            self.logger.error(f"Failed to setup environment logging: {e}")
            raise
    
    def perform_maintenance(self, force: bool = False) -> LogRotationStatus:
        """Perform log rotation maintenance tasks.
        
        Args:
            force: Force maintenance even if interval hasn't elapsed
            
        Returns:
            LogRotationStatus with operation results
        """
        start_time = time.time()
        operation_start = datetime.now()
        
        try:
            with self._lock:
                # Check if maintenance is needed
                if not force and self._is_maintenance_recent():
                    self.logger.info("Maintenance was performed recently, skipping")
                    return LogRotationStatus(
                        timestamp=operation_start.isoformat(),
                        operation="maintenance_skipped",
                        success=True,
                        files_processed=0,
                        space_freed_mb=0.0,
                        errors=[],
                        warnings=["Maintenance skipped - performed recently"],
                        duration_seconds=time.time() - start_time
                    )
                
                self.logger.info("Starting log rotation maintenance")
                
                errors = []
                warnings = []
                files_processed = 0
                space_freed_mb = 0.0
                
                # 1. Check disk space and generate alerts
                disk_status = self._check_disk_space()
                if disk_status.get('alert'):
                    warnings.append(f"Disk space alert: {disk_status['alert']['severity']}")
                
                # 2. Cleanup old log files
                try:
                    cleanup_result = cleanup_old_logs(
                        str(self.log_config.log_dir),
                        self.log_config.retention_days
                    )
                    
                    if cleanup_result.get('error'):
                        errors.append(f"Cleanup error: {cleanup_result['error']}")
                    else:
                        files_processed += cleanup_result.get('removed_files', 0)
                        space_freed_mb += cleanup_result.get('total_size_freed_mb', 0.0)
                        self.logger.info(f"Cleaned up {files_processed} old log files, freed {space_freed_mb:.2f}MB")
                
                except Exception as e:
                    errors.append(f"Log cleanup failed: {str(e)}")
                
                # 3. Rotate large log files if needed
                try:
                    rotation_result = self._rotate_large_files()
                    files_processed += rotation_result.get('files_rotated', 0)
                    space_freed_mb += rotation_result.get('space_freed_mb', 0.0)
                    
                    if rotation_result.get('errors'):
                        errors.extend(rotation_result['errors'])
                    
                except Exception as e:
                    errors.append(f"Log rotation failed: {str(e)}")
                
                # 4. Emergency cleanup if disk space is critical
                if disk_status.get('usage_percent', 0) > self.disk_emergency_threshold:
                    try:
                        emergency_result = self._emergency_cleanup()
                        files_processed += emergency_result.get('files_processed', 0)
                        space_freed_mb += emergency_result.get('space_freed_mb', 0.0)
                        warnings.append("Emergency cleanup performed due to low disk space")
                        
                    except Exception as e:
                        errors.append(f"Emergency cleanup failed: {str(e)}")
                
                # Record maintenance completion
                self.last_maintenance_time = operation_start
                
                # Create status record
                status = LogRotationStatus(
                    timestamp=operation_start.isoformat(),
                    operation="maintenance",
                    success=len(errors) == 0,
                    files_processed=files_processed,
                    space_freed_mb=space_freed_mb,
                    errors=errors,
                    warnings=warnings,
                    duration_seconds=time.time() - start_time
                )
                
                # Add to history
                self._add_to_history(status)
                
                if errors:
                    self.logger.error(f"Maintenance completed with errors: {errors}")
                else:
                    self.logger.info(f"Maintenance completed successfully - processed {files_processed} files, freed {space_freed_mb:.2f}MB")
                
                return status
                
        except Exception as e:
            self.logger.error(f"Maintenance operation failed: {e}")
            status = LogRotationStatus(
                timestamp=operation_start.isoformat(),
                operation="maintenance",
                success=False,
                files_processed=0,
                space_freed_mb=0.0,
                errors=[f"Maintenance failed: {str(e)}"],
                warnings=[],
                duration_seconds=time.time() - start_time
            )
            self._add_to_history(status)
            return status
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of log rotation system."""
        try:
            with self._lock:
                # Get disk usage
                disk_info = get_disk_usage(str(self.log_config.log_dir))
                
                # Check log file status
                log_files_status = self._check_log_files_status()
                
                # Get recent maintenance history
                recent_history = self.rotation_history[-10:] if self.rotation_history else []
                
                # Determine overall health
                is_healthy = (
                    not disk_info.get('error') and
                    disk_info.get('disk_usage_percent', 100) < self.disk_warning_threshold and
                    len([h for h in recent_history if not h.success]) == 0 and
                    log_files_status.get('all_accessible', False)
                )
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'overall_status': 'healthy' if is_healthy else 'degraded',
                    'environment': self.environment,
                    'log_config': {
                        'log_dir': str(self.log_config.log_dir),
                        'max_file_size_mb': self.log_config.max_file_size / (1024 * 1024),
                        'backup_count': self.log_config.backup_count,
                        'retention_days': self.log_config.retention_days,
                        'compress_backups': self.log_config.compress_backups
                    },
                    'disk_status': disk_info,
                    'log_files': log_files_status,
                    'maintenance': {
                        'last_run': self.last_maintenance_time.isoformat() if self.last_maintenance_time else None,
                        'interval_hours': self.maintenance_interval_hours,
                        'is_due': not self._is_maintenance_recent()
                    },
                    'recent_operations': [asdict(h) for h in recent_history],
                    'active_alerts': [asdict(alert) for alert in self.disk_alerts[-5:]]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get health status: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'unhealthy',
                'error': str(e)
            }
    
    def get_rotation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get log rotation operation history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of rotation status dictionaries
        """
        with self._lock:
            return [asdict(status) for status in self.rotation_history[-limit:]]
    
    def force_emergency_cleanup(self) -> LogRotationStatus:
        """Force emergency cleanup regardless of disk space."""
        self.logger.warning("Forcing emergency cleanup")
        return self._emergency_cleanup()
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate log rotation configuration."""
        errors = []
        warnings = []
        
        # Check log directory
        if not self.log_config.log_dir.exists():
            try:
                self.log_config.log_dir.mkdir(parents=True, exist_ok=True)
                warnings.append(f"Created log directory: {self.log_config.log_dir}")
            except Exception as e:
                errors.append(f"Cannot create log directory {self.log_config.log_dir}: {e}")
        
        # Check write permissions
        if self.log_config.log_dir.exists():
            test_file = self.log_config.log_dir / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                errors.append(f"No write permission to log directory: {e}")
        
        # Check disk space
        disk_info = get_disk_usage(str(self.log_config.log_dir))
        if disk_info.get('error'):
            errors.append(f"Cannot check disk usage: {disk_info['error']}")
        elif disk_info.get('disk_usage_percent', 0) > self.disk_warning_threshold:
            warnings.append(f"Low disk space: {disk_info['disk_usage_percent']:.1f}% used")
        
        # Check retention settings
        if self.log_config.retention_days < 1:
            warnings.append("Retention period is less than 1 day")
        elif self.log_config.retention_days > 365:
            warnings.append("Retention period is very long (>1 year)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config': {
                'log_dir': str(self.log_config.log_dir),
                'max_file_size_mb': self.log_config.max_file_size / (1024 * 1024),
                'backup_count': self.log_config.backup_count,
                'retention_days': self.log_config.retention_days,
                'compress_backups': self.log_config.compress_backups,
                'environment': self.environment
            }
        }
    
    def _is_maintenance_recent(self) -> bool:
        """Check if maintenance was performed recently."""
        if not self.last_maintenance_time:
            return False
        
        elapsed = datetime.now() - self.last_maintenance_time
        return elapsed.total_seconds() < (self.maintenance_interval_hours * 3600)
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space and generate alerts if needed."""
        disk_info = get_disk_usage(str(self.log_config.log_dir))
        
        if disk_info.get('error'):
            return {'error': disk_info['error']}
        
        usage_percent = disk_info.get('disk_usage_percent', 0)
        available_gb = disk_info.get('disk_free_gb', 0)
        
        # Determine alert level
        alert = None
        if usage_percent >= self.disk_emergency_threshold:
            alert = DiskSpaceAlert(
                timestamp=datetime.now().isoformat(),
                severity="emergency",
                current_usage_percent=usage_percent,
                available_space_gb=available_gb,
                threshold_percent=self.disk_emergency_threshold,
                log_dir_size_mb=disk_info.get('log_dir_size_mb', 0),
                recommended_action="Immediate cleanup required - system may become unstable"
            )
        elif usage_percent >= self.disk_critical_threshold:
            alert = DiskSpaceAlert(
                timestamp=datetime.now().isoformat(),
                severity="critical",
                current_usage_percent=usage_percent,
                available_space_gb=available_gb,
                threshold_percent=self.disk_critical_threshold,
                log_dir_size_mb=disk_info.get('log_dir_size_mb', 0),
                recommended_action="Cleanup required within 24 hours"
            )
        elif usage_percent >= self.disk_warning_threshold:
            alert = DiskSpaceAlert(
                timestamp=datetime.now().isoformat(),
                severity="warning",
                current_usage_percent=usage_percent,
                available_space_gb=available_gb,
                threshold_percent=self.disk_warning_threshold,
                log_dir_size_mb=disk_info.get('log_dir_size_mb', 0),
                recommended_action="Schedule cleanup within next week"
            )
        
        if alert:
            with self._lock:
                self.disk_alerts.append(alert)
                # Keep only recent alerts
                self.disk_alerts = self.disk_alerts[-20:]
            
            self.logger.warning(f"Disk space alert: {alert.severity} - {usage_percent:.1f}% used")
        
        result = dict(disk_info)
        if alert:
            result['alert'] = asdict(alert)
        
        return result
    
    def _rotate_large_files(self) -> Dict[str, Any]:
        """Rotate log files that exceed size limits."""
        files_rotated = 0
        space_freed_mb = 0.0
        errors = []
        
        try:
            for log_file in self.log_config.log_dir.glob("*.log"):
                try:
                    file_info = get_log_file_info(str(log_file))
                    
                    if file_info.get('error'):
                        continue
                    
                    # Check if file needs rotation
                    if file_info.get('size_bytes', 0) > self.log_config.max_file_size:
                        # Create backup name
                        backup_name = f"{log_file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        backup_path = log_file.parent / backup_name
                        
                        # Move current file to backup
                        shutil.move(str(log_file), str(backup_path))
                        
                        # Compress if enabled
                        if self.log_config.compress_backups:
                            import gzip
                            with open(backup_path, 'rb') as f_in:
                                with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            backup_path.unlink()
                            space_freed_mb += (file_info['size_bytes'] - backup_path.with_suffix('.gz').stat().st_size) / (1024 * 1024)
                        
                        files_rotated += 1
                        self.logger.info(f"Rotated large log file: {log_file}")
                        
                except Exception as e:
                    errors.append(f"Failed to rotate {log_file}: {str(e)}")
                    
        except Exception as e:
            errors.append(f"Log file rotation failed: {str(e)}")
        
        return {
            'files_rotated': files_rotated,
            'space_freed_mb': space_freed_mb,
            'errors': errors
        }
    
    def _emergency_cleanup(self) -> Dict[str, Any]:
        """Perform emergency cleanup to free disk space."""
        start_time = time.time()
        files_processed = 0
        space_freed_mb = 0.0
        
        self.logger.warning("Performing emergency cleanup")
        
        try:
            # 1. Remove compressed log files older than 7 days
            cutoff_time = time.time() - (7 * 24 * 3600)
            
            for compressed_file in self.log_config.log_dir.glob("*.gz"):
                if compressed_file.stat().st_mtime < cutoff_time:
                    size_mb = compressed_file.stat().st_size / (1024 * 1024)
                    compressed_file.unlink()
                    files_processed += 1
                    space_freed_mb += size_mb
            
            # 2. Truncate current log files that are too large
            for log_file in self.log_config.log_dir.glob("*.log"):
                file_info = get_log_file_info(str(log_file))
                
                if file_info.get('size_mb', 0) > 50:  # Truncate files > 50MB
                    # Keep only the last 10MB of the file
                    with open(log_file, 'rb') as f:
                        f.seek(-10 * 1024 * 1024, 2)  # Seek to 10MB from end
                        data = f.read()
                    
                    with open(log_file, 'wb') as f:
                        f.write(data)
                    
                    files_processed += 1
                    space_freed_mb += (file_info['size_mb'] - 10)
            
            self.logger.warning(f"Emergency cleanup completed: {files_processed} files processed, {space_freed_mb:.2f}MB freed")
            
        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")
            raise
        
        return {
            'files_processed': files_processed,
            'space_freed_mb': space_freed_mb,
            'duration_seconds': time.time() - start_time
        }
    
    def _check_log_files_status(self) -> Dict[str, Any]:
        """Check status of all log files."""
        log_files = []
        all_accessible = True
        total_size_mb = 0.0
        
        try:
            for log_file in self.log_config.log_dir.glob("*.log*"):
                file_info = get_log_file_info(str(log_file))
                
                if file_info.get('error'):
                    all_accessible = False
                else:
                    total_size_mb += file_info.get('size_mb', 0)
                
                log_files.append({
                    'name': log_file.name,
                    'path': str(log_file),
                    'info': file_info
                })
            
            return {
                'all_accessible': all_accessible,
                'total_files': len(log_files),
                'total_size_mb': round(total_size_mb, 2),
                'files': log_files
            }
            
        except Exception as e:
            return {
                'all_accessible': False,
                'error': str(e)
            }
    
    def _add_to_history(self, status: LogRotationStatus):
        """Add operation status to history."""
        with self._lock:
            self.rotation_history.append(status)
            # Keep only recent history
            if len(self.rotation_history) > self.max_history_entries:
                self.rotation_history = self.rotation_history[-self.max_history_entries:]


# Singleton instance for global access
_log_rotation_manager: Optional[LogRotationManager] = None


def get_log_rotation_manager() -> LogRotationManager:
    """Get the global log rotation manager instance."""
    global _log_rotation_manager
    
    if _log_rotation_manager is None:
        _log_rotation_manager = LogRotationManager()
    
    return _log_rotation_manager


def initialize_log_rotation_manager(config_file: Optional[str] = None) -> LogRotationManager:
    """Initialize the global log rotation manager."""
    global _log_rotation_manager
    
    _log_rotation_manager = LogRotationManager(config_file)
    return _log_rotation_manager


if __name__ == "__main__":
    # Command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="NightScan Log Rotation Manager")
    parser.add_argument('command', choices=['setup', 'maintenance', 'status', 'cleanup', 'validate'],
                       help='Command to execute')
    parser.add_argument('--force', action='store_true', help='Force operation')
    parser.add_argument('--environment', help='Environment name')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = get_log_rotation_manager()
    
    if args.environment:
        os.environ['NIGHTSCAN_ENV'] = args.environment
    
    if args.command == 'setup':
        print("Setting up environment logging...")
        loggers = manager.setup_environment_logging()
        print(f"Setup completed. Specialized loggers: {list(loggers.keys())}")
        
    elif args.command == 'maintenance':
        print("Performing maintenance...")
        status = manager.perform_maintenance(force=args.force)
        print(f"Maintenance {'successful' if status.success else 'failed'}")
        print(f"Files processed: {status.files_processed}")
        print(f"Space freed: {status.space_freed_mb:.2f}MB")
        if status.errors:
            print(f"Errors: {status.errors}")
        
    elif args.command == 'status':
        health = manager.get_health_status()
        print(json.dumps(health, indent=2))
        
    elif args.command == 'cleanup':
        print("Performing emergency cleanup...")
        status = manager.force_emergency_cleanup()
        print(f"Emergency cleanup completed")
        
    elif args.command == 'validate':
        validation = manager.validate_configuration()
        print(f"Configuration {'valid' if validation['valid'] else 'invalid'}")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")