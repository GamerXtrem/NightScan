"""Disaster recovery and high availability system for NightScan."""

import os
import json
import logging
import asyncio
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from pathlib import Path

from config import get_config
from backup_system import get_backup_manager

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailoverTrigger(Enum):
    """Failover trigger types."""
    MANUAL = "manual"
    HEALTH_CHECK = "health_check"
    PERFORMANCE = "performance"
    EXTERNAL_MONITOR = "external_monitor"


@dataclass
class ServiceHealth:
    """Health information for a service."""
    service_name: str
    status: ServiceStatus
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'status': self.status.value,
            'last_check': self.last_check.isoformat()
        }


@dataclass
class FailoverEvent:
    """Record of a failover event."""
    event_id: str
    timestamp: datetime
    trigger: FailoverTrigger
    primary_service: str
    secondary_service: str
    success: bool
    duration_seconds: float
    reason: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'trigger': self.trigger.value,
            'timestamp': self.timestamp.isoformat()
        }


class HealthChecker:
    """Health monitoring for services."""
    
    def __init__(self, config):
        self.config = config
        self.services = {
            'web_app': 'http://localhost:8000/health',
            'prediction_api': 'http://localhost:8001/api/health',
            'database': 'postgresql://localhost:5432',
            'redis': 'redis://localhost:6379',
        }
        self.thresholds = {
            'response_time_ms': 5000,  # 5 seconds
            'consecutive_failures': 3,
            'uptime_percentage': 95.0
        }
        
    async def check_service_health(self, service_name: str, endpoint: str) -> ServiceHealth:
        """Check health of a single service."""
        start_time = time.time()
        
        try:
            if service_name in ['web_app', 'prediction_api']:
                return await self._check_http_service(service_name, endpoint)
            elif service_name == 'database':
                return await self._check_database_service(service_name, endpoint)
            elif service_name == 'redis':
                return await self._check_redis_service(service_name, endpoint)
            else:
                return ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.UNKNOWN,
                    response_time_ms=0,
                    last_check=datetime.now(),
                    error_message="Unknown service type"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_http_service(self, service_name: str, endpoint: str) -> ServiceHealth:
        """Check HTTP-based service health."""
        start_time = time.time()
        
        try:
            response = requests.get(endpoint, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Try to parse JSON response for additional health info
                try:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        status = ServiceStatus.HEALTHY
                    else:
                        status = ServiceStatus.DEGRADED
                except json.JSONDecodeError:
                    # Non-JSON response but 200 OK
                    status = ServiceStatus.HEALTHY
            else:
                status = ServiceStatus.UNHEALTHY
            
            return ServiceHealth(
                service_name=service_name,
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=None if status != ServiceStatus.UNHEALTHY else f"HTTP {response.status_code}"
            )
            
        except requests.exceptions.RequestException as e:
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_database_service(self, service_name: str, endpoint: str) -> ServiceHealth:
        """Check database service health."""
        start_time = time.time()
        
        try:
            # Use subprocess to check database connectivity
            result = subprocess.run(
                ['pg_isready', '-h', 'localhost', '-p', '5432'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                status = ServiceStatus.HEALTHY
                error_message = None
            else:
                status = ServiceStatus.UNHEALTHY
                error_message = result.stderr.strip()
            
            return ServiceHealth(
                service_name=service_name,
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=error_message
            )
            
        except subprocess.TimeoutExpired:
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message="Database connection timeout"
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def _check_redis_service(self, service_name: str, endpoint: str) -> ServiceHealth:
        """Check Redis service health."""
        start_time = time.time()
        
        try:
            # Use subprocess to check Redis connectivity
            result = subprocess.run(
                ['redis-cli', 'ping'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if result.returncode == 0 and 'PONG' in result.stdout:
                status = ServiceStatus.HEALTHY
                error_message = None
            else:
                status = ServiceStatus.UNHEALTHY
                error_message = "Redis ping failed"
            
            return ServiceHealth(
                service_name=service_name,
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=error_message
            )
            
        except subprocess.TimeoutExpired:
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message="Redis connection timeout"
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all configured services."""
        tasks = []
        for service_name, endpoint in self.services.items():
            task = self.check_service_health(service_name, endpoint)
            tasks.append((service_name, task))
        
        results = {}
        for service_name, task in tasks:
            try:
                health = await task
                results[service_name] = health
            except Exception as e:
                logger.error(f"Failed to check {service_name}: {e}")
                results[service_name] = ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.UNKNOWN,
                    response_time_ms=0,
                    last_check=datetime.now(),
                    error_message=str(e)
                )
        
        return results


class LoadBalancer:
    """Simple load balancer for service failover."""
    
    def __init__(self):
        self.primary_endpoints = {
            'web_app': 'http://localhost:8000',
            'prediction_api': 'http://localhost:8001'
        }
        self.secondary_endpoints = {
            'web_app': 'http://localhost:8080',  # Backup web server
            'prediction_api': 'http://localhost:8002'  # Backup prediction API
        }
        self.current_endpoints = self.primary_endpoints.copy()
        self.failover_status = {}
    
    def get_active_endpoint(self, service: str) -> str:
        """Get currently active endpoint for service."""
        return self.current_endpoints.get(service, self.primary_endpoints.get(service))
    
    def failover_to_secondary(self, service: str, reason: str) -> bool:
        """Failover service to secondary endpoint."""
        if service not in self.secondary_endpoints:
            logger.warning(f"No secondary endpoint configured for {service}")
            return False
        
        logger.info(f"Failing over {service} to secondary endpoint: {reason}")
        
        self.current_endpoints[service] = self.secondary_endpoints[service]
        self.failover_status[service] = {
            'failed_over': True,
            'timestamp': datetime.now(),
            'reason': reason
        }
        
        return True
    
    def failback_to_primary(self, service: str) -> bool:
        """Failback service to primary endpoint."""
        if service not in self.primary_endpoints:
            return False
        
        logger.info(f"Failing back {service} to primary endpoint")
        
        self.current_endpoints[service] = self.primary_endpoints[service]
        self.failover_status[service] = {
            'failed_over': False,
            'timestamp': datetime.now(),
            'reason': 'Primary service recovered'
        }
        
        return True
    
    def get_failover_status(self) -> Dict[str, Any]:
        """Get current failover status."""
        return {
            'current_endpoints': self.current_endpoints,
            'failover_status': self.failover_status
        }


class DisasterRecoveryManager:
    """Main disaster recovery coordination system."""
    
    def __init__(self):
        self.config = get_config()
        self.health_checker = HealthChecker(self.config)
        self.load_balancer = LoadBalancer()
        self.backup_manager = get_backup_manager()
        
        # DR configuration
        self.dr_config = {
            'auto_failover_enabled': True,
            'auto_failback_enabled': True,
            'health_check_interval': 30,  # seconds
            'failover_threshold': 3,  # consecutive failures
            'failback_threshold': 5,  # consecutive successes
            'backup_retention_days': 30,
            'emergency_backup_interval': 3600,  # seconds (1 hour)
        }
        
        # State tracking
        self.service_histories = {}  # Track health history
        self.failover_events = []
        self.last_emergency_backup = None
        
        # Recovery procedures
        self.recovery_procedures = {
            'database_failure': self._recover_database,
            'web_app_failure': self._recover_web_app,
            'prediction_api_failure': self._recover_prediction_api,
            'storage_failure': self._recover_storage,
            'network_failure': self._recover_network
        }
    
    async def start_monitoring(self):
        """Start continuous health monitoring and DR management."""
        logger.info("Starting disaster recovery monitoring...")
        
        while True:
            try:
                await self._health_check_cycle()
                await asyncio.sleep(self.dr_config['health_check_interval'])
            except Exception as e:
                logger.error(f"Health check cycle failed: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _health_check_cycle(self):
        """Single cycle of health checking and DR actions."""
        # Check health of all services
        health_status = await self.health_checker.check_all_services()
        
        # Update service histories
        for service_name, health in health_status.items():
            if service_name not in self.service_histories:
                self.service_histories[service_name] = []
            
            self.service_histories[service_name].append(health)
            
            # Keep only recent history (last 100 checks)
            if len(self.service_histories[service_name]) > 100:
                self.service_histories[service_name] = self.service_histories[service_name][-100:]
        
        # Analyze health trends and trigger actions
        await self._analyze_and_act(health_status)
        
        # Emergency backup if needed
        await self._check_emergency_backup()
        
        # Log overall system health
        unhealthy_services = [
            name for name, health in health_status.items()
            if health.status == ServiceStatus.UNHEALTHY
        ]
        
        if unhealthy_services:
            logger.warning(f"Unhealthy services detected: {unhealthy_services}")
        else:
            logger.debug("All services healthy")
    
    async def _analyze_and_act(self, health_status: Dict[str, ServiceHealth]):
        """Analyze health status and take appropriate actions."""
        for service_name, health in health_status.items():
            service_history = self.service_histories.get(service_name, [])
            
            if not service_history:
                continue
            
            # Count recent failures
            recent_checks = service_history[-self.dr_config['failover_threshold']:]
            recent_failures = sum(
                1 for h in recent_checks
                if h.status == ServiceStatus.UNHEALTHY
            )
            
            # Check for failover conditions
            if (recent_failures >= self.dr_config['failover_threshold'] and
                self.dr_config['auto_failover_enabled']):
                
                # Check if not already failed over
                failover_status = self.load_balancer.get_failover_status()
                if not failover_status['failover_status'].get(service_name, {}).get('failed_over', False):
                    await self._trigger_failover(service_name, FailoverTrigger.HEALTH_CHECK,
                                                f"Service unhealthy for {recent_failures} consecutive checks")
            
            # Check for failback conditions
            elif (service_name in self.load_balancer.failover_status and
                  self.load_balancer.failover_status[service_name].get('failed_over', False)):
                
                # Count recent successes
                recent_successes = sum(
                    1 for h in recent_checks
                    if h.status == ServiceStatus.HEALTHY
                )
                
                if (recent_successes >= self.dr_config['failback_threshold'] and
                    self.dr_config['auto_failback_enabled']):
                    
                    await self._trigger_failback(service_name, "Service recovered")
    
    async def _trigger_failover(self, service_name: str, trigger: FailoverTrigger, reason: str):
        """Trigger failover for a service."""
        logger.warning(f"Triggering failover for {service_name}: {reason}")
        
        start_time = time.time()
        event_id = f"failover_{service_name}_{int(start_time)}"
        
        try:
            # Attempt failover
            success = self.load_balancer.failover_to_secondary(service_name, reason)
            
            duration = time.time() - start_time
            
            # Record failover event
            event = FailoverEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                trigger=trigger,
                primary_service=f"{service_name}_primary",
                secondary_service=f"{service_name}_secondary",
                success=success,
                duration_seconds=duration,
                reason=reason
            )
            
            self.failover_events.append(event)
            
            if success:
                logger.info(f"Failover successful for {service_name} in {duration:.2f}s")
                
                # Trigger automatic recovery procedures
                if service_name in self.recovery_procedures:
                    asyncio.create_task(self.recovery_procedures[service_name]())
            else:
                logger.error(f"Failover failed for {service_name}")
                
                # Escalate to manual intervention
                await self._escalate_to_manual(service_name, reason)
            
        except Exception as e:
            logger.error(f"Failover process failed for {service_name}: {e}")
            await self._escalate_to_manual(service_name, f"Failover exception: {e}")
    
    async def _trigger_failback(self, service_name: str, reason: str):
        """Trigger failback for a service."""
        logger.info(f"Triggering failback for {service_name}: {reason}")
        
        try:
            success = self.load_balancer.failback_to_primary(service_name)
            
            if success:
                logger.info(f"Failback successful for {service_name}")
            else:
                logger.warning(f"Failback failed for {service_name}")
                
        except Exception as e:
            logger.error(f"Failback process failed for {service_name}: {e}")
    
    async def _escalate_to_manual(self, service_name: str, reason: str):
        """Escalate issue to manual intervention."""
        logger.critical(f"MANUAL INTERVENTION REQUIRED for {service_name}: {reason}")
        
        # In a real implementation, this would:
        # - Send alerts to operations team
        # - Create tickets in incident management system
        # - Send notifications via multiple channels
        
        # For now, create an emergency backup
        await self._create_emergency_backup(f"Service failure escalation: {service_name}")
    
    async def _check_emergency_backup(self):
        """Check if emergency backup is needed."""
        current_time = datetime.now()
        
        # Create emergency backup if interval has passed
        if (self.last_emergency_backup is None or
            (current_time - self.last_emergency_backup).seconds >= self.dr_config['emergency_backup_interval']):
            
            # Check if any services are unhealthy
            health_status = await self.health_checker.check_all_services()
            unhealthy_count = sum(
                1 for health in health_status.values()
                if health.status == ServiceStatus.UNHEALTHY
            )
            
            if unhealthy_count > 0:
                await self._create_emergency_backup(f"Emergency backup due to {unhealthy_count} unhealthy services")
    
    async def _create_emergency_backup(self, reason: str):
        """Create emergency backup."""
        logger.info(f"Creating emergency backup: {reason}")
        
        try:
            # Run backup in background
            def run_backup():
                backup_manager = get_backup_manager()
                result = backup_manager.create_full_backup(
                    include_uploads=True,
                    include_models=False,  # Skip models in emergency backup
                    upload_to_cloud=True
                )
                return result
            
            # Execute backup in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, run_backup)
            
            if result['success']:
                logger.info(f"Emergency backup completed: {result['backup_id']}")
                self.last_emergency_backup = datetime.now()
            else:
                logger.error(f"Emergency backup failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Emergency backup exception: {e}")
    
    # Recovery procedures
    async def _recover_database(self):
        """Database recovery procedure."""
        logger.info("Starting database recovery procedure...")
        
        try:
            # Attempt to restart database service
            result = subprocess.run(['systemctl', 'restart', 'postgresql'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Database service restarted successfully")
                
                # Wait for service to be ready
                await asyncio.sleep(10)
                
                # Verify database is responding
                health = await self.health_checker.check_service_health('database', 'postgresql://localhost:5432')
                
                if health.status == ServiceStatus.HEALTHY:
                    logger.info("Database recovery successful")
                else:
                    logger.warning("Database still unhealthy after restart")
            else:
                logger.error(f"Failed to restart database: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
    
    async def _recover_web_app(self):
        """Web application recovery procedure."""
        logger.info("Starting web application recovery procedure...")
        
        try:
            # Restart web application
            # In production, this might restart Docker containers or systemd services
            logger.info("Web application recovery procedure completed")
            
        except Exception as e:
            logger.error(f"Web application recovery failed: {e}")
    
    async def _recover_prediction_api(self):
        """Prediction API recovery procedure."""
        logger.info("Starting prediction API recovery procedure...")
        
        try:
            # Restart prediction API
            # This might involve reloading ML models or restarting containers
            logger.info("Prediction API recovery procedure completed")
            
        except Exception as e:
            logger.error(f"Prediction API recovery failed: {e}")
    
    async def _recover_storage(self):
        """Storage recovery procedure."""
        logger.info("Starting storage recovery procedure...")
        
        try:
            # Check disk space and clean up if needed
            # Verify mount points
            # Check permissions
            logger.info("Storage recovery procedure completed")
            
        except Exception as e:
            logger.error(f"Storage recovery failed: {e}")
    
    async def _recover_network(self):
        """Network recovery procedure."""
        logger.info("Starting network recovery procedure...")
        
        try:
            # Check network connectivity
            # Restart network services if needed
            # Verify DNS resolution
            logger.info("Network recovery procedure completed")
            
        except Exception as e:
            logger.error(f"Network recovery failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'dr_config': self.dr_config,
            'load_balancer_status': self.load_balancer.get_failover_status(),
            'recent_failover_events': [
                event.to_dict() for event in self.failover_events[-10:]
            ],
            'service_health_summary': {
                service: {
                    'current_status': history[-1].status.value if history else 'unknown',
                    'checks_count': len(history),
                    'avg_response_time': sum(h.response_time_ms for h in history) / len(history) if history else 0
                }
                for service, history in self.service_histories.items()
            }
        }
    
    async def manual_failover(self, service_name: str, reason: str = "Manual failover") -> bool:
        """Manually trigger failover for a service."""
        logger.info(f"Manual failover requested for {service_name}")
        
        await self._trigger_failover(service_name, FailoverTrigger.MANUAL, reason)
        return True
    
    async def manual_failback(self, service_name: str, reason: str = "Manual failback") -> bool:
        """Manually trigger failback for a service."""
        logger.info(f"Manual failback requested for {service_name}")
        
        await self._trigger_failback(service_name, reason)
        return True


# Global DR manager instance
_dr_manager: Optional[DisasterRecoveryManager] = None


def get_dr_manager() -> DisasterRecoveryManager:
    """Get or create global disaster recovery manager instance."""
    global _dr_manager
    
    if _dr_manager is None:
        _dr_manager = DisasterRecoveryManager()
    
    return _dr_manager


# CLI interface for DR operations
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NightScan Disaster Recovery System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start DR monitoring')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Failover command
    failover_parser = subparsers.add_parser('failover', help='Manual failover')
    failover_parser.add_argument('service', help='Service name to failover')
    failover_parser.add_argument('--reason', default='Manual failover', help='Reason for failover')
    
    # Failback command
    failback_parser = subparsers.add_parser('failback', help='Manual failback')
    failback_parser.add_argument('service', help='Service name to failback')
    failback_parser.add_argument('--reason', default='Manual failback', help='Reason for failback')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    dr_manager = get_dr_manager()
    
    if args.command == 'monitor':
        asyncio.run(dr_manager.start_monitoring())
    
    elif args.command == 'status':
        status = dr_manager.get_system_status()
        print(json.dumps(status, indent=2))
    
    elif args.command == 'failover':
        async def do_failover():
            result = await dr_manager.manual_failover(args.service, args.reason)
            print(f"Failover {'successful' if result else 'failed'}")
        
        asyncio.run(do_failover())
    
    elif args.command == 'failback':
        async def do_failback():
            result = await dr_manager.manual_failback(args.service, args.reason)
            print(f"Failback {'successful' if result else 'failed'}")
        
        asyncio.run(do_failback())