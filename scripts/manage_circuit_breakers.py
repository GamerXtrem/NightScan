#!/usr/bin/env python3
"""
Circuit Breaker Management Script for NightScan

This script provides command-line utilities for managing circuit breakers
including monitoring, resetting, and configuration management.

Usage:
    python scripts/manage_circuit_breakers.py status
    python scripts/manage_circuit_breakers.py metrics
    python scripts/manage_circuit_breakers.py reset [service_name]
    python scripts/manage_circuit_breakers.py config [--save/--load] [file]
    python scripts/manage_circuit_breakers.py monitor [--interval=30]
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from circuit_breaker_config import (
    get_circuit_breaker_manager, initialize_circuit_breakers,
    CircuitBreakerManager
)
from circuit_breaker import CircuitState


def print_status(health_data: Dict[str, Any]):
    """Print circuit breaker status in a readable format."""
    print("\n🔵 Circuit Breaker Status Report")
    print("=" * 50)
    print(f"Timestamp: {health_data['timestamp']}")
    print(f"Overall Status: {health_data['overall_status'].upper()}")
    
    summary = health_data['summary']
    print(f"\nSummary:")
    print(f"  Total: {summary['total']}")
    print(f"  ✅ Healthy: {summary['healthy']}")
    print(f"  ⚠️ Degraded: {summary['degraded']}")
    print(f"  ❌ Unhealthy: {summary['unhealthy']}")
    
    print(f"\nCircuit Breaker Details:")
    print("-" * 50)
    
    for name, details in health_data['circuit_breakers'].items():
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️', 
            'unhealthy': '❌'
        }.get(details['status'], '❓')
        
        print(f"{status_emoji} {name}: {details['status'].upper()}")
        
        if 'details' in details:
            circuit_details = details['details']
            if 'circuit_state' in circuit_details:
                print(f"    State: {circuit_details['circuit_state']}")
            if 'available' in circuit_details:
                print(f"    Available: {circuit_details['available']}")
        
        if 'error' in details:
            print(f"    Error: {details['error']}")
        
        print()


def print_metrics(metrics_data: Dict[str, Any]):
    """Print circuit breaker metrics in a readable format."""
    print("\n📊 Circuit Breaker Metrics")
    print("=" * 50)
    print(f"Timestamp: {metrics_data['timestamp']}")
    
    manager_info = metrics_data['manager_info']
    print(f"\nManager Info:")
    print(f"  Total Circuits: {manager_info['total_circuits']}")
    print(f"  Services Configured: {manager_info['services_configured']}")
    print(f"  Environment: {manager_info['environment']}")
    print(f"  Monitoring Enabled: {manager_info['monitoring_enabled']}")
    
    print(f"\nCircuit Breaker Metrics:")
    print("-" * 50)
    
    for name, metrics in metrics_data['circuit_breakers'].items():
        if 'error' in metrics:
            print(f"❌ {name}: Error - {metrics['error']}")
            continue
            
        print(f"🔵 {name}:")
        print(f"    State: {metrics.get('state', 'unknown')}")
        print(f"    Total Requests: {metrics.get('total_requests', 0)}")
        print(f"    Success Rate: {metrics.get('success_rate', 0):.2%}")
        print(f"    Failed Requests: {metrics.get('failed_requests', 0)}")
        print(f"    Rejected Requests: {metrics.get('rejected_requests', 0)}")
        
        if 'average_response_time' in metrics:
            print(f"    Avg Response Time: {metrics['average_response_time']:.2f}s")
        
        if 'last_failure_time' in metrics and metrics['last_failure_time']:
            print(f"    Last Failure: {metrics['last_failure_time']}")
        
        print()


def reset_circuits(manager: CircuitBreakerManager, service_name: str = None):
    """Reset circuit breakers."""
    if service_name:
        print(f"🔄 Resetting circuit breakers for service: {service_name}")
        # Reset specific service circuits
        circuits_reset = 0
        for circuit_name, circuit in manager.circuit_breakers.items():
            if circuit_name.startswith(service_name):
                circuit.reset()
                circuits_reset += 1
                print(f"  ✅ Reset: {circuit_name}")
        
        if circuits_reset == 0:
            print(f"  ⚠️ No circuits found for service: {service_name}")
        else:
            print(f"  ✅ Reset {circuits_reset} circuits for {service_name}")
    else:
        print("🔄 Resetting all circuit breakers...")
        manager.reset_all_circuits()
        print("  ✅ All circuits reset")


def save_config(manager: CircuitBreakerManager, file_path: str):
    """Save current configuration to file."""
    print(f"💾 Saving configuration to: {file_path}")
    try:
        manager.save_configuration(file_path)
        print("  ✅ Configuration saved successfully")
    except Exception as e:
        print(f"  ❌ Failed to save configuration: {e}")


def load_config(file_path: str):
    """Load configuration from file."""
    print(f"📂 Loading configuration from: {file_path}")
    try:
        manager = initialize_circuit_breakers(file_path)
        print("  ✅ Configuration loaded successfully")
        return manager
    except Exception as e:
        print(f"  ❌ Failed to load configuration: {e}")
        return None


def monitor_circuits(manager: CircuitBreakerManager, interval: int = 30):
    """Monitor circuit breakers continuously."""
    print(f"👀 Monitoring circuit breakers (interval: {interval}s)")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
            
            health_data = manager.get_health_status()
            print_status(health_data)
            
            print(f"\nNext update in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="Manage NightScan Circuit Breakers")
    parser.add_argument('command', choices=['status', 'metrics', 'reset', 'config', 'monitor'],
                       help='Command to execute')
    parser.add_argument('--service', help='Service name for reset command')
    parser.add_argument('--save', action='store_true', help='Save configuration (for config command)')
    parser.add_argument('--load', action='store_true', help='Load configuration (for config command)')
    parser.add_argument('--file', default='config/circuit_breakers.json', 
                       help='Configuration file path')
    parser.add_argument('--interval', type=int, default=30,
                       help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Initialize circuit breaker manager
    config_file = args.file if args.load else None
    manager = initialize_circuit_breakers(config_file)
    
    if args.command == 'status':
        health_data = manager.get_health_status()
        print_status(health_data)
        
        # Exit with error code if unhealthy
        if health_data['overall_status'] == 'unhealthy':
            sys.exit(1)
        elif health_data['overall_status'] == 'degraded':
            sys.exit(2)
    
    elif args.command == 'metrics':
        metrics_data = manager.get_all_metrics()
        print_metrics(metrics_data)
    
    elif args.command == 'reset':
        reset_circuits(manager, args.service)
    
    elif args.command == 'config':
        if args.save:
            save_config(manager, args.file)
        elif args.load:
            load_config(args.file)
        else:
            print("Use --save or --load with config command")
            sys.exit(1)
    
    elif args.command == 'monitor':
        monitor_circuits(manager, args.interval)


if __name__ == '__main__':
    main()