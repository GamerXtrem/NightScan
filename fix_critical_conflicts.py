#!/usr/bin/env python3
"""
NightScan Critical Conflict Resolution Script
Automatically fixes the most critical conflicts identified in the analysis.
"""

import os
import re
import shutil
from pathlib import Path
import json

class ConflictResolver:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path)
        self.fixes_applied = []
        
    def resolve_all_critical(self):
        """Resolve all critical conflicts."""
        print("🔧 Resolving critical conflicts in NightScan repository...")
        
        self.fix_dependency_conflicts()
        self.fix_duplicate_functions()
        self.fix_api_endpoint_conflicts()
        self.standardize_port_configuration()
        
        self.generate_fix_report()
        
    def fix_dependency_conflicts(self):
        """Fix dependency version conflicts by standardizing versions."""
        print("  📦 Fixing dependency conflicts...")
        
        # Create standardized requirements
        standardized_deps = {
            'torch': '==2.1.1',  # Stable ML version
            'torchvision': '==0.16.1',
            'numpy': '==1.24.3',  # Compatible with torch
            'redis': '==4.6.0',
            'celery': '==5.3.6', 
            'pydantic': '==2.3.0',
            'flask': '==2.3.3',
            'sqlalchemy': '==1.4.52'
        }
        
        # Update pyproject.toml
        pyproject_path = self.root_path / 'pyproject.toml'
        if pyproject_path.exists():
            with open(pyproject_path, 'r') as f:
                content = f.read()
                
            # Update dependency versions in pyproject.toml
            for package, version in standardized_deps.items():
                # Replace version ranges with exact versions
                pattern = rf'"{package}>=[\d.]+"'
                replacement = f'"{package}{version}"'
                content = re.sub(pattern, replacement, content)
                
            with open(pyproject_path, 'w') as f:
                f.write(content)
                
            self.fixes_applied.append({
                'type': 'Dependency Standardization',
                'file': 'pyproject.toml',
                'description': 'Standardized dependency versions to match requirements.txt'
            })
            
        # Create requirements-lock.txt for development
        lock_file_path = self.root_path / 'requirements-lock.txt'
        with open(lock_file_path, 'w') as f:
            f.write("# Locked dependency versions for NightScan\n")
            f.write("# Generated automatically to resolve version conflicts\n\n")
            for package, version in standardized_deps.items():
                f.write(f"{package}{version}\n")
                
        self.fixes_applied.append({
            'type': 'Lock File Creation',
            'file': 'requirements-lock.txt',
            'description': 'Created locked dependency versions file'
        })
        
    def fix_duplicate_functions(self):
        """Fix duplicate function implementations."""
        print("  🔀 Fixing duplicate functions...")
        
        # Create shared training module
        shared_dir = self.root_path / 'shared'
        shared_dir.mkdir(exist_ok=True)
        
        # Create shared training framework
        training_framework_content = '''"""
Shared training framework for NightScan ML models.
Eliminates code duplication between audio and picture training.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """Base trainer class for all NightScan ML models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    @abstractmethod
    def prepare_data(self, data_path: str) -> Any:
        """Prepare training data - must be implemented by subclasses."""
        pass
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """Shared training epoch logic."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item:.4f}')
                
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'samples': total_samples
        }
        
    def validate_epoch(self, dataloader, criterion):
        """Shared validation epoch logic."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
                
        val_loss = total_loss / len(dataloader)
        val_accuracy = correct_predictions / total_samples
        
        return {
            'loss': val_loss,
            'accuracy': val_accuracy,
            'samples': total_samples
        }
        
    def save_model(self, path: str, metadata: Optional[Dict] = None):
        """Save model with metadata."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)
        logger.info(f'Model saved to {path}')
        
    def load_model(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Model loaded from {path}')
        return checkpoint.get('metadata', {})


class AudioTrainer(BaseTrainer):
    """Specialized trainer for audio models."""
    
    def prepare_data(self, data_path: str):
        """Prepare audio training data."""
        # Audio-specific data preparation
        from audio_training.scripts.preprocess import preprocess_audio_data
        return preprocess_audio_data(data_path)


class ImageTrainer(BaseTrainer):
    """Specialized trainer for image models."""
    
    def prepare_data(self, data_path: str):
        """Prepare image training data."""
        # Image-specific data preparation
        from picture_training.scripts.prepare_csv import prepare_image_data
        return prepare_image_data(data_path)
'''
        
        shared_training_path = shared_dir / 'training_framework.py'
        with open(shared_training_path, 'w') as f:
            f.write(training_framework_content)
            
        # Create __init__.py
        init_content = '''"""
Shared modules for NightScan to eliminate code duplication.
"""

from .training_framework import BaseTrainer, AudioTrainer, ImageTrainer

__all__ = ['BaseTrainer', 'AudioTrainer', 'ImageTrainer']
'''
        
        with open(shared_dir / '__init__.py', 'w') as f:
            f.write(init_content)
            
        self.fixes_applied.append({
            'type': 'Code Deduplication',
            'file': 'shared/training_framework.py',
            'description': 'Created shared training framework to eliminate duplicate train_epoch functions'
        })
        
        # Create shared notification module
        notification_utils_content = '''"""
Shared notification utilities for NightScan.
Eliminates duplicate notification functions across services.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class NotificationCoordinator:
    """Coordinates notifications across different services to avoid duplication."""
    
    def __init__(self):
        self.notification_service = None
        self.websocket_service = None
        
    def register_notification_service(self, service):
        """Register the main notification service."""
        self.notification_service = service
        
    def register_websocket_service(self, service):
        """Register the WebSocket service."""
        self.websocket_service = service
        
    async def send_prediction_complete_notification(self, prediction_data: Dict[str, Any], user_id: int):
        """Centralized prediction complete notification."""
        if self.notification_service:
            await self.notification_service.send_prediction_complete_notification(
                prediction_data, user_id
            )
            
        if self.websocket_service:
            await self.websocket_service.notify_prediction_complete(
                prediction_data, user_id
            )
            
        logger.info(f"Prediction complete notification sent for user {user_id}")
        
    async def notify_prediction_complete(self, prediction_data: Dict[str, Any], user_id: int):
        """Alias for backward compatibility."""
        await self.send_prediction_complete_notification(prediction_data, user_id)

# Global coordinator instance
_coordinator = NotificationCoordinator()

def get_notification_coordinator():
    """Get the global notification coordinator."""
    return _coordinator

# Backward compatible functions
async def send_prediction_complete_notification(prediction_data: Dict[str, Any], user_id: int):
    """Backward compatible function."""
    coordinator = get_notification_coordinator()
    await coordinator.send_prediction_complete_notification(prediction_data, user_id)

async def notify_prediction_complete(prediction_data: Dict[str, Any], user_id: int):
    """Backward compatible function."""
    coordinator = get_notification_coordinator()
    await coordinator.notify_prediction_complete(prediction_data, user_id)
'''
        
        notification_utils_path = shared_dir / 'notification_utils.py'
        with open(notification_utils_path, 'w') as f:
            f.write(notification_utils_content)
            
        self.fixes_applied.append({
            'type': 'Notification Deduplication',
            'file': 'shared/notification_utils.py',
            'description': 'Created shared notification coordinator to eliminate duplicate functions'
        })
        
    def fix_api_endpoint_conflicts(self):
        """Fix API endpoint conflicts by implementing proper routing."""
        print("  🌐 Fixing API endpoint conflicts...")
        
        # Create API routing configuration
        api_config_content = '''"""
NightScan API Routing Configuration
Resolves endpoint conflicts and standardizes API structure.
"""

from flask import Blueprint

# API versioning configuration
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'

# Health check endpoints configuration
HEALTH_ENDPOINTS = {
    'web_app': '/health',
    'prediction_api': '/api/health', 
    'ml_service': '/ml/health'
}

# Service-specific routing
SERVICE_ROUTES = {
    'web': {
        'prefix': '',
        'health': '/health',
        'ready': '/ready',
        'metrics': '/metrics'
    },
    'prediction_api': {
        'prefix': '/api/v1',
        'health': '/api/v1/health',
        'ready': '/api/v1/ready',
        'predict': '/api/v1/predict'
    },
    'ml_service': {
        'prefix': '/ml',
        'health': '/ml/health',
        'predict': '/ml/predict'
    }
}

def create_service_blueprint(service_name: str, import_name: str):
    """Create a service-specific blueprint with proper routing."""
    config = SERVICE_ROUTES.get(service_name, {})
    prefix = config.get('prefix', f'/{service_name}')
    
    blueprint = Blueprint(
        f'{service_name}_service',
        import_name,
        url_prefix=prefix
    )
    
    return blueprint, config

# Endpoint conflict resolution mapping
ENDPOINT_MAPPING = {
    '/': {
        'web_app': '/',
        'demo_service': '/demo'
    },
    '/health': {
        'web_app': '/health',
        'api_v1': '/api/v1/health', 
        'prediction_api': '/api/health'
    },
    '/ready': {
        'web_app': '/ready',
        'api_v1': '/api/v1/ready',
        'prediction_api': '/api/ready'
    },
    '/metrics': {
        'web_app': '/metrics',
        'prediction_api': '/api/metrics'
    }
}

def get_service_endpoint(service: str, endpoint: str) -> str:
    """Get the correct endpoint for a specific service."""
    return ENDPOINT_MAPPING.get(endpoint, {}).get(service, endpoint)
'''
        
        api_config_path = self.root_path / 'api_routing_config.py'
        with open(api_config_path, 'w') as f:
            f.write(api_config_content)
            
        self.fixes_applied.append({
            'type': 'API Conflict Resolution',
            'file': 'api_routing_config.py', 
            'description': 'Created API routing configuration to resolve endpoint conflicts'
        })
        
    def standardize_port_configuration(self):
        """Standardize port configuration across all services."""
        print("  🔌 Standardizing port configuration...")
        
        # Create port configuration module
        port_config_content = '''"""
NightScan Port Configuration
Centralized port management to prevent conflicts.
"""

import os
from typing import Dict, Optional

# Default port assignments
DEFAULT_PORTS = {
    'web_app': 8000,
    'prediction_api': 8001, 
    'ml_service': 8002,
    'websocket': 8003,
    'metrics': 9090,
    'health_check': 8080,
    'redis': 6379,
    'postgres': 5432,
    'monitoring': 3000
}

# Environment variable mapping
PORT_ENV_VARS = {
    'web_app': 'WEB_PORT',
    'prediction_api': 'PREDICTION_PORT',
    'ml_service': 'ML_SERVICE_PORT', 
    'websocket': 'WEBSOCKET_PORT',
    'metrics': 'METRICS_PORT',
    'health_check': 'HEALTH_CHECK_PORT',
    'redis': 'REDIS_PORT',
    'postgres': 'POSTGRES_PORT',
    'monitoring': 'MONITORING_PORT'
}

def get_port(service: str) -> int:
    """Get port for a specific service from environment or default."""
    env_var = PORT_ENV_VARS.get(service)
    if env_var and env_var in os.environ:
        return int(os.environ[env_var])
    return DEFAULT_PORTS.get(service, 8000)

def get_all_ports() -> Dict[str, int]:
    """Get all configured ports."""
    return {service: get_port(service) for service in DEFAULT_PORTS.keys()}

def check_port_conflicts() -> Dict[str, list]:
    """Check for port conflicts in current configuration."""
    ports = get_all_ports()
    conflicts = {}
    
    port_usage = {}
    for service, port in ports.items():
        if port in port_usage:
            if port not in conflicts:
                conflicts[port] = [port_usage[port]]
            conflicts[port].append(service)
        else:
            port_usage[port] = service
            
    return conflicts

def get_service_url(service: str, host: str = 'localhost', protocol: str = 'http') -> str:
    """Get full URL for a service."""
    port = get_port(service)
    return f"{protocol}://{host}:{port}"

# Port ranges for different environments
PORT_RANGES = {
    'development': {
        'start': 8000,
        'end': 8099
    },
    'testing': {
        'start': 9000, 
        'end': 9099
    },
    'production': {
        'start': 80,
        'end': 8999
    }
}

def allocate_port_range(environment: str = 'development') -> Dict[str, int]:
    """Allocate ports within a specific range for an environment."""
    range_config = PORT_RANGES.get(environment, PORT_RANGES['development'])
    start_port = range_config['start']
    
    allocated_ports = {}
    for i, service in enumerate(DEFAULT_PORTS.keys()):
        allocated_ports[service] = start_port + i
        
    return allocated_ports
'''
        
        port_config_path = self.root_path / 'port_config.py'
        with open(port_config_path, 'w') as f:
            f.write(port_config_content)
            
        # Create environment template
        env_template_content = '''# NightScan Environment Configuration
# Copy this file to .env and adjust values as needed

# Service Ports (configurable to avoid conflicts)
WEB_PORT=8000
PREDICTION_PORT=8001
ML_SERVICE_PORT=8002
WEBSOCKET_PORT=8003
METRICS_PORT=9090
HEALTH_CHECK_PORT=8080

# Database Ports
REDIS_PORT=6379
POSTGRES_PORT=5432

# Monitoring
MONITORING_PORT=3000

# Application Configuration
FLASK_ENV=development
SQLALCHEMY_DATABASE_URI=postgresql://user:password@localhost:5432/nightscan
REDIS_URL=redis://localhost:6379/0

# ML Configuration
TORCH_DEVICE=cpu
MODEL_PATH=./models/
BATCH_SIZE=32

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# External Services
PREDICTION_API_URL=http://localhost:8001/api/predict
WEBSOCKET_URL=ws://localhost:8003

# Feature Flags
ENABLE_NOTIFICATIONS=true
ENABLE_METRICS=true
ENABLE_WEBSOCKETS=true
'''
        
        env_template_path = self.root_path / '.env.example'
        with open(env_template_path, 'w') as f:
            f.write(env_template_content)
            
        self.fixes_applied.append({
            'type': 'Port Standardization',
            'file': 'port_config.py',
            'description': 'Created centralized port configuration to prevent conflicts'
        })
        
        self.fixes_applied.append({
            'type': 'Environment Template',
            'file': '.env.example',
            'description': 'Created environment template with configurable ports'
        })
        
    def generate_fix_report(self):
        """Generate a report of all fixes applied."""
        print("\n" + "="*60)
        print("🔧 NIGHTSCAN CONFLICT RESOLUTION REPORT")
        print("="*60)
        
        print(f"\n✅ Successfully applied {len(self.fixes_applied)} fixes:")
        
        for i, fix in enumerate(self.fixes_applied, 1):
            print(f"\n{i}. {fix['type']}")
            print(f"   File: {fix['file']}")
            print(f"   Description: {fix['description']}")
            
        print(f"\n📋 NEXT STEPS:")
        print("1. Review the created files and integrate them into your services")
        print("2. Update import statements to use shared modules")
        print("3. Configure environment variables using .env.example")
        print("4. Test all services to ensure conflicts are resolved")
        print("5. Update CI/CD pipelines to use new configurations")
        
        print(f"\n📁 NEW FILES CREATED:")
        new_files = [fix['file'] for fix in self.fixes_applied]
        for file_path in new_files:
            print(f"   - {file_path}")
            
        print(f"\n🔄 REQUIRED REFACTORING:")
        print("1. Update Audio_Training/scripts/train.py to use shared.AudioTrainer")
        print("2. Update Picture_Training/scripts/train.py to use shared.ImageTrainer") 
        print("3. Replace duplicate notification functions with shared.notification_utils")
        print("4. Update all services to use port_config.get_port()")
        print("5. Update API routes to use api_routing_config")
        
        # Save fix report
        report_data = {
            'fixes_applied': self.fixes_applied,
            'timestamp': str(Path(__file__).stat().st_mtime),
            'total_fixes': len(self.fixes_applied)
        }
        
        with open('conflict_resolution_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\n💾 Fix report saved to: conflict_resolution_report.json")


if __name__ == "__main__":
    resolver = ConflictResolver()
    resolver.resolve_all_critical()