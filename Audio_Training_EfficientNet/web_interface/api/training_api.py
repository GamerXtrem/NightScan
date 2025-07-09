"""
API REST pour le Contrôle d'Entraînement EfficientNet - NightScan

API complète pour gérer l'entraînement, la configuration et le monitoring
des modèles EfficientNet avec intégration au système NightScan.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import torch
import psutil

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.efficientnet_config import get_config, get_available_configs
from utils.training_utils import TrainingHistory
from utils.metrics import MetricsCalculator

# Configure logging
logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__, url_prefix='/api')

# Global training state (shared with main app)
training_state = None

def init_api(app, training_manager, state):
    """Initialize API with Flask app and training manager."""
    global training_state
    training_state = state
    app.register_blueprint(api)

@api.route('/configs', methods=['GET'])
def get_configs():
    """Get all available EfficientNet configurations."""
    try:
        configs = get_available_configs()
        
        # Add detailed information for each config
        detailed_configs = []
        for config_name in configs:
            try:
                config = get_config(config_name)
                config_dict = config.to_dict()
                config_dict['name'] = config_name
                detailed_configs.append(config_dict)
            except Exception as e:
                logger.warning(f"Could not load config {config_name}: {e}")
        
        return jsonify({
            'success': True,
            'configs': detailed_configs,
            'count': len(detailed_configs)
        })
    except Exception as e:
        logger.error(f"Get configs error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/configs/<config_name>', methods=['GET'])
def get_config_details(config_name):
    """Get detailed configuration for a specific model."""
    try:
        config = get_config(config_name)
        config_dict = config.to_dict()
        config_dict['name'] = config_name
        
        # Add estimated training time and resource requirements
        estimated_time = _estimate_training_time(config_dict)
        memory_requirements = _estimate_memory_requirements(config_dict)
        
        return jsonify({
            'success': True,
            'config': config_dict,
            'estimates': {
                'training_time_hours': estimated_time,
                'memory_gb': memory_requirements
            }
        })
    except Exception as e:
        logger.error(f"Get config details error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 404

@api.route('/training/start', methods=['POST'])
def start_training():
    """Start a new training session with extended parameters."""
    try:
        data = request.get_json()
        
        # Enhanced validation
        required_fields = ['config_name', 'train_csv', 'val_csv']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        # Validate file paths
        for csv_field in ['train_csv', 'val_csv']:
            if not Path(data[csv_field]).exists():
                return jsonify({'success': False, 'error': f'File not found: {data[csv_field]}'}), 400
        
        # Get and validate configuration
        config_name = data['config_name']
        try:
            config = get_config(config_name)
            config_dict = config.to_dict()
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid configuration: {str(e)}'}), 400
        
        # Override config with user parameters
        overrides = ['epochs', 'batch_size', 'learning_rate', 'mixed_precision', 
                    'gradient_clipping', 'early_stopping_patience', 'num_workers',
                    'use_augmentation', 'freq_mask', 'time_mask', 'mixup_alpha']
        
        for param in overrides:
            if param in data:
                config_dict[param] = data[param]
        
        # Add output directory
        if 'output_dir' in data:
            config_dict['output_dir'] = data['output_dir']
        else:
            config_dict['output_dir'] = f"models/efficientnet_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Data paths
        data_paths = {
            'train_csv': data['train_csv'],
            'val_csv': data['val_csv'],
            'test_csv': data.get('test_csv')
        }
        
        # Check if training is already active
        if training_state and training_state['active']:
            return jsonify({'success': False, 'error': 'Training already in progress'}), 409
        
        # Start training (this would be handled by the training manager)
        from web_interface.training_app import training_manager
        session_id = training_manager.start_training(config_dict, data_paths)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'config': config_dict,
            'message': 'Training started successfully'
        })
        
    except Exception as e:
        logger.error(f"Start training error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/training/stop', methods=['POST'])
def stop_training():
    """Stop the current training session."""
    try:
        if not training_state or not training_state['active']:
            return jsonify({'success': False, 'error': 'No active training session'}), 400
        
        from web_interface.training_app import training_manager
        training_manager.stop_training()
        
        return jsonify({
            'success': True,
            'message': 'Training stop requested'
        })
    except Exception as e:
        logger.error(f"Stop training error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/training/status', methods=['GET'])
def get_training_status():
    """Get comprehensive training status."""
    try:
        if not training_state:
            return jsonify({
                'success': True,
                'active': False,
                'session_id': None,
                'progress': None,
                'config': None
            })
        
        # Calculate additional metrics
        additional_metrics = {}
        if training_state['active'] and training_state['history']:
            history = training_state['history']
            if len(history) > 1:
                # Calculate improvement rates
                latest = history[-1]
                previous = history[-2]
                
                additional_metrics = {
                    'loss_improvement': (previous['val_loss'] - latest['val_loss']) / previous['val_loss'] * 100,
                    'accuracy_improvement': latest['val_accuracy'] - previous['val_accuracy'],
                    'epochs_completed': len(history),
                    'avg_epoch_time': _calculate_avg_epoch_time(history)
                }
        
        return jsonify({
            'success': True,
            'active': training_state['active'],
            'session_id': training_state['session_id'],
            'progress': training_state['progress'],
            'config': training_state['config'],
            'start_time': training_state['start_time'].isoformat() if training_state['start_time'] else None,
            'metrics': additional_metrics
        })
    except Exception as e:
        logger.error(f"Get training status error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/training/history', methods=['GET'])
def get_training_history():
    """Get training history with optional filtering."""
    try:
        limit = request.args.get('limit', type=int, default=100)
        metric = request.args.get('metric', default='all')
        
        history = training_state['history'] if training_state else []
        logs = training_state['logs'] if training_state else []
        
        # Limit results
        if limit > 0:
            history = history[-limit:]
            logs = logs[-limit:]
        
        # Filter by metric if specified
        if metric != 'all' and history:
            filtered_history = []
            for entry in history:
                if metric in entry:
                    filtered_history.append({
                        'epoch': entry['current_epoch'],
                        'value': entry[metric],
                        'timestamp': entry.get('timestamp', '')
                    })
            history = filtered_history
        
        return jsonify({
            'success': True,
            'history': history,
            'logs': logs,
            'total_epochs': len(history)
        })
    except Exception as e:
        logger.error(f"Get training history error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/training/metrics', methods=['GET'])
def get_training_metrics():
    """Get detailed training metrics and statistics."""
    try:
        if not training_state or not training_state['history']:
            return jsonify({
                'success': True,
                'metrics': {},
                'statistics': {}
            })
        
        history = training_state['history']
        
        # Calculate comprehensive metrics
        metrics = {
            'current': training_state['progress'],
            'best': _get_best_metrics(history),
            'worst': _get_worst_metrics(history),
            'trends': _calculate_trends(history)
        }
        
        # Calculate statistics
        statistics = {
            'total_epochs': len(history),
            'convergence_epoch': _find_convergence_epoch(history),
            'overfitting_detected': _detect_overfitting(history),
            'learning_rate_schedule': _get_lr_schedule(history)
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'statistics': statistics
        })
    except Exception as e:
        logger.error(f"Get training metrics error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/system/info', methods=['GET'])
def get_system_info():
    """Get comprehensive system information."""
    try:
        # GPU information
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info = {
                'available': True,
                'device_count': gpu_count,
                'devices': []
            }
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'id': i,
                    'name': props.name,
                    'memory_total': props.total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i),
                    'utilization': torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else None
                })
        elif torch.backends.mps.is_available():
            gpu_info = {
                'available': True,
                'device_type': 'MPS (Apple Silicon)',
                'device_count': 1,
            }
        else:
            gpu_info = {'available': False}
        
        # CPU information
        cpu_info = {
            'count': psutil.cpu_count(),
            'usage': psutil.cpu_percent(interval=1),
            'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': (disk.used / disk.total) * 100
        }
        
        # Network information
        network = psutil.net_io_counters()
        network_info = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        return jsonify({
            'success': True,
            'system': {
                'gpu': gpu_info,
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'network': network_info,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Get system info error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/models/compare', methods=['POST'])
def compare_models():
    """Compare two model configurations or results."""
    try:
        data = request.get_json()
        
        if 'model1' not in data or 'model2' not in data:
            return jsonify({'success': False, 'error': 'Both model1 and model2 must be specified'}), 400
        
        # Get model configurations
        model1_config = get_config(data['model1']).to_dict()
        model2_config = get_config(data['model2']).to_dict()
        
        # Calculate comparison metrics
        comparison = {
            'model1': {
                'name': data['model1'],
                'config': model1_config,
                'estimated_time': _estimate_training_time(model1_config),
                'memory_requirements': _estimate_memory_requirements(model1_config)
            },
            'model2': {
                'name': data['model2'],
                'config': model2_config,
                'estimated_time': _estimate_training_time(model2_config),
                'memory_requirements': _estimate_memory_requirements(model2_config)
            },
            'comparison': {
                'speed_ratio': _estimate_training_time(model1_config) / _estimate_training_time(model2_config),
                'memory_ratio': _estimate_memory_requirements(model1_config) / _estimate_memory_requirements(model2_config),
                'parameter_ratio': model1_config.get('num_parameters', 1) / model2_config.get('num_parameters', 1)
            }
        }
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
        
    except Exception as e:
        logger.error(f"Compare models error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/data/validate', methods=['POST'])
def validate_data():
    """Validate training data files."""
    try:
        data = request.get_json()
        
        if 'files' not in data:
            return jsonify({'success': False, 'error': 'Files list is required'}), 400
        
        validation_results = {}
        
        for file_path in data['files']:
            try:
                path = Path(file_path)
                if not path.exists():
                    validation_results[file_path] = {
                        'valid': False,
                        'error': 'File not found'
                    }
                    continue
                
                # Basic CSV validation
                if path.suffix.lower() == '.csv':
                    import pandas as pd
                    df = pd.read_csv(path)
                    
                    validation_results[file_path] = {
                        'valid': True,
                        'rows': len(df),
                        'columns': list(df.columns),
                        'size_mb': path.stat().st_size / (1024 * 1024)
                    }
                else:
                    validation_results[file_path] = {
                        'valid': False,
                        'error': 'Unsupported file type'
                    }
                    
            except Exception as e:
                validation_results[file_path] = {
                    'valid': False,
                    'error': str(e)
                }
        
        return jsonify({
            'success': True,
            'validation_results': validation_results
        })
        
    except Exception as e:
        logger.error(f"Validate data error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Helper functions

def _estimate_training_time(config):
    """Estimate training time based on configuration."""
    base_time = 0.1  # Base time per epoch in hours
    
    # Adjust based on model complexity
    if 'b0' in config.get('model_name', ''):
        multiplier = 1.0
    elif 'b1' in config.get('model_name', ''):
        multiplier = 1.5
    elif 'b2' in config.get('model_name', ''):
        multiplier = 2.0
    else:
        multiplier = 1.0
    
    # Adjust based on batch size
    batch_size = config.get('batch_size', 32)
    batch_multiplier = 32 / batch_size  # Smaller batch = longer training
    
    # Calculate total time
    epochs = config.get('epochs', 50)
    total_time = base_time * multiplier * batch_multiplier * epochs
    
    return round(total_time, 2)

def _estimate_memory_requirements(config):
    """Estimate memory requirements in GB."""
    base_memory = 2.0  # Base memory in GB
    
    # Adjust based on model size
    if 'b0' in config.get('model_name', ''):
        multiplier = 1.0
    elif 'b1' in config.get('model_name', ''):
        multiplier = 1.5
    elif 'b2' in config.get('model_name', ''):
        multiplier = 2.0
    else:
        multiplier = 1.0
    
    # Adjust based on batch size
    batch_size = config.get('batch_size', 32)
    batch_multiplier = batch_size / 32
    
    # Mixed precision reduces memory usage
    precision_multiplier = 0.7 if config.get('mixed_precision', True) else 1.0
    
    total_memory = base_memory * multiplier * batch_multiplier * precision_multiplier
    
    return round(total_memory, 2)

def _calculate_avg_epoch_time(history):
    """Calculate average epoch time from history."""
    if len(history) < 2:
        return 0
    
    # Simple estimation based on epoch progression
    return 120  # 2 minutes per epoch (placeholder)

def _get_best_metrics(history):
    """Get best metrics from training history."""
    if not history:
        return {}
    
    best_train_loss = min(history, key=lambda x: x['train_loss'])
    best_val_loss = min(history, key=lambda x: x['val_loss'])
    best_train_acc = max(history, key=lambda x: x['train_accuracy'])
    best_val_acc = max(history, key=lambda x: x['val_accuracy'])
    
    return {
        'train_loss': best_train_loss['train_loss'],
        'val_loss': best_val_loss['val_loss'],
        'train_accuracy': best_train_acc['train_accuracy'],
        'val_accuracy': best_val_acc['val_accuracy']
    }

def _get_worst_metrics(history):
    """Get worst metrics from training history."""
    if not history:
        return {}
    
    worst_train_loss = max(history, key=lambda x: x['train_loss'])
    worst_val_loss = max(history, key=lambda x: x['val_loss'])
    worst_train_acc = min(history, key=lambda x: x['train_accuracy'])
    worst_val_acc = min(history, key=lambda x: x['val_accuracy'])
    
    return {
        'train_loss': worst_train_loss['train_loss'],
        'val_loss': worst_val_loss['val_loss'],
        'train_accuracy': worst_train_acc['train_accuracy'],
        'val_accuracy': worst_val_acc['val_accuracy']
    }

def _calculate_trends(history):
    """Calculate trends in training metrics."""
    if len(history) < 2:
        return {}
    
    recent = history[-5:]  # Last 5 epochs
    
    # Calculate simple trends
    train_loss_trend = (recent[-1]['train_loss'] - recent[0]['train_loss']) / len(recent)
    val_loss_trend = (recent[-1]['val_loss'] - recent[0]['val_loss']) / len(recent)
    train_acc_trend = (recent[-1]['train_accuracy'] - recent[0]['train_accuracy']) / len(recent)
    val_acc_trend = (recent[-1]['val_accuracy'] - recent[0]['val_accuracy']) / len(recent)
    
    return {
        'train_loss_trend': train_loss_trend,
        'val_loss_trend': val_loss_trend,
        'train_accuracy_trend': train_acc_trend,
        'val_accuracy_trend': val_acc_trend
    }

def _find_convergence_epoch(history):
    """Find the epoch where the model converged."""
    if len(history) < 10:
        return None
    
    # Simple heuristic: look for plateau in validation loss
    for i in range(10, len(history)):
        recent_losses = [h['val_loss'] for h in history[i-10:i]]
        if max(recent_losses) - min(recent_losses) < 0.01:
            return i
    
    return None

def _detect_overfitting(history):
    """Detect if overfitting is occurring."""
    if len(history) < 10:
        return False
    
    # Check if validation loss is increasing while training loss decreases
    recent = history[-10:]
    train_losses = [h['train_loss'] for h in recent]
    val_losses = [h['val_loss'] for h in recent]
    
    train_trend = train_losses[-1] - train_losses[0]
    val_trend = val_losses[-1] - val_losses[0]
    
    return train_trend < -0.1 and val_trend > 0.1

def _get_lr_schedule(history):
    """Get learning rate schedule from history."""
    if not history:
        return []
    
    return [{'epoch': h['current_epoch'], 'lr': h.get('learning_rate', 0)} for h in history]