"""
Interface Web pour l'Entraînement EfficientNet - NightScan

Application Flask moderne pour gérer et monitorer l'entraînement des modèles EfficientNet
avec interface temps réel, contrôle complet et intégration au système NightScan.
"""

import os
import json
import logging
import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from flask_login import LoginManager, login_required, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
import torch

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_config import get_config, get_available_configs
from utils.training_utils import TrainingHistory
from utils.metrics import MetricsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Global training state
training_state = {
    'active': False,
    'session_id': None,
    'config': None,
    'progress': {
        'current_epoch': 0,
        'total_epochs': 0,
        'train_loss': 0.0,
        'val_loss': 0.0,
        'train_accuracy': 0.0,
        'val_accuracy': 0.0,
        'learning_rate': 0.0,
        'eta': '00:00:00'
    },
    'history': [],
    'start_time': None,
    'model_path': None,
    'logs': []
}

# Thread pool for training tasks
executor = ThreadPoolExecutor(max_workers=1)


class TrainingManager:
    """Manages EfficientNet training sessions with real-time monitoring."""
    
    def __init__(self):
        self.current_session = None
        self.history = TrainingHistory(Path("training_sessions"))
        self.metrics_calculator = None
        
    def start_training(self, config_dict: Dict[str, Any], data_paths: Dict[str, str]) -> str:
        """Start a new training session."""
        session_id = str(uuid.uuid4())
        
        # Update global state
        training_state['active'] = True
        training_state['session_id'] = session_id
        training_state['config'] = config_dict
        training_state['start_time'] = datetime.now()
        training_state['progress']['total_epochs'] = config_dict.get('epochs', 50)
        training_state['logs'] = []
        
        # Start training in background thread
        executor.submit(self._run_training, session_id, config_dict, data_paths)
        
        logger.info(f"Started training session: {session_id}")
        return session_id
    
    def stop_training(self):
        """Stop the current training session."""
        if training_state['active']:
            training_state['active'] = False
            logger.info("Training stop requested")
            self._emit_log("Training stopped by user", "warning")
    
    def _run_training(self, session_id: str, config_dict: Dict[str, Any], data_paths: Dict[str, str]):
        """Run the training process in a separate thread."""
        try:
            self._emit_log(f"Starting training session {session_id}", "info")
            
            # Import training modules
            from scripts.train_efficientnet import train_model
            from models.efficientnet_config import EfficientNetConfig
            
            # Create configuration
            config = EfficientNetConfig(**config_dict)
            
            # Simulate training with real-time updates (for now)
            # In a real implementation, this would call the actual training function
            self._simulate_training(config, data_paths)
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self._emit_log(f"Training error: {str(e)}", "error")
            training_state['active'] = False
    
    def _simulate_training(self, config, data_paths):
        """Simulate training process with realistic progression."""
        total_epochs = config.epochs
        
        for epoch in range(1, total_epochs + 1):
            if not training_state['active']:
                break
                
            # Simulate epoch training
            train_loss = 2.0 * (0.8 ** epoch) + 0.1  # Decreasing loss
            val_loss = train_loss + 0.1 + (0.05 * (epoch % 3))  # Slightly higher with noise
            train_acc = min(95.0, 60.0 + epoch * 0.8)  # Increasing accuracy
            val_acc = train_acc - 2.0 + (epoch % 2)  # Slightly lower with validation gap
            lr = config.learning_rate * (0.95 ** epoch)  # Decreasing learning rate
            
            # Calculate ETA
            elapsed = time.time() - training_state['start_time'].timestamp()
            eta_seconds = (elapsed / epoch) * (total_epochs - epoch)
            eta = str(timedelta(seconds=int(eta_seconds)))
            
            # Update progress
            progress = {
                'current_epoch': epoch,
                'total_epochs': total_epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'learning_rate': lr,
                'eta': eta,
                'progress_percent': (epoch / total_epochs) * 100
            }
            
            training_state['progress'] = progress
            training_state['history'].append(progress.copy())
            
            # Emit real-time updates
            socketio.emit('training_progress', progress, room='training')
            
            # Emit logs
            self._emit_log(
                f"Epoch {epoch}/{total_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%",
                "info"
            )
            
            # Simulate epoch duration
            time.sleep(2)  # 2 seconds per epoch for demo
        
        # Training completed
        if training_state['active']:
            training_state['active'] = False
            self._emit_log("Training completed successfully!", "success")
            socketio.emit('training_complete', {'status': 'success'}, room='training')
    
    def _emit_log(self, message: str, level: str = "info"):
        """Emit log message to connected clients."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'level': level
        }
        training_state['logs'].append(log_entry)
        
        # Keep only last 100 logs
        if len(training_state['logs']) > 100:
            training_state['logs'] = training_state['logs'][-100:]
        
        socketio.emit('training_log', log_entry, room='training')


# Initialize training manager
training_manager = TrainingManager()


@app.route('/')
def index():
    """Main dashboard for training interface."""
    return render_template('training_dashboard.html')


@app.route('/config')
def config_page():
    """Training configuration page."""
    available_configs = get_available_configs()
    return render_template('training_config.html', configs=available_configs)


@app.route('/comparison')
def comparison_page():
    """Model comparison page."""
    return render_template('model_comparison.html')


@app.route('/api/start_training', methods=['POST'])
def start_training():
    """API endpoint to start training."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['config_name', 'train_csv', 'val_csv']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get configuration
        config_name = data['config_name']
        try:
            config = get_config(config_name)
            config_dict = config.to_dict()
        except Exception as e:
            return jsonify({'error': f'Invalid configuration: {str(e)}'}), 400
        
        # Override config with user parameters
        if 'epochs' in data:
            config_dict['epochs'] = int(data['epochs'])
        if 'batch_size' in data:
            config_dict['batch_size'] = int(data['batch_size'])
        if 'learning_rate' in data:
            config_dict['learning_rate'] = float(data['learning_rate'])
        
        # Data paths
        data_paths = {
            'train_csv': data['train_csv'],
            'val_csv': data['val_csv']
        }
        
        # Check if training is already active
        if training_state['active']:
            return jsonify({'error': 'Training already in progress'}), 409
        
        # Start training
        session_id = training_manager.start_training(config_dict, data_paths)
        
        return jsonify({
            'status': 'started',
            'session_id': session_id,
            'message': 'Training started successfully'
        })
        
    except Exception as e:
        logger.error(f"Start training error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """API endpoint to stop training."""
    try:
        training_manager.stop_training()
        return jsonify({
            'status': 'stopped',
            'message': 'Training stop requested'
        })
    except Exception as e:
        logger.error(f"Stop training error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/training_status')
def training_status():
    """Get current training status."""
    return jsonify({
        'active': training_state['active'],
        'session_id': training_state['session_id'],
        'progress': training_state['progress'],
        'config': training_state['config'],
        'start_time': training_state['start_time'].isoformat() if training_state['start_time'] else None
    })


@app.route('/api/training_history')
def training_history():
    """Get training history for current session."""
    return jsonify({
        'history': training_state['history'],
        'logs': training_state['logs']
    })


@app.route('/api/available_configs')
def available_configs():
    """Get list of available EfficientNet configurations."""
    try:
        configs = get_available_configs()
        return jsonify({'configs': configs})
    except Exception as e:
        logger.error(f"Available configs error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/system_info')
def system_info():
    """Get system information for training."""
    try:
        import psutil
        
        # GPU information
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'memory_allocated': torch.cuda.memory_allocated(),
            }
        elif torch.backends.mps.is_available():
            gpu_info = {
                'available': True,
                'device_type': 'MPS (Apple Silicon)',
                'device_count': 1,
            }
        else:
            gpu_info = {'available': False}
        
        # System information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return jsonify({
            'gpu': gpu_info,
            'cpu': {
                'count': psutil.cpu_count(),
                'usage': psutil.cpu_percent(interval=1)
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            }
        })
        
    except Exception as e:
        logger.error(f"System info error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# SocketIO Events

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'Connected to training server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('join_training')
def handle_join_training():
    """Join training room for real-time updates."""
    join_room('training')
    logger.info(f"Client {request.sid} joined training room")
    
    # Send current state to newly connected client
    emit('training_status', {
        'active': training_state['active'],
        'progress': training_state['progress'],
        'config': training_state['config']
    })


@socketio.on('leave_training')
def handle_leave_training():
    """Leave training room."""
    leave_room('training')
    logger.info(f"Client {request.sid} left training room")


if __name__ == '__main__':
    # Create upload directory
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the application
    logger.info("Starting EfficientNet Training Interface")
    logger.info(f"GPU Available: {torch.cuda.is_available() or torch.backends.mps.is_available()}")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)