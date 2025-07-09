#!/usr/bin/env python3
"""
Interface Web d'Entraînement pour les Modèles Edge NightScan

Interface dédiée à l'entraînement des modèles légers optimisés pour mobile.
Sépare l'entraînement edge des gros modèles serveur.
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.lightweight_models import create_lightweight_model, get_model_complexity, compare_models

# Configuration
UPLOAD_FOLDER = Path('uploads')
MODELS_FOLDER = Path('trained_models')
DATASETS_FOLDER = Path('datasets')

for folder in [UPLOAD_FOLDER, MODELS_FOLDER, DATASETS_FOLDER]:
    folder.mkdir(exist_ok=True)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application Flask
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'nightscan_edge_training_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# CORS et SocketIO
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
socketio = SocketIO(app, cors_allowed_origins="*")

# Variables globales
training_sessions = {}
current_training = None


class EdgeTrainingSession:
    """Session d'entraînement pour modèles edge."""
    
    def __init__(self, session_id: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        self.start_time = datetime.now()
        self.is_training = False
        self.is_paused = False
        self.current_epoch = 0
        self.total_epochs = config.get('epochs', 50)
        
        self.metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        self.best_accuracy = 0.0
        self.best_model_path = None
        
        logger.info(f"Created training session {session_id}")
    
    def setup_model(self):
        """Configure le modèle selon la configuration."""
        try:
            model_type = self.config['model_type']
            model_config = {
                'num_classes': self.config['num_classes'],
                'pretrained': self.config.get('pretrained', True)
            }
            
            if model_type in ['audio', 'ultra_audio']:
                model_config['input_size'] = self.config.get('input_size', (128, 128))
            
            self.model = create_lightweight_model(model_type, model_config)
            
            # Analyser la complexité
            complexity = get_model_complexity(self.model)
            self.config['model_complexity'] = complexity
            
            # Optimiseur adaptatif selon la taille du modèle
            if complexity['model_size_mb'] < 2:
                # Modèle ultra-léger : Adam avec LR plus élevé
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config.get('learning_rate', 0.001),
                    weight_decay=self.config.get('weight_decay', 1e-4)
                )
            else:
                # Modèle léger : SGD avec momentum
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.config.get('learning_rate', 0.01),
                    momentum=0.9,
                    weight_decay=self.config.get('weight_decay', 1e-4)
                )
            
            # Scheduler adaptatif
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(device)
            
            logger.info(f"Model setup complete: {complexity['model_size_mb']:.2f}MB")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            return False
    
    def setup_data_loaders(self):
        """Configure les data loaders."""
        try:
            # Import des datasets selon le type de modèle
            if self.config['model_type'] in ['audio', 'ultra_audio']:
                from ...Audio_Training_EfficientNet.data_preprocessing import SpectrogramDataset
                dataset_class = SpectrogramDataset
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:  # photo
                from ...Picture_Training_Enhanced.data_preprocessing import WildlifeDataset
                dataset_class = WildlifeDataset
                # Transformations légères pour modèles edge
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.3),  # Moins d'augmentation
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            # Créer les datasets
            train_dataset = dataset_class(
                root_dir=self.config['train_data_path'],
                transform=transform,
                mode='train'
            )
            
            val_dataset = dataset_class(
                root_dir=self.config['val_data_path'],
                transform=transforms.Compose([
                    transforms.Resize((224, 224) if 'photo' in self.config['model_type'] else (128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                mode='val'
            )
            
            # Batch size adaptatif selon la taille du modèle
            complexity = self.config.get('model_complexity', {})
            if complexity.get('model_size_mb', 10) < 2:
                batch_size = 64  # Modèle ultra-léger
            elif complexity.get('model_size_mb', 10) < 5:
                batch_size = 32  # Modèle léger
            else:
                batch_size = 16  # Modèle standard
            
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            
            logger.info(f"Data loaders created: {len(train_dataset)} train, {len(val_dataset)} val")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up data loaders: {e}")
            return False
    
    def train_epoch(self):
        """Entraîne une époque."""
        if not self.model or not self.train_loader:
            return False
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        epoch_start_time = time.time()
        
        try:
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                if not self.is_training or self.is_paused:
                    break
                
                data, targets = data.to(device), targets.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                
                # Gradient clipping pour les modèles légers
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
                
                # Émettre le progrès
                if batch_idx % 10 == 0:
                    progress = (batch_idx + 1) / len(self.train_loader)
                    socketio.emit('training_progress', {
                        'session_id': self.session_id,
                        'epoch': self.current_epoch,
                        'batch_progress': progress,
                        'current_loss': loss.item(),
                        'current_accuracy': correct_predictions / total_samples
                    })
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct_predictions / total_samples
            
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['train_accuracy'].append(accuracy)
            self.metrics['epoch_times'].append(epoch_time)
            self.metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training epoch: {e}")
            return False
    
    def validate_epoch(self):
        """Valide une époque."""
        if not self.model or not self.val_loader:
            return False
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        try:
            with torch.no_grad():
                for data, targets in self.val_loader:
                    data, targets = data.to(device), targets.to(device)
                    
                    outputs = self.model(data)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += targets.size(0)
                    correct_predictions += (predicted == targets).sum().item()
            
            avg_loss = total_loss / len(self.val_loader)
            accuracy = correct_predictions / total_samples
            
            self.metrics['val_loss'].append(avg_loss)
            self.metrics['val_accuracy'].append(accuracy)
            
            # Sauvegarder le meilleur modèle
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_best_model()
            
            # Ajuster le learning rate
            self.scheduler.step(accuracy)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False
    
    def save_best_model(self):
        """Sauvegarde le meilleur modèle."""
        try:
            model_filename = f"{self.session_id}_best_model.pth"
            self.best_model_path = MODELS_FOLDER / model_filename
            
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
                'best_accuracy': self.best_accuracy,
                'config': self.config,
                'metrics': self.metrics
            }, self.best_model_path)
            
            logger.info(f"Best model saved: {self.best_model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False


def start_training_session(config: Dict[str, Any]) -> str:
    """Démarre une nouvelle session d'entraînement."""
    global current_training
    
    session_id = f"edge_training_{int(time.time())}"
    
    try:
        # Créer la session
        session = EdgeTrainingSession(session_id, config)
        training_sessions[session_id] = session
        
        # Setup du modèle et des données
        if not session.setup_model():
            raise Exception("Failed to setup model")
        
        if not session.setup_data_loaders():
            raise Exception("Failed to setup data loaders")
        
        current_training = session
        
        # Démarrer l'entraînement dans un thread séparé
        training_thread = threading.Thread(
            target=run_training_loop,
            args=(session,),
            daemon=True
        )
        training_thread.start()
        
        return session_id
        
    except Exception as e:
        logger.error(f"Error starting training session: {e}")
        raise


def run_training_loop(session: EdgeTrainingSession):
    """Boucle d'entraînement principale."""
    session.is_training = True
    
    try:
        socketio.emit('training_started', {
            'session_id': session.session_id,
            'config': session.config,
            'model_complexity': session.config.get('model_complexity', {})
        })
        
        for epoch in range(session.total_epochs):
            if not session.is_training:
                break
            
            session.current_epoch = epoch + 1
            
            # Entraînement
            epoch_start = time.time()
            success = session.train_epoch()
            if not success:
                break
            
            # Validation
            success = session.validate_epoch()
            if not success:
                break
            
            epoch_time = time.time() - epoch_start
            
            # Émettre les métriques de l'époque
            socketio.emit('epoch_completed', {
                'session_id': session.session_id,
                'epoch': session.current_epoch,
                'train_loss': session.metrics['train_loss'][-1],
                'train_accuracy': session.metrics['train_accuracy'][-1],
                'val_loss': session.metrics['val_loss'][-1],
                'val_accuracy': session.metrics['val_accuracy'][-1],
                'learning_rate': session.metrics['learning_rates'][-1],
                'epoch_time': epoch_time,
                'best_accuracy': session.best_accuracy
            })
            
            # Arrêt anticipé si la validation stagne
            if len(session.metrics['val_accuracy']) > 10:
                recent_acc = session.metrics['val_accuracy'][-10:]
                if max(recent_acc) - min(recent_acc) < 0.01:  # Moins de 1% d'amélioration
                    logger.info("Early stopping triggered - validation accuracy stagnant")
                    break
        
        session.is_training = False
        
        # Sauvegarder le modèle final et générer les métriques
        session.save_best_model()
        
        socketio.emit('training_completed', {
            'session_id': session.session_id,
            'final_accuracy': session.best_accuracy,
            'total_epochs': session.current_epoch,
            'model_path': str(session.best_model_path),
            'metrics': session.metrics
        })
        
        logger.info(f"Training completed: {session.session_id}")
        
    except Exception as e:
        session.is_training = False
        logger.error(f"Error in training loop: {e}")
        socketio.emit('training_error', {
            'session_id': session.session_id,
            'error': str(e)
        })


# Routes Flask
@app.route('/')
def index():
    """Page d'accueil de l'interface d'entraînement edge."""
    return render_template('edge_training.html')


@app.route('/api/models/comparison')
def get_models_comparison():
    """Retourne la comparaison des modèles légers."""
    try:
        comparison = compare_models()
        return jsonify({'success': True, 'models': comparison})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Démarre un entraînement."""
    try:
        config = request.get_json()
        
        # Validation de la configuration
        required_fields = ['model_type', 'num_classes', 'train_data_path', 'val_data_path']
        for field in required_fields:
            if field not in config:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400
        
        session_id = start_training_session(config)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Training started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/stop/<session_id>', methods=['POST'])
def stop_training(session_id):
    """Arrête un entraînement."""
    try:
        if session_id in training_sessions:
            session = training_sessions[session_id]
            session.is_training = False
            
            return jsonify({'success': True, 'message': 'Training stopped'})
        else:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/training/status/<session_id>')
def get_training_status(session_id):
    """Retourne le statut d'un entraînement."""
    try:
        if session_id in training_sessions:
            session = training_sessions[session_id]
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'is_training': session.is_training,
                'current_epoch': session.current_epoch,
                'total_epochs': session.total_epochs,
                'best_accuracy': session.best_accuracy,
                'metrics': session.metrics
            })
        else:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models/download/<session_id>')
def download_model(session_id):
    """Télécharge un modèle entraîné."""
    try:
        if session_id in training_sessions:
            session = training_sessions[session_id]
            if session.best_model_path and session.best_model_path.exists():
                return send_file(
                    session.best_model_path,
                    as_attachment=True,
                    download_name=f"{session_id}_edge_model.pth"
                )
            else:
                return jsonify({'success': False, 'error': 'Model not found'}), 404
        else:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Events SocketIO
@socketio.on('connect')
def handle_connect():
    """Gère les connexions WebSocket."""
    print('Client connecté à l\'interface d\'entraînement edge')
    emit('connected', {'message': 'Connecté au serveur d\'entraînement edge'})


@socketio.on('disconnect')
def handle_disconnect():
    """Gère les déconnexions WebSocket."""
    print('Client déconnecté de l\'interface d\'entraînement edge')


if __name__ == '__main__':
    print("🚀 Démarrage de l'Interface d'Entraînement Edge NightScan")
    print("=" * 60)
    print("🌐 Interface web: http://localhost:5002")
    print("📱 Modèles supportés: Audio léger, Photo MobileNet, Ultra-audio")
    print("📊 Optimisé pour: Modèles <10MB, TensorFlow Lite ready")
    print("=" * 60)
    
    socketio.run(
        app,
        host='0.0.0.0',
        port=5002,
        debug=True,
        allow_unsafe_werkzeug=True
    )