"""
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
        from Audio_Training.scripts.preprocess import preprocess_audio_data
        return preprocess_audio_data(data_path)


class ImageTrainer(BaseTrainer):
    """Specialized trainer for image models."""
    
    def prepare_data(self, data_path: str):
        """Prepare image training data."""
        # Image-specific data preparation
        from Picture_Training.scripts.prepare_csv import prepare_image_data
        return prepare_image_data(data_path)
