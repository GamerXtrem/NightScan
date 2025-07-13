"""
EfficientNet Configuration for Audio Classification
Specialized configuration for audio spectrogram analysis in NightScan
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class EfficientNetConfig:
    """Configuration for EfficientNet audio models."""
    
    model_name: str = "efficientnet-b1"
    num_classes: int = 6
    pretrained: bool = True
    dropout_rate: float = 0.3
    input_channels: int = 3  # RGB representation of spectrograms
    input_size: tuple = (128, 128)  # Mel spectrogram dimensions
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.input_channels not in [1, 3]:
            raise ValueError("input_channels must be 1 (grayscale) or 3 (RGB)")

class AudioEfficientNet(nn.Module):
    """EfficientNet specialized for audio spectrogram classification."""
    
    def __init__(self, config: EfficientNetConfig):
        super(AudioEfficientNet, self).__init__()
        self.config = config
        
        # Load EfficientNet base model
        if config.model_name == "efficientnet-b0":
            self.backbone = models.efficientnet_b0(pretrained=config.pretrained)
        elif config.model_name == "efficientnet-b1":
            self.backbone = models.efficientnet_b1(pretrained=config.pretrained)
        elif config.model_name == "efficientnet-b2":
            self.backbone = models.efficientnet_b2(pretrained=config.pretrained)
        else:
            raise ValueError(f"Unsupported model: {config.model_name}")
        
        # Adapt first layer for spectrograms if needed
        if config.input_channels != 3:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                config.input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
        
        # Replace classifier for audio classes
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(num_features, config.num_classes)
        )
    
    def forward(self, x):
        """Forward pass for audio spectrogram classification."""
        return self.backbone(x)
    
    def get_feature_extractor(self):
        """Get the feature extraction part of the model."""
        return self.backbone.features
    
    def get_classifier(self):
        """Get the classification head of the model."""
        return self.backbone.classifier

def create_model(config: EfficientNetConfig) -> AudioEfficientNet:
    """
    Factory function to create an AudioEfficientNet model.
    
    Args:
        config: EfficientNetConfig object with model parameters
        
    Returns:
        Configured AudioEfficientNet model
    """
    return AudioEfficientNet(config)

def create_audio_model(
    num_classes: int = 6,
    model_name: str = "efficientnet-b1",
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> AudioEfficientNet:
    """
    Convenience function to create audio model with common parameters.
    
    Args:
        num_classes: Number of audio classes to classify
        model_name: EfficientNet variant to use
        pretrained: Whether to use pretrained ImageNet weights
        dropout_rate: Dropout rate for regularization
        
    Returns:
        AudioEfficientNet model ready for training
    """
    config = EfficientNetConfig(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    return create_model(config)

def get_audio_classes():
    """Get the standard audio classes for NightScan."""
    return [
        "bird_song",
        "mammal_call", 
        "insect_sound",
        "amphibian_call",
        "environmental_sound",
        "unknown_species"
    ]

def print_model_info(model: AudioEfficientNet):
    """Print detailed information about the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.config.model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
    print(f"Input size: {model.config.input_size}")
    print(f"Number of classes: {model.config.num_classes}")
    print(f"Dropout rate: {model.config.dropout_rate}")

if __name__ == "__main__":
    # Test model creation
    print("Testing AudioEfficientNet model creation...")
    
    # Create test model
    model = create_audio_model()
    print_model_info(model)
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 128, 128)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nTest forward pass:")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output probabilities: {torch.softmax(output, dim=1)}")
    
    print("\nAudioEfficientNet model test completed successfully!")