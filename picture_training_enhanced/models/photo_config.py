"""
EfficientNet Configuration for Photo Classification
Specialized configuration for wildlife photo analysis in NightScan
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class PhotoConfig:
    """Configuration for EfficientNet photo models."""
    
    model_name: str = "efficientnet-b1"
    architecture: str = "efficientnet"  # For compatibility with existing code
    num_classes: int = 8
    pretrained: bool = True
    dropout_rate: float = 0.3
    input_channels: int = 3  # RGB images
    input_size: tuple = (224, 224)  # Standard image classification size
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.input_channels not in [1, 3, 4]:
            raise ValueError("input_channels must be 1 (grayscale), 3 (RGB), or 4 (RGBA)")

class PhotoEfficientNet(nn.Module):
    """EfficientNet specialized for wildlife photo classification."""
    
    def __init__(self, config: PhotoConfig):
        super(PhotoEfficientNet, self).__init__()
        self.config = config
        
        # Load EfficientNet base model
        if config.model_name == "efficientnet-b0":
            self.backbone = models.efficientnet_b0(pretrained=config.pretrained)
        elif config.model_name == "efficientnet-b1":
            self.backbone = models.efficientnet_b1(pretrained=config.pretrained)
        elif config.model_name == "efficientnet-b2":
            self.backbone = models.efficientnet_b2(pretrained=config.pretrained)
        elif config.model_name == "efficientnet-b3":
            self.backbone = models.efficientnet_b3(pretrained=config.pretrained)
        else:
            raise ValueError(f"Unsupported model: {config.model_name}")
        
        # Adapt first layer for different input channels if needed
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
        
        # Replace classifier for wildlife classes
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(num_features, config.num_classes)
        )
    
    def forward(self, x):
        """Forward pass for wildlife photo classification."""
        return self.backbone(x)
    
    def get_feature_extractor(self):
        """Get the feature extraction part of the model."""
        return self.backbone.features
    
    def get_classifier(self):
        """Get the classification head of the model."""
        return self.backbone.classifier
    
    def extract_features(self, x):
        """Extract features before classification."""
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features

def create_model(config: PhotoConfig) -> PhotoEfficientNet:
    """
    Factory function to create a PhotoEfficientNet model.
    
    Args:
        config: PhotoConfig object with model parameters
        
    Returns:
        Configured PhotoEfficientNet model
    """
    return PhotoEfficientNet(config)

def create_photo_model(
    num_classes: int = 8,
    model_name: str = "efficientnet-b1",
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> PhotoEfficientNet:
    """
    Convenience function to create photo model with common parameters.
    
    Args:
        num_classes: Number of wildlife classes to classify
        model_name: EfficientNet variant to use
        pretrained: Whether to use pretrained ImageNet weights
        dropout_rate: Dropout rate for regularization
        
    Returns:
        PhotoEfficientNet model ready for training
    """
    config = PhotoConfig(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    return create_model(config)

def get_wildlife_classes():
    """Get the standard wildlife classes for NightScan."""
    return [
        "bat",
        "owl", 
        "raccoon",
        "opossum",
        "deer",
        "fox",
        "coyote",
        "unknown"
    ]

def get_class_weights():
    """Get class weights for imbalanced dataset handling."""
    # These weights can be adjusted based on dataset statistics
    return {
        "bat": 1.5,        # Rare class
        "owl": 1.2,        # Less common
        "raccoon": 1.0,    # Baseline
        "opossum": 1.1,    # Slightly less common
        "deer": 0.8,       # More common
        "fox": 1.0,        # Baseline
        "coyote": 1.3,     # Less common
        "unknown": 0.5     # Background class
    }

def print_model_info(model: PhotoEfficientNet):
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

def create_data_transforms():
    """Create data transforms for training and validation."""
    from torchvision import transforms
    
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

if __name__ == "__main__":
    # Test model creation
    print("Testing PhotoEfficientNet model creation...")
    
    # Create test model
    model = create_photo_model()
    print_model_info(model)
    
    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        features = model.extract_features(dummy_input)
        
        print(f"\nTest forward pass:")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Output probabilities: {torch.softmax(output, dim=1)}")
    
    # Print class information
    print(f"\nWildlife classes: {get_wildlife_classes()}")
    print(f"Class weights: {get_class_weights()}")
    
    print("\nPhotoEfficientNet model test completed successfully!")