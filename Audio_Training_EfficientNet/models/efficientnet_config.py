"""
EfficientNet Configuration for Wildlife Audio Classification

Optimized configuration for EfficientNet models processing mel-spectrograms
for wildlife audio classification tasks.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


@dataclass
class EfficientNetConfig:
    """Configuration for EfficientNet model."""
    
    # Model architecture
    model_name: str = "efficientnet-b1"  # b0 to b7 available
    num_classes: int = 6
    pretrained: bool = True
    dropout_rate: float = 0.3
    
    # Input specifications
    input_size: Tuple[int, int] = (224, 224)
    num_channels: int = 3  # RGB channels for spectrograms
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10  # Early stopping patience
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, reduce_on_plateau
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Data augmentation
    use_augmentation: bool = True
    spec_augment_freq_mask: int = 30
    spec_augment_time_mask: int = 40
    mixup_alpha: float = 0.2
    
    # Class weights (for imbalanced datasets)
    use_class_weights: bool = True
    class_weights: Optional[Dict[str, float]] = None
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Default class weights for wildlife audio
        if self.use_class_weights and self.class_weights is None:
            self.class_weights = {
                "bird_song": 1.0,
                "mammal_call": 1.2,
                "insect_sound": 0.8,
                "amphibian_call": 1.1,
                "environmental_sound": 0.9,
                "unknown_species": 0.6
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "dropout_rate": self.dropout_rate,
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "mixed_precision": self.mixed_precision,
            "gradient_clipping": self.gradient_clipping,
            "use_augmentation": self.use_augmentation,
            "spec_augment_freq_mask": self.spec_augment_freq_mask,
            "spec_augment_time_mask": self.spec_augment_time_mask,
            "mixup_alpha": self.mixup_alpha,
            "use_class_weights": self.use_class_weights,
            "class_weights": self.class_weights,
            "device": self.device
        }


class EfficientNetClassifier(nn.Module):
    """EfficientNet classifier for wildlife audio spectrograms."""
    
    def __init__(self, config: EfficientNetConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained EfficientNet
        if config.pretrained:
            self.backbone = EfficientNet.from_pretrained(
                config.model_name,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate
            )
        else:
            self.backbone = EfficientNet.from_name(
                config.model_name,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate
            )
        
        # Adapt input layer if needed
        if config.num_channels != 3:
            self._adapt_input_layer(config.num_channels)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.backbone._fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate / 2),
            nn.Linear(512, config.num_classes)
        )
        
        # Replace the default classifier
        self.backbone._fc = self.classifier
    
    def _adapt_input_layer(self, num_channels: int):
        """Adapt the first convolutional layer for different input channels."""
        if num_channels == 3:
            return
        
        # Get the first convolution layer
        first_conv = self.backbone._conv_stem
        
        # Create new convolution with different input channels
        new_conv = nn.Conv2d(
            num_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        # Copy weights for RGB channels if num_channels > 3
        if num_channels > 3:
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = first_conv.weight
                # Initialize additional channels with average of RGB
                avg_weight = first_conv.weight.mean(dim=1, keepdim=True)
                for i in range(3, num_channels):
                    new_conv.weight[:, i, :, :] = avg_weight.squeeze(1)
        
        self.backbone._conv_stem = new_conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.backbone(x)
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps before classification."""
        return self.backbone.extract_features(x)
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True


def create_model(config: EfficientNetConfig) -> EfficientNetClassifier:
    """Create an EfficientNet classifier with the given configuration."""
    return EfficientNetClassifier(config)


def get_model_info(model: EfficientNetClassifier) -> Dict[str, Any]:
    """Get information about the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_name": model.config.model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "input_size": model.config.input_size,
        "num_classes": model.config.num_classes,
        "dropout_rate": model.config.dropout_rate
    }


# Pre-configured models for different use cases
CONFIGS = {
    "efficientnet_b0_fast": EfficientNetConfig(
        model_name="efficientnet-b0",
        batch_size=64,
        learning_rate=2e-4,
        epochs=30
    ),
    
    "efficientnet_b1_balanced": EfficientNetConfig(
        model_name="efficientnet-b1",
        batch_size=32,
        learning_rate=1e-4,
        epochs=50
    ),
    
    "efficientnet_b2_quality": EfficientNetConfig(
        model_name="efficientnet-b2",
        batch_size=16,
        learning_rate=5e-5,
        epochs=100,
        patience=15
    ),
    
    "efficientnet_b3_best": EfficientNetConfig(
        model_name="efficientnet-b3",
        batch_size=8,
        learning_rate=2e-5,
        epochs=150,
        patience=20
    )
}


def get_config(config_name: str = "efficientnet_b1_balanced") -> EfficientNetConfig:
    """Get a pre-configured EfficientNet configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]


if __name__ == "__main__":
    # Test the configuration
    config = get_config("efficientnet_b1_balanced")
    print("EfficientNet Configuration:")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    # Test model creation
    model = create_model(config)
    model_info = get_model_info(model)
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"{key}: {value}")