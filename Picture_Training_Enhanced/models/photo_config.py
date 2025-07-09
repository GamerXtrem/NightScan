"""
Photo Model Configuration for Wildlife Image Classification

Advanced configuration system for various deep learning architectures
optimized for nocturnal wildlife image classification.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


@dataclass
class PhotoConfig:
    """Configuration for photo classification models."""
    
    # Model architecture
    model_name: str = "efficientnet-b1"  # efficientnet-b0 to b7, resnet18/34/50/101, vit_b_16
    architecture: str = "efficientnet"  # efficientnet, resnet, vit
    num_classes: int = 8  # Default for wildlife classification
    pretrained: bool = True
    dropout_rate: float = 0.3
    
    # Input specifications
    input_size: Tuple[int, int] = (224, 224)  # Standard ImageNet size
    num_channels: int = 3  # RGB images
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15  # Early stopping patience
    
    # Optimization
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, step, reduce_on_plateau, one_cycle
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_degrees: int = 15
    color_jitter_strength: float = 0.3
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.2
    cutmix_prob: float = 0.5
    mixup_alpha: float = 0.2
    
    # Class balancing
    use_class_weights: bool = True
    class_weights: Optional[Dict[str, float]] = None
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Fine-tuning options
    freeze_backbone: bool = False
    unfreeze_at_epoch: int = 10
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Validate architecture and model name compatibility
        if self.architecture == "efficientnet" and not self.model_name.startswith("efficientnet"):
            raise ValueError(f"Model {self.model_name} incompatible with {self.architecture} architecture")
        elif self.architecture == "resnet" and not self.model_name.startswith("resnet"):
            raise ValueError(f"Model {self.model_name} incompatible with {self.architecture} architecture")
        elif self.architecture == "vit" and not self.model_name.startswith("vit"):
            raise ValueError(f"Model {self.model_name} incompatible with {self.architecture} architecture")
        
        # Default class weights for wildlife classification
        if self.use_class_weights and self.class_weights is None:
            self.class_weights = {
                "bat": 1.2,
                "owl": 1.0,
                "raccoon": 0.9,
                "opossum": 1.1,
                "deer": 0.8,
                "fox": 1.3,
                "coyote": 1.4,
                "unknown": 0.7
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
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
            "rotation_degrees": self.rotation_degrees,
            "color_jitter_strength": self.color_jitter_strength,
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "vertical_flip_prob": self.vertical_flip_prob,
            "cutmix_prob": self.cutmix_prob,
            "mixup_alpha": self.mixup_alpha,
            "use_class_weights": self.use_class_weights,
            "class_weights": self.class_weights,
            "device": self.device,
            "freeze_backbone": self.freeze_backbone,
            "unfreeze_at_epoch": self.unfreeze_at_epoch
        }


class PhotoClassifier(nn.Module):
    """Unified photo classifier supporting multiple architectures."""
    
    def __init__(self, config: PhotoConfig):
        super().__init__()
        self.config = config
        
        if config.architecture == "efficientnet":
            self.model = self._create_efficientnet(config)
        elif config.architecture == "resnet":
            self.model = self._create_resnet(config)
        elif config.architecture == "vit":
            self.model = self._create_vit(config)
        else:
            raise ValueError(f"Unsupported architecture: {config.architecture}")
        
        self.feature_extractor = self._get_feature_extractor()
        
        if config.freeze_backbone:
            self.freeze_backbone()
    
    def _create_efficientnet(self, config: PhotoConfig) -> nn.Module:
        """Create EfficientNet model."""
        if config.pretrained:
            model = EfficientNet.from_pretrained(
                config.model_name,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate
            )
        else:
            model = EfficientNet.from_name(
                config.model_name,
                num_classes=config.num_classes,
                dropout_rate=config.dropout_rate
            )
        
        # Custom classification head
        model._fc = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(model._fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate / 2),
            nn.Linear(512, config.num_classes)
        )
        
        return model
    
    def _create_resnet(self, config: PhotoConfig) -> nn.Module:
        """Create ResNet model."""
        model_mapping = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101
        }
        
        if config.model_name not in model_mapping:
            raise ValueError(f"Unsupported ResNet model: {config.model_name}")
        
        weights = getattr(models, f"ResNet{config.model_name[6:]}_Weights").DEFAULT if config.pretrained else None
        model = model_mapping[config.model_name](weights=weights)
        
        # Custom classification head
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate / 2),
            nn.Linear(512, config.num_classes)
        )
        
        return model
    
    def _create_vit(self, config: PhotoConfig) -> nn.Module:
        """Create Vision Transformer model."""
        if config.model_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if config.pretrained else None
            model = models.vit_b_16(weights=weights)
        elif config.model_name == "vit_b_32":
            weights = models.ViT_B_32_Weights.DEFAULT if config.pretrained else None
            model = models.vit_b_32(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT model: {config.model_name}")
        
        # Custom classification head
        num_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate / 2),
            nn.Linear(512, config.num_classes)
        )
        
        return model
    
    def _get_feature_extractor(self):
        """Get feature extractor based on architecture."""
        if self.config.architecture == "efficientnet":
            return lambda x: self.model.extract_features(x)
        elif self.config.architecture == "resnet":
            return lambda x: torch.nn.Sequential(*list(self.model.children())[:-1])(x).flatten(1)
        elif self.config.architecture == "vit":
            def vit_features(x):
                x = self.model._process_input(x)
                n = x.shape[0]
                batch_class_token = self.model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                return self.model.encoder(x)[:, 0]
            return vit_features
        else:
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification."""
        return self.feature_extractor(x)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        if self.config.architecture == "efficientnet":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model._fc.parameters():
                param.requires_grad = True
        elif self.config.architecture == "resnet":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif self.config.architecture == "vit":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.heads.parameters():
                param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True


def create_model(config: PhotoConfig) -> PhotoClassifier:
    """Create a photo classifier with the given configuration."""
    return PhotoClassifier(config)


def get_model_info(model: PhotoClassifier) -> Dict[str, Any]:
    """Get information about the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "architecture": model.config.architecture,
        "model_name": model.config.model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
        "input_size": model.config.input_size,
        "num_classes": model.config.num_classes,
        "dropout_rate": model.config.dropout_rate
    }


# Pre-configured models for different use cases
CONFIGS = {
    "efficientnet_b0_fast": PhotoConfig(
        model_name="efficientnet-b0",
        architecture="efficientnet",
        batch_size=64,
        learning_rate=2e-4,
        epochs=50,
        patience=10
    ),
    
    "efficientnet_b1_balanced": PhotoConfig(
        model_name="efficientnet-b1",
        architecture="efficientnet",
        batch_size=32,
        learning_rate=1e-4,
        epochs=100,
        patience=15
    ),
    
    "efficientnet_b2_quality": PhotoConfig(
        model_name="efficientnet-b2",
        architecture="efficientnet",
        batch_size=16,
        learning_rate=5e-5,
        epochs=150,
        patience=20
    ),
    
    "resnet18_fast": PhotoConfig(
        model_name="resnet18",
        architecture="resnet",
        batch_size=64,
        learning_rate=1e-3,
        epochs=50,
        patience=10
    ),
    
    "resnet50_balanced": PhotoConfig(
        model_name="resnet50",
        architecture="resnet",
        batch_size=32,
        learning_rate=1e-4,
        epochs=100,
        patience=15
    ),
    
    "vit_b_16_quality": PhotoConfig(
        model_name="vit_b_16",
        architecture="vit",
        batch_size=16,
        learning_rate=5e-5,
        epochs=120,
        patience=20,
        freeze_backbone=True,
        unfreeze_at_epoch=20
    )
}


def get_config(config_name: str = "efficientnet_b1_balanced") -> PhotoConfig:
    """Get a pre-configured photo classification configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    
    return CONFIGS[config_name]


if __name__ == "__main__":
    # Test the configuration
    config = get_config("efficientnet_b1_balanced")
    print("Photo Configuration:")
    print(f"Architecture: {config.architecture}")
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