"""
Modèles Légers pour l'Entraînement Edge NightScan

Ce module définit des architectures optimisées pour mobile :
- MobileNetV3 pour les images
- Lightweight CNN pour les spectrogrammes audio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LightweightAudioModel(nn.Module):
    """
    Modèle CNN léger optimisé pour les spectrogrammes audio.
    Conçu pour être <3MB après quantification.
    """
    
    def __init__(self, num_classes: int = 6, input_size: tuple = (128, 128)):
        super(LightweightAudioModel, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Couches convolutionnelles légères
        self.features = nn.Sequential(
            # Block 1: 3x128x128 -> 32x64x64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32x64x64 -> 64x32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64x32x32 -> 96x16x16
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            # Block 4: 96x16x16 -> 128x8x8
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 5: 128x8x8 -> 160x4x4
            nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling: 160x4x4 -> 160x1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classificateur léger
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(160, 80),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(80, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids avec Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "LightweightAudioModel",
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_size_mb": (total_params * 4) / (1024 * 1024),  # Assuming float32
        }


class LightweightPhotoModel(nn.Module):
    """
    Modèle basé sur MobileNetV3-Small optimisé pour les images nocturnes.
    Conçu pour être <8MB après quantification.
    """
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super(LightweightPhotoModel, self).__init__()
        self.num_classes = num_classes
        
        # Utiliser MobileNetV3-Small comme backbone
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Modifier le classificateur pour nos classes
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1, inplace=True),
            nn.Linear(128, num_classes)
        )
        
        # Optimisations pour les images nocturnes
        self._adapt_for_night_vision()
    
    def _adapt_for_night_vision(self):
        """Adapte le modèle pour les images nocturnes."""
        # Modifier les premières couches pour mieux traiter les images sombres
        first_conv = self.backbone.features[0][0]
        
        # Réduire le stride pour conserver plus de détails
        if first_conv.stride == (2, 2):
            first_conv.stride = (1, 1)
            first_conv.padding = (1, 1)
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "LightweightPhotoModel",
            "backbone": "MobileNetV3-Small",
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_size_mb": (total_params * 4) / (1024 * 1024),
        }


class DepthwiseSeparableConv2d(nn.Module):
    """
    Convolution séparable en profondeur pour réduire les paramètres.
    Utilisée dans les architectures mobiles.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class UltraLightAudioModel(nn.Module):
    """
    Modèle ultra-léger pour audio utilisant des convolutions séparables.
    Objectif: <1MB après quantification.
    """
    
    def __init__(self, num_classes: int = 6):
        super(UltraLightAudioModel, self).__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            # Block 1: 3 -> 16
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Block 2: 16 -> 32 (depthwise separable)
            DepthwiseSeparableConv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 3: 32 -> 48 (depthwise separable)
            DepthwiseSeparableConv2d(32, 48, 3, 2, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            # Block 4: 48 -> 64 (depthwise separable)
            DepthwiseSeparableConv2d(48, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations du modèle."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "UltraLightAudioModel",
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_size_mb": (total_params * 4) / (1024 * 1024),
        }


def create_lightweight_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function pour créer des modèles légers.
    
    Args:
        model_type: 'audio', 'photo', 'ultra_audio'
        config: Configuration du modèle
        
    Returns:
        nn.Module: Modèle léger
    """
    if model_type == 'audio':
        return LightweightAudioModel(
            num_classes=config.get('num_classes', 6),
            input_size=config.get('input_size', (128, 128))
        )
    elif model_type == 'photo':
        return LightweightPhotoModel(
            num_classes=config.get('num_classes', 8),
            pretrained=config.get('pretrained', True)
        )
    elif model_type == 'ultra_audio':
        return UltraLightAudioModel(
            num_classes=config.get('num_classes', 6)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_complexity(model: nn.Module) -> Dict[str, Any]:
    """
    Analyse la complexité d'un modèle.
    
    Args:
        model: Modèle PyTorch
        
    Returns:
        Dict: Métriques de complexité
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimer les FLOPs pour une entrée standard
    def estimate_flops(model, input_size=(1, 3, 128, 128)):
        """Estimation approximative des FLOPs."""
        flops = 0
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # FLOPs = kernel_size * in_channels * out_channels * output_size
                kernel_flops = module.kernel_size[0] * module.kernel_size[1]
                output_elements = 1  # Approximation
                flops += kernel_flops * module.in_channels * module.out_channels * output_elements
            elif isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features
        return flops
    
    estimated_flops = estimate_flops(model)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": (total_params * 4) / (1024 * 1024),  # float32
        "estimated_flops": estimated_flops,
        "efficiency_score": estimated_flops / total_params if total_params > 0 else 0
    }


def compare_models() -> Dict[str, Dict[str, Any]]:
    """
    Compare tous les modèles légers disponibles.
    
    Returns:
        Dict: Comparaison des modèles
    """
    models_to_compare = {
        'audio': create_lightweight_model('audio', {'num_classes': 6}),
        'ultra_audio': create_lightweight_model('ultra_audio', {'num_classes': 6}),
        'photo': create_lightweight_model('photo', {'num_classes': 8})
    }
    
    comparison = {}
    
    for name, model in models_to_compare.items():
        model.eval()
        complexity = get_model_complexity(model)
        model_info = model.get_model_info()
        
        comparison[name] = {
            **model_info,
            **complexity,
            "mobile_ready": complexity["model_size_mb"] < 10,  # <10MB
            "ultra_light": complexity["model_size_mb"] < 5     # <5MB
        }
    
    return comparison


if __name__ == "__main__":
    # Test et comparaison des modèles
    print("🔍 Comparaison des Modèles Légers NightScan")
    print("=" * 50)
    
    comparison = compare_models()
    
    for model_name, metrics in comparison.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Paramètres: {metrics['total_parameters']:,}")
        print(f"  Taille: {metrics['model_size_mb']:.2f} MB")
        print(f"  FLOPs estimés: {metrics['estimated_flops']:,}")
        print(f"  Mobile ready: {'✅' if metrics['mobile_ready'] else '❌'}")
        print(f"  Ultra léger: {'✅' if metrics['ultra_light'] else '❌'}")
    
    print("\n" + "=" * 50)
    print("✅ Tous les modèles légers définis avec succès")