#!/usr/bin/env python3
"""
Configuration et modèle EfficientNet avec support dynamique du nombre de classes.
Sélection automatique de l'architecture selon la complexité.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DynamicPhotoConfig:
    """Configuration dynamique pour les modèles photo."""
    
    num_classes: int
    model_name: Optional[str] = None  # Auto-sélection si None
    architecture: str = "efficientnet"
    pretrained: bool = True
    dropout_rate: float = 0.3
    input_channels: int = 3
    input_size: Tuple[int, int] = (224, 224)
    
    # Paramètres additionnels pour l'optimisation
    use_attention: bool = False
    use_mixup: bool = False
    label_smoothing: float = 0.0
    
    def __post_init__(self):
        """Validation et auto-configuration après initialisation."""
        if self.num_classes <= 0:
            raise ValueError("num_classes doit être positif")
        
        # Auto-sélection du modèle selon le nombre de classes
        if self.model_name is None:
            self.model_name = self._auto_select_model()
            logger.info(f"Modèle auto-sélectionné: {self.model_name}")
        
        # Ajuster le dropout selon la complexité
        if self.num_classes < 10:
            self.dropout_rate = min(self.dropout_rate, 0.2)
        elif self.num_classes > 50:
            self.dropout_rate = max(self.dropout_rate, 0.4)
        
        # Validation des paramètres
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError("dropout_rate doit être entre 0 et 1")
        if not (0 <= self.label_smoothing <= 1):
            raise ValueError("label_smoothing doit être entre 0 et 1")
    
    def _auto_select_model(self) -> str:
        """
        Sélectionne automatiquement le modèle selon le nombre de classes.
        
        Returns:
            Nom du modèle EfficientNet approprié
        """
        if self.num_classes < 10:
            return "efficientnet-b0"  # Modèle léger pour peu de classes
        elif self.num_classes < 30:
            return "efficientnet-b1"  # Modèle moyen
        elif self.num_classes < 100:
            return "efficientnet-b2"  # Modèle plus complexe
        elif self.num_classes < 500:
            return "efficientnet-b3"  # Modèle lourd
        else:
            return "efficientnet-b4"  # Modèle très lourd pour beaucoup de classes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        return {
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'architecture': self.architecture,
            'pretrained': self.pretrained,
            'dropout_rate': self.dropout_rate,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'use_attention': self.use_attention,
            'use_mixup': self.use_mixup,
            'label_smoothing': self.label_smoothing
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DynamicPhotoConfig':
        """Crée une configuration depuis un dictionnaire."""
        return cls(**config_dict)


class AttentionModule(nn.Module):
    """Module d'attention spatial pour améliorer les performances."""
    
    def __init__(self, in_channels: int):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global Average Pooling
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        
        # Attention weights
        attention = self.conv1(avg_pool)
        attention = torch.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention


class DynamicPhotoEfficientNet(nn.Module):
    """EfficientNet dynamique pour classification d'images."""
    
    def __init__(self, config: DynamicPhotoConfig):
        super(DynamicPhotoEfficientNet, self).__init__()
        self.config = config
        
        # Charger le modèle EfficientNet approprié
        self.backbone = self._load_efficientnet()
        
        # Adapter pour différents nombres de canaux si nécessaire
        if config.input_channels != 3:
            self._adapt_input_channels()
        
        # Obtenir le nombre de features
        num_features = self._get_num_features()
        
        # Ajouter module d'attention si demandé
        self.attention = None
        if config.use_attention:
            self.attention = AttentionModule(num_features)
        
        # Remplacer le classificateur
        self._build_classifier(num_features)
        
        # Initialiser les poids du nouveau classificateur
        self._init_weights()
    
    def _load_efficientnet(self) -> nn.Module:
        """Charge le modèle EfficientNet approprié."""
        model_name = self.config.model_name
        
        if model_name == "efficientnet-b0":
            model = models.efficientnet_b0(pretrained=self.config.pretrained)
        elif model_name == "efficientnet-b1":
            model = models.efficientnet_b1(pretrained=self.config.pretrained)
        elif model_name == "efficientnet-b2":
            model = models.efficientnet_b2(pretrained=self.config.pretrained)
        elif model_name == "efficientnet-b3":
            model = models.efficientnet_b3(pretrained=self.config.pretrained)
        elif model_name == "efficientnet-b4":
            model = models.efficientnet_b4(pretrained=self.config.pretrained)
        else:
            # Fallback vers B1
            logger.warning(f"Modèle {model_name} non supporté, utilisation de B1")
            model = models.efficientnet_b1(pretrained=self.config.pretrained)
        
        return model
    
    def _adapt_input_channels(self):
        """Adapte le modèle pour un nombre différent de canaux d'entrée."""
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            self.config.input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
    
    def _get_num_features(self) -> int:
        """Obtient le nombre de features du backbone."""
        # Pour EfficientNet, le nombre de features est dans le dernier layer
        return self.backbone.classifier[1].in_features
    
    def _build_classifier(self, num_features: int):
        """Construit un classificateur adapté au nombre de classes."""
        layers = []
        
        # Dropout principal
        layers.append(nn.Dropout(p=self.config.dropout_rate))
        
        # Pour beaucoup de classes, ajouter une couche intermédiaire
        if self.config.num_classes > 100:
            hidden_dim = min(512, self.config.num_classes * 2)
            layers.extend([
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=self.config.dropout_rate * 0.7),
                nn.Linear(hidden_dim, self.config.num_classes)
            ])
        else:
            # Classification directe pour peu de classes
            layers.append(nn.Linear(num_features, self.config.num_classes))
        
        self.backbone.classifier = nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialise les poids du nouveau classificateur."""
        for module in self.backbone.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass du modèle."""
        # Extraction des features
        features = self.backbone.features(x)
        
        # Attention si activée
        if self.attention is not None:
            features = self.attention(features)
        
        # Pooling et classification
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.backbone.classifier(features)
        
        return output
    
    def extract_features(self, x):
        """Extrait les features avant la classification."""
        features = self.backbone.features(x)
        if self.attention is not None:
            features = self.attention(features)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features
    
    def freeze_backbone(self, freeze: bool = True):
        """Gèle ou dégèle le backbone pour le fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze
    
    def get_param_groups(self, base_lr: float = 0.001) -> list:
        """
        Retourne les groupes de paramètres avec différents learning rates.
        Utile pour le fine-tuning avec des taux d'apprentissage différenciés.
        """
        # Backbone avec learning rate réduit
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': base_lr * 0.1},  # LR réduit pour backbone
            {'params': classifier_params, 'lr': base_lr}       # LR normal pour classifier
        ]
        
        return param_groups


def create_dynamic_model(num_classes: int,
                        model_name: Optional[str] = None,
                        pretrained: bool = True,
                        dropout_rate: float = 0.3,
                        use_attention: bool = False,
                        **kwargs) -> DynamicPhotoEfficientNet:
    """
    Fonction factory pour créer un modèle dynamique.
    
    Args:
        num_classes: Nombre de classes à prédire
        model_name: Nom du modèle (auto-sélection si None)
        pretrained: Utiliser les poids pré-entraînés
        dropout_rate: Taux de dropout
        use_attention: Utiliser un module d'attention
        **kwargs: Autres paramètres de configuration
    
    Returns:
        Modèle DynamicPhotoEfficientNet configuré
    """
    config = DynamicPhotoConfig(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_attention=use_attention,
        **kwargs
    )
    
    return DynamicPhotoEfficientNet(config)


def estimate_model_size(num_classes: int) -> Dict[str, Any]:
    """
    Estime la taille et les besoins en ressources du modèle.
    
    Args:
        num_classes: Nombre de classes
    
    Returns:
        Dictionnaire avec les estimations
    """
    # Créer une config temporaire
    config = DynamicPhotoConfig(num_classes=num_classes)
    
    # Paramètres par modèle (approximatifs)
    model_params = {
        'efficientnet-b0': 5.3e6,
        'efficientnet-b1': 7.8e6,
        'efficientnet-b2': 9.1e6,
        'efficientnet-b3': 12.2e6,
        'efficientnet-b4': 19.3e6,
    }
    
    base_params = model_params.get(config.model_name, 7.8e6)
    
    # Ajouter les paramètres du classificateur
    if config.num_classes > 100:
        hidden_dim = min(512, config.num_classes * 2)
        classifier_params = 1280 * hidden_dim + hidden_dim * config.num_classes
    else:
        classifier_params = 1280 * config.num_classes
    
    total_params = base_params + classifier_params
    
    # Estimation de la mémoire (4 bytes par paramètre)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    # Estimation VRAM pour batch size 32
    # Approximation: modèle + gradients + activations
    vram_estimate_gb = (model_size_mb * 3 + 500) / 1024  # +500MB pour activations
    
    return {
        'model_name': config.model_name,
        'num_parameters': int(total_params),
        'model_size_mb': round(model_size_mb, 1),
        'estimated_vram_gb': round(vram_estimate_gb, 1),
        'recommended_batch_size': 64 if vram_estimate_gb < 8 else 32,
        'dropout_rate': config.dropout_rate
    }


if __name__ == "__main__":
    # Tests
    print("Test du modèle dynamique\n" + "="*50)
    
    # Test avec différents nombres de classes
    for num_classes in [5, 15, 50, 150, 500]:
        print(f"\n📊 Nombre de classes: {num_classes}")
        
        # Estimer les ressources
        estimates = estimate_model_size(num_classes)
        print(f"  Modèle sélectionné: {estimates['model_name']}")
        print(f"  Paramètres: {estimates['num_parameters']:,}")
        print(f"  Taille: {estimates['model_size_mb']} MB")
        print(f"  VRAM estimée: {estimates['estimated_vram_gb']} GB")
        print(f"  Batch size recommandé: {estimates['recommended_batch_size']}")
        
        # Créer et tester le modèle
        model = create_dynamic_model(num_classes=num_classes)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        
        # Test extraction de features
        features = model.extract_features(dummy_input)
        print(f"  Features shape: {features.shape}")