"""
Configuration EfficientNet optimisée pour les datasets à grande échelle (1000+ classes)
Inclut des optimisations mémoire et des architectures adaptées
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class EfficientNetLargeScale(nn.Module):
    """
    EfficientNet adapté pour la classification à grande échelle.
    Utilise une projection intermédiaire pour réduire la charge mémoire.
    """
    
    def __init__(self, 
                 num_classes: int = 1500,
                 model_name: str = 'efficientnet-b3',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 intermediate_dim: int = 512):
        """
        Args:
            num_classes: Nombre de classes de sortie
            model_name: Version d'EfficientNet à utiliser
            pretrained: Utiliser les poids pré-entraînés
            dropout_rate: Taux de dropout
            intermediate_dim: Dimension de la couche intermédiaire
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.intermediate_dim = intermediate_dim
        
        # Mapper les noms vers les fonctions torchvision
        model_mapping = {
            'efficientnet-b0': models.efficientnet_b0,
            'efficientnet-b1': models.efficientnet_b1,
            'efficientnet-b2': models.efficientnet_b2,
            'efficientnet-b3': models.efficientnet_b3,
            'efficientnet-b4': models.efficientnet_b4,
            'efficientnet-b5': models.efficientnet_b5,
            'efficientnet-b6': models.efficientnet_b6,
            'efficientnet-b7': models.efficientnet_b7,
        }
        
        if model_name not in model_mapping:
            raise ValueError(f"Model {model_name} non supporté")
        
        # Charger le modèle de base
        if pretrained:
            weights = 'DEFAULT'
        else:
            weights = None
        
        self.base_model = model_mapping[model_name](weights=weights)
        
        # Obtenir la dimension des features
        in_features = self.base_model.classifier[1].in_features
        
        # Remplacer le classifier par une architecture à deux niveaux
        # pour réduire le nombre de paramètres avec beaucoup de classes
        self.base_model.classifier = nn.Identity()
        
        # Projection en deux étapes pour économiser la mémoire
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(intermediate_dim, num_classes)
        )
        
        # Initialisation des couches
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids des nouvelles couches."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Features extraction
        features = self.base_model(x)
        
        # Classification
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrait les features sans classification."""
        return self.base_model(x)
    
    def freeze_backbone(self):
        """Gèle les couches du backbone pour le fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Dégèle les couches du backbone."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def get_param_groups(self, base_lr: float = 0.001):
        """
        Retourne des groupes de paramètres avec des learning rates différents.
        Utile pour le fine-tuning avec des LR différents pour backbone et classifier.
        """
        return [
            {'params': self.base_model.parameters(), 'lr': base_lr * 0.1},
            {'params': self.classifier.parameters(), 'lr': base_lr}
        ]


def create_large_scale_model(
    num_classes: int = 1500,
    model_name: str = 'efficientnet-b3',
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    checkpoint_path: Optional[str] = None
) -> EfficientNetLargeScale:
    """
    Crée un modèle EfficientNet optimisé pour les grandes échelles.
    
    Args:
        num_classes: Nombre de classes
        model_name: Version d'EfficientNet
        pretrained: Utiliser les poids pré-entraînés
        dropout_rate: Taux de dropout
        checkpoint_path: Chemin vers un checkpoint à charger
    
    Returns:
        Modèle EfficientNetLargeScale
    """
    # Dimensions intermédiaires recommandées selon le modèle
    intermediate_dims = {
        'efficientnet-b0': 256,
        'efficientnet-b1': 256,
        'efficientnet-b2': 384,
        'efficientnet-b3': 512,
        'efficientnet-b4': 640,
        'efficientnet-b5': 768,
        'efficientnet-b6': 896,
        'efficientnet-b7': 1024,
    }
    
    intermediate_dim = intermediate_dims.get(model_name, 512)
    
    model = EfficientNetLargeScale(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        intermediate_dim=intermediate_dim
    )
    
    # Charger un checkpoint si fourni
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Checkpoint chargé depuis {checkpoint_path}")
    
    return model


def get_model_info(model: EfficientNetLargeScale) -> dict:
    """
    Retourne des informations sur le modèle.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compter les paramètres par section
    backbone_params = sum(p.numel() for p in model.base_model.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    info = {
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'backbone_params': backbone_params,
        'classifier_params': classifier_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'intermediate_dim': model.intermediate_dim
    }
    
    return info


if __name__ == "__main__":
    # Test du modèle
    print("Test du modèle EfficientNet Large Scale")
    
    # Créer un modèle pour 1500 classes
    model = create_large_scale_model(
        num_classes=1500,
        model_name='efficientnet-b3',
        pretrained=True
    )
    
    # Afficher les infos
    info = get_model_info(model)
    print("\nInformations du modèle:")
    for key, value in info.items():
        if 'params' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 128, 128)
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test extraction de features
    features = model.get_features(dummy_input)
    print(f"\nTest extraction de features:")
    print(f"  Features shape: {features.shape}")