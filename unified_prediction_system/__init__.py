"""
Système de Prédiction Unifiée NightScan

Ce package fournit un système automatisé d'aiguillage des fichiers audio et photo
vers les modèles de prédiction appropriés basé sur le format des fichiers NightScan.

Composants principaux:
- FileTypeDetector: Détection automatique du type de fichier
- UnifiedModelManager: Gestion des modèles audio et photo
- PredictionRouter: Routeur d'aiguillage automatique
- UnifiedPredictionAPI: API Flask pour les prédictions

Usage rapide:
    from unified_prediction_system import predict_file
    result = predict_file("AUD_20240109_143045_4695_0745.wav")
    print(f"Prédiction: {result['predicted_class']}")
"""

__version__ = "1.0.0"
__author__ = "NightScan Team"

# Imports principaux pour faciliter l'utilisation
from .prediction_router import predict_file, get_prediction_router
from .file_type_detector import FileTypeDetector, FileType
from .model_manager import UnifiedModelManager, get_model_manager
from .unified_prediction_api import UnifiedPredictionAPI, create_app

__all__ = [
    'predict_file',
    'get_prediction_router',
    'FileTypeDetector',
    'FileType',
    'UnifiedModelManager',
    'get_model_manager',
    'UnifiedPredictionAPI',
    'create_app'
]