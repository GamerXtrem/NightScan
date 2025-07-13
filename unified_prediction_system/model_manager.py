"""
Gestionnaire de Modèles Unifié pour NightScan

Ce module gère le chargement, la mise en cache et l'utilisation des modèles
audio et photo pour les prédictions.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
import json
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Add audio training path
sys.path.append(str(Path(__file__).parent.parent / "audio_training_efficientnet"))
sys.path.append(str(Path(__file__).parent.parent / "picture_training_enhanced"))

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Exception levée quand le chargement d'un modèle échoue."""
    pass


class PredictionError(Exception):
    """Exception levée quand une prédiction échoue."""
    pass


class ModelInfo:
    """Informations sur un modèle chargé."""
    
    def __init__(self, model_type: str, model_path: Path, config: Dict):
        self.model_type = model_type
        self.model_path = model_path
        self.config = config
        self.loaded_at = datetime.now()
        self.last_used = datetime.now()
        self.predictions_count = 0
        self.model = None
        self.device = None
        self.class_names = config.get('class_names', [])
    
    def update_usage(self):
        """Met à jour les statistiques d'utilisation."""
        self.last_used = datetime.now()
        self.predictions_count += 1
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du modèle."""
        return {
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "loaded_at": self.loaded_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "predictions_count": self.predictions_count,
            "device": str(self.device),
            "class_names": self.class_names
        }


class AudioModelLoader:
    """Chargeur spécialisé pour les modèles audio."""
    
    @staticmethod
    def load_model(model_path: Path, config: Dict) -> torch.nn.Module:
        """Charge un modèle audio (ResNet18 ou EfficientNet)."""
        try:
            import torchvision.models as models
            
            model_name = config.get('model_name', 'resnet18')
            num_classes = config.get('num_classes', 6)
            
            # Créer le modèle selon le type
            if model_name == 'resnet18':
                model = models.resnet18(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            elif 'efficientnet' in model_name:
                # Fallback vers EfficientNet si disponible
                try:
                    from models.efficientnet_config import EfficientNetConfig, create_model
                    model_config = EfficientNetConfig(**config)
                    model = create_model(model_config)
                except ImportError:
                    logger.warning("EfficientNet non disponible, utilisation ResNet18")
                    model = models.resnet18(pretrained=False)
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            else:
                raise ModelLoadError(f"Modèle non supporté: {model_name}")
            
            # Charger les poids
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                # Adapter les poids si nécessaire
                model = AudioModelLoader._adapt_weights(model, state_dict, num_classes)
                logger.info(f"Modèle audio chargé: {model_path}")
            else:
                logger.warning(f"Modèle audio introuvable: {model_path}")
                raise ModelLoadError(f"Fichier modèle audio non trouvé: {model_path}")
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle audio: {e}")
            raise ModelLoadError(f"Impossible de charger le modèle audio: {e}")
    
    @staticmethod
    def _adapt_weights(model, state_dict, num_classes):
        """Adapte les poids chargés au modèle."""
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Chargement strict échoué: {e}")
            # Charger partiellement en ignorant les couches incompatibles
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Chargement partiel: {len(filtered_dict)}/{len(state_dict)} couches")
        
        return model
    
    @staticmethod
    def preprocess_spectrogram(spectrogram: np.ndarray) -> torch.Tensor:
        """Prétraite un spectrogramme pour l'inférence."""
        try:
            # Normaliser le spectrogramme
            spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
            
            # Convertir en tensor
            if len(spectrogram.shape) == 2:
                # Ajouter la dimension channel (spectrogramme → RGB)
                spectrogram = np.stack([spectrogram, spectrogram, spectrogram], axis=0)
            
            tensor = torch.FloatTensor(spectrogram)
            
            # Ajouter la dimension batch
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            raise PredictionError(f"Erreur prétraitement spectrogramme: {e}")


class PhotoModelLoader:
    """Chargeur spécialisé pour les modèles photo."""
    
    @staticmethod
    def load_model(model_path: Path, config: Dict) -> torch.nn.Module:
        """Charge un modèle photo (ResNet18, EfficientNet, etc.)."""
        try:
            import torchvision.models as models
            
            model_name = config.get('model_name', 'resnet18')
            num_classes = config.get('num_classes', 8)
            
            # Créer le modèle selon le type
            if model_name == 'resnet18':
                model = models.resnet18(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            elif 'efficientnet' in model_name:
                # Fallback vers EfficientNet si disponible
                try:
                    from models.photo_config import PhotoConfig, create_model
                    model_config = PhotoConfig(**config)
                    model = create_model(model_config)
                except ImportError:
                    logger.warning("Configuration photo non disponible, utilisation ResNet18")
                    model = models.resnet18(pretrained=False)
                    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            else:
                raise ModelLoadError(f"Modèle non supporté: {model_name}")
            
            # Charger les poids
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                # Adapter les poids si nécessaire
                model = PhotoModelLoader._adapt_weights(model, state_dict, num_classes)
                logger.info(f"Modèle photo chargé: {model_path}")
            else:
                logger.warning(f"Modèle photo introuvable: {model_path}")
                raise ModelLoadError(f"Fichier modèle photo non trouvé: {model_path}")
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle photo: {e}")
            raise ModelLoadError(f"Impossible de charger le modèle photo: {e}")
    
    @staticmethod
    def _adapt_weights(model, state_dict, num_classes):
        """Adapte les poids chargés au modèle."""
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Chargement strict échoué: {e}")
            # Charger partiellement en ignorant les couches incompatibles
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Chargement partiel: {len(filtered_dict)}/{len(state_dict)} couches")
        
        return model
    
    @staticmethod
    def preprocess_image(image_path: Path, input_size: tuple = (224, 224)) -> torch.Tensor:
        """Prétraite une image pour l'inférence."""
        try:
            # Charger l'image
            image = Image.open(image_path).convert('RGB')
            
            # Transformations standard
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Appliquer les transformations
            tensor = transform(image)
            
            # Ajouter la dimension batch
            tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            raise PredictionError(f"Erreur prétraitement image: {e}")


class UnifiedModelManager:
    """Gestionnaire unifié pour les modèles audio et photo."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialise le gestionnaire de modèles.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.models: Dict[str, ModelInfo] = {}
        self.device = self._get_device()
        self.config = self._load_config(config_path)
        self._lock = threading.Lock()
        
        logger.info(f"Gestionnaire de modèles initialisé sur {self.device}")
    
    def _get_device(self) -> torch.device:
        """Détermine le device optimal pour l'inférence."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Charge la configuration des modèles depuis le registre central."""
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Tenter de charger depuis le registre de modèles
        try:
            registry_path = Path("model_registry.json")
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                # Extraire la configuration pour les modèles heavy (VPS)
                config = {}
                for model_id, model_data in registry_data.get('models', {}).items():
                    if model_data.get('variant') == 'heavy':
                        model_type = model_data.get('model_type')
                        config[f"{model_type}_model"] = {
                            "model_path": model_data.get('file_path'),
                            "config": {
                                "model_name": "resnet18",  # Basé sur ResNet18
                                "num_classes": model_data.get('num_classes'),
                                "pretrained": False,
                                "dropout_rate": 0.3
                            },
                            "class_names": model_data.get('class_names', [])
                        }
                
                if config:
                    logger.info("Configuration chargée depuis le registre de modèles")
                    return config
        except Exception as e:
            logger.warning(f"Impossible de charger le registre: {e}")
        
        # Configuration par défaut (fallback)
        return {
            "audio_model": {
                "model_path": "models/resnet18/best_model.pth",
                "config": {
                    "model_name": "resnet18",
                    "num_classes": 6,
                    "pretrained": False,
                    "dropout_rate": 0.3
                },
                "class_names": ["bird_song", "mammal_call", "insect_sound", 
                              "amphibian_call", "environmental_sound", "unknown_species"]
            },
            "photo_model": {
                "model_path": "models/resnet18/best_model.pth",
                "config": {
                    "model_name": "resnet18",
                    "num_classes": 8,
                    "pretrained": False,
                    "dropout_rate": 0.3
                },
                "class_names": ["bat", "owl", "raccoon", "opossum", "deer", "fox", "coyote", "unknown"]
            }
        }
    
    def load_audio_model(self, model_id: str = "default_audio") -> bool:
        """
        Charge le modèle audio.
        
        Args:
            model_id: Identifiant du modèle
            
        Returns:
            bool: True si le chargement réussit
        """
        try:
            with self._lock:
                if model_id in self.models:
                    logger.info(f"Modèle audio {model_id} déjà chargé")
                    return True
                
                # Configuration du modèle audio
                audio_config = self.config["audio_model"]
                model_path = Path(audio_config["model_path"])
                
                # Charger le modèle
                model = AudioModelLoader.load_model(model_path, audio_config["config"])
                model.to(self.device)
                
                # Créer l'objet ModelInfo
                model_info = ModelInfo("audio", model_path, audio_config)
                model_info.model = model
                model_info.device = self.device
                
                self.models[model_id] = model_info
                logger.info(f"Modèle audio {model_id} chargé avec succès")
                return True
                
        except Exception as e:
            logger.error(f"Erreur chargement modèle audio {model_id}: {e}")
            return False
    
    def load_photo_model(self, model_id: str = "default_photo") -> bool:
        """
        Charge le modèle photo.
        
        Args:
            model_id: Identifiant du modèle
            
        Returns:
            bool: True si le chargement réussit
        """
        try:
            with self._lock:
                if model_id in self.models:
                    logger.info(f"Modèle photo {model_id} déjà chargé")
                    return True
                
                # Configuration du modèle photo
                photo_config = self.config["photo_model"]
                model_path = Path(photo_config["model_path"])
                
                # Charger le modèle
                model = PhotoModelLoader.load_model(model_path, photo_config["config"])
                model.to(self.device)
                
                # Créer l'objet ModelInfo
                model_info = ModelInfo("photo", model_path, photo_config)
                model_info.model = model
                model_info.device = self.device
                
                self.models[model_id] = model_info
                logger.info(f"Modèle photo {model_id} chargé avec succès")
                return True
                
        except Exception as e:
            logger.error(f"Erreur chargement modèle photo {model_id}: {e}")
            return False
    
    def predict_audio(self, spectrogram: np.ndarray, model_id: str = "default_audio") -> Dict:
        """
        Effectue une prédiction audio.
        
        Args:
            spectrogram: Spectrogramme numpy
            model_id: Identifiant du modèle
            
        Returns:
            Dict: Résultat de la prédiction
        """
        try:
            # Charger le modèle si nécessaire
            if model_id not in self.models:
                if not self.load_audio_model(model_id):
                    raise PredictionError(f"Impossible de charger le modèle audio {model_id}")
            
            model_info = self.models[model_id]
            model_info.update_usage()
            
            # Prétraiter le spectrogramme
            input_tensor = AudioModelLoader.preprocess_spectrogram(spectrogram)
            input_tensor = input_tensor.to(self.device)
            
            # Prédiction
            with torch.no_grad():
                outputs = model_info.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
            
            # Formater les résultats
            probs = probabilities.cpu().numpy()[0]
            pred_class = predictions.cpu().numpy()[0]
            
            # Top-3 prédictions
            top_indices = np.argsort(probs)[::-1][:3]
            top_predictions = []
            
            for idx in top_indices:
                top_predictions.append({
                    "class": model_info.class_names[idx],
                    "confidence": float(probs[idx]),
                    "class_id": int(idx)
                })
            
            return {
                "model_type": "audio",
                "model_id": model_id,
                "predicted_class": model_info.class_names[pred_class],
                "predicted_class_id": int(pred_class),
                "confidence": float(probs[pred_class]),
                "top_predictions": top_predictions,
                "processing_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Erreur prédiction audio: {e}")
            raise PredictionError(f"Prédiction audio échouée: {e}")
    
    def predict_photo(self, image_path: Path, model_id: str = "default_photo") -> Dict:
        """
        Effectue une prédiction photo.
        
        Args:
            image_path: Chemin vers l'image
            model_id: Identifiant du modèle
            
        Returns:
            Dict: Résultat de la prédiction
        """
        try:
            # Charger le modèle si nécessaire
            if model_id not in self.models:
                if not self.load_photo_model(model_id):
                    raise PredictionError(f"Impossible de charger le modèle photo {model_id}")
            
            model_info = self.models[model_id]
            model_info.update_usage()
            
            # Prétraiter l'image
            input_tensor = PhotoModelLoader.preprocess_image(image_path)
            input_tensor = input_tensor.to(self.device)
            
            # Prédiction
            with torch.no_grad():
                outputs = model_info.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
            
            # Formater les résultats
            probs = probabilities.cpu().numpy()[0]
            pred_class = predictions.cpu().numpy()[0]
            
            # Top-3 prédictions
            top_indices = np.argsort(probs)[::-1][:3]
            top_predictions = []
            
            for idx in top_indices:
                top_predictions.append({
                    "class": model_info.class_names[idx],
                    "confidence": float(probs[idx]),
                    "class_id": int(idx)
                })
            
            return {
                "model_type": "photo",
                "model_id": model_id,
                "predicted_class": model_info.class_names[pred_class],
                "predicted_class_id": int(pred_class),
                "confidence": float(probs[pred_class]),
                "top_predictions": top_predictions,
                "processing_time": time.time()
            }
            
        except Exception as e:
            logger.error(f"Erreur prédiction photo: {e}")
            raise PredictionError(f"Prédiction photo échouée: {e}")
    
    def unload_model(self, model_id: str) -> bool:
        """
        Décharge un modèle de la mémoire.
        
        Args:
            model_id: Identifiant du modèle
            
        Returns:
            bool: True si le déchargement réussit
        """
        try:
            with self._lock:
                if model_id in self.models:
                    del self.models[model_id]
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    logger.info(f"Modèle {model_id} déchargé")
                    return True
                return False
        except Exception as e:
            logger.error(f"Erreur déchargement modèle {model_id}: {e}")
            return False
    
    def get_model_stats(self) -> Dict:
        """Retourne les statistiques de tous les modèles."""
        return {
            "device": str(self.device),
            "loaded_models": len(self.models),
            "models": {model_id: model_info.get_stats() 
                      for model_id, model_info in self.models.items()}
        }
    
    def cleanup_unused_models(self, max_idle_time: int = 3600):
        """
        Nettoie les modèles non utilisés depuis un certain temps.
        
        Args:
            max_idle_time: Temps d'inactivité max en secondes
        """
        try:
            with self._lock:
                current_time = datetime.now()
                to_remove = []
                
                for model_id, model_info in self.models.items():
                    idle_time = (current_time - model_info.last_used).total_seconds()
                    if idle_time > max_idle_time:
                        to_remove.append(model_id)
                
                for model_id in to_remove:
                    self.unload_model(model_id)
                    logger.info(f"Modèle {model_id} nettoyé (inactif depuis {idle_time}s)")
                
        except Exception as e:
            logger.error(f"Erreur nettoyage modèles: {e}")


# Instance globale du gestionnaire
_model_manager = None


def get_model_manager() -> UnifiedModelManager:
    """Retourne l'instance globale du gestionnaire de modèles."""
    global _model_manager
    if _model_manager is None:
        _model_manager = UnifiedModelManager()
    return _model_manager


def main():
    """Fonction de test."""
    print("=== Test du Gestionnaire de Modèles Unifié ===\n")
    
    manager = get_model_manager()
    
    # Afficher les stats
    stats = manager.get_model_stats()
    print(f"Device: {stats['device']}")
    print(f"Modèles chargés: {stats['loaded_models']}")
    
    # Tester le chargement des modèles
    print("\nChargement des modèles...")
    audio_loaded = manager.load_audio_model()
    photo_loaded = manager.load_photo_model()
    
    print(f"Modèle audio chargé: {audio_loaded}")
    print(f"Modèle photo chargé: {photo_loaded}")
    
    # Afficher les stats finales
    final_stats = manager.get_model_stats()
    print(f"\nModèles finaux chargés: {final_stats['loaded_models']}")
    for model_id, model_stats in final_stats['models'].items():
        print(f"  {model_id}: {model_stats['model_type']}")


if __name__ == "__main__":
    main()