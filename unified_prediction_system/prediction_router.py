"""
Routeur de Prédiction pour NightScan

Ce module gère l'aiguillage automatique des fichiers vers les modèles appropriés
(audio ou photo) basé sur leur type détecté.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .file_type_detector import FileTypeDetector, FileType, NightScanFile
from .model_manager import UnifiedModelManager, ModelLoadError, PredictionError

logger = logging.getLogger(__name__)


class PredictionRouter:
    """Routeur principal pour l'aiguillage automatique des prédictions."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialise le routeur de prédiction.
        
        Args:
            config_path: Chemin vers le fichier de configuration des modèles
        """
        self.file_detector = FileTypeDetector()
        self.model_manager = UnifiedModelManager(config_path)
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "by_type": {
                "audio": 0,
                "photo": 0,
                "unknown": 0
            },
            "processing_times": []
        }
        
        logger.info("Routeur de prédiction initialisé")
    
    def predict_file(self, file_path: Union[str, Path], 
                    model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Effectue une prédiction automatique sur un fichier.
        
        Args:
            file_path: Chemin vers le fichier à analyser
            model_id: ID du modèle spécifique à utiliser (optionnel)
            
        Returns:
            Dict: Résultat de la prédiction avec métadonnées
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        try:
            # Détecter le type de fichier
            nightscan_file = self.file_detector.detect_file_type(file_path)
            
            if not nightscan_file.is_valid:
                return self._create_error_result(
                    file_path, 
                    f"Fichier invalide: {nightscan_file.error_message}",
                    start_time
                )
            
            # Aiguiller vers le bon modèle
            result = self._route_to_model(nightscan_file, model_id)
            
            # Ajouter les métadonnées du fichier
            result["file_metadata"] = nightscan_file.metadata
            result["file_type"] = nightscan_file.file_type.value
            result["processing_time"] = time.time() - start_time
            
            # Mettre à jour les statistiques
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur prédiction fichier {file_path}: {e}")
            return self._create_error_result(file_path, str(e), start_time)
    
    def _route_to_model(self, nightscan_file: NightScanFile, 
                       model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Aiguille le fichier vers le modèle approprié.
        
        Args:
            nightscan_file: Objet fichier NightScan analysé
            model_id: ID du modèle spécifique (optionnel)
            
        Returns:
            Dict: Résultat de la prédiction
        """
        if nightscan_file.file_type == FileType.AUDIO_SPECTROGRAM:
            return self._predict_audio_spectrogram(nightscan_file, model_id)
        
        elif nightscan_file.file_type == FileType.AUDIO_RAW:
            return self._predict_audio_raw(nightscan_file, model_id)
        
        elif nightscan_file.file_type == FileType.IMAGE:
            return self._predict_image(nightscan_file, model_id)
        
        else:
            raise PredictionError(f"Type de fichier non supporté: {nightscan_file.file_type}")
    
    def _predict_audio_spectrogram(self, nightscan_file: NightScanFile, 
                                  model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Prédit un spectrogramme audio.
        
        Args:
            nightscan_file: Fichier spectrogramme
            model_id: ID du modèle audio
            
        Returns:
            Dict: Résultat de la prédiction audio
        """
        try:
            # Charger le spectrogramme
            spectrogram = np.load(nightscan_file.file_path)
            
            # Utiliser le modèle audio
            audio_model_id = model_id or "default_audio"
            result = self.model_manager.predict_audio(spectrogram, audio_model_id)
            
            # Ajouter des informations spécifiques
            result["input_type"] = "spectrogram"
            result["spectrogram_shape"] = spectrogram.shape
            
            return result
            
        except Exception as e:
            raise PredictionError(f"Erreur prédiction spectrogramme: {e}")
    
    def _predict_audio_raw(self, nightscan_file: NightScanFile, 
                          model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Prédit un fichier audio brut (WAV).
        
        Args:
            nightscan_file: Fichier audio WAV
            model_id: ID du modèle audio
            
        Returns:
            Dict: Résultat de la prédiction audio
        """
        try:
            # Convertir WAV en spectrogramme
            spectrogram = self._wav_to_spectrogram(nightscan_file.file_path)
            
            # Utiliser le modèle audio
            audio_model_id = model_id or "default_audio"
            result = self.model_manager.predict_audio(spectrogram, audio_model_id)
            
            # Ajouter des informations spécifiques
            result["input_type"] = "wav_converted"
            result["spectrogram_shape"] = spectrogram.shape
            result["audio_metadata"] = {
                "channels": nightscan_file.metadata.get("audio_channels"),
                "sample_rate": nightscan_file.metadata.get("audio_frame_rate"),
                "duration": nightscan_file.metadata.get("audio_duration")
            }
            
            return result
            
        except Exception as e:
            raise PredictionError(f"Erreur prédiction audio WAV: {e}")
    
    def _predict_image(self, nightscan_file: NightScanFile, 
                      model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Prédit une image.
        
        Args:
            nightscan_file: Fichier image
            model_id: ID du modèle photo
            
        Returns:
            Dict: Résultat de la prédiction photo
        """
        try:
            # Utiliser le modèle photo
            photo_model_id = model_id or "default_photo"
            result = self.model_manager.predict_photo(nightscan_file.file_path, photo_model_id)
            
            # Ajouter des informations spécifiques
            result["input_type"] = "image"
            result["image_metadata"] = {
                "width": nightscan_file.metadata.get("image_width"),
                "height": nightscan_file.metadata.get("image_height"),
                "format": nightscan_file.metadata.get("image_format")
            }
            
            return result
            
        except Exception as e:
            raise PredictionError(f"Erreur prédiction image: {e}")
    
    def _wav_to_spectrogram(self, wav_path: Path) -> np.ndarray:
        """
        Convertit un fichier WAV en spectrogramme.
        
        Args:
            wav_path: Chemin vers le fichier WAV
            
        Returns:
            np.ndarray: Spectrogramme mel
        """
        try:
            import librosa
            
            # Charger le fichier audio
            y, sr = librosa.load(wav_path, sr=16000)  # 16kHz pour cohérence
            
            # Créer spectrogramme mel
            mel_spectrogram = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_mels=128, 
                fmax=8000
            )
            
            # Convertir en dB
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            return mel_spectrogram_db
            
        except ImportError:
            raise PredictionError("librosa requis pour la conversion WAV → spectrogramme")
        except Exception as e:
            raise PredictionError(f"Erreur conversion WAV: {e}")
    
    def _create_error_result(self, file_path: Path, error_message: str, 
                           start_time: float) -> Dict[str, Any]:
        """
        Crée un résultat d'erreur standardisé.
        
        Args:
            file_path: Chemin du fichier
            error_message: Message d'erreur
            start_time: Temps de début du traitement
            
        Returns:
            Dict: Résultat d'erreur
        """
        self.stats["failed_predictions"] += 1
        self.stats["by_type"]["unknown"] += 1
        
        return {
            "success": False,
            "error": error_message,
            "file_path": str(file_path),
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_stats(self, result: Dict[str, Any]):
        """
        Met à jour les statistiques de prédiction.
        
        Args:
            result: Résultat de prédiction
        """
        self.stats["total_predictions"] += 1
        
        if result.get("success", True):
            self.stats["successful_predictions"] += 1
            
            model_type = result.get("model_type", "unknown")
            if model_type in self.stats["by_type"]:
                self.stats["by_type"][model_type] += 1
            else:
                self.stats["by_type"]["unknown"] += 1
        else:
            self.stats["failed_predictions"] += 1
            self.stats["by_type"]["unknown"] += 1
        
        # Enregistrer le temps de traitement
        processing_time = result.get("processing_time", 0)
        self.stats["processing_times"].append(processing_time)
        
        # Garder seulement les 1000 derniers temps
        if len(self.stats["processing_times"]) > 1000:
            self.stats["processing_times"] = self.stats["processing_times"][-1000:]
    
    def batch_predict(self, file_paths: List[Union[str, Path]], 
                     model_ids: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Effectue des prédictions en lot sur plusieurs fichiers.
        
        Args:
            file_paths: Liste des chemins de fichiers
            model_ids: Dict optionnel {file_path: model_id}
            
        Returns:
            Dict: {file_path: result} pour chaque fichier
        """
        results = {}
        model_ids = model_ids or {}
        
        logger.info(f"Prédiction en lot de {len(file_paths)} fichiers")
        
        for file_path in file_paths:
            file_path_str = str(file_path)
            model_id = model_ids.get(file_path_str)
            
            try:
                result = self.predict_file(file_path, model_id)
                results[file_path_str] = result
                
                if result.get("success", True):
                    logger.debug(f"Prédiction réussie: {file_path_str}")
                else:
                    logger.warning(f"Prédiction échouée: {file_path_str} - {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"Erreur prédiction {file_path_str}: {e}")
                results[file_path_str] = self._create_error_result(
                    Path(file_path), str(e), time.time()
                )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du routeur.
        
        Returns:
            Dict: Statistiques complètes
        """
        stats = self.stats.copy()
        
        # Calculer les statistiques de temps
        if stats["processing_times"]:
            import numpy as np
            times = np.array(stats["processing_times"])
            stats["avg_processing_time"] = float(np.mean(times))
            stats["median_processing_time"] = float(np.median(times))
            stats["max_processing_time"] = float(np.max(times))
            stats["min_processing_time"] = float(np.min(times))
        
        # Ajouter les statistiques des modèles
        stats["model_stats"] = self.model_manager.get_model_stats()
        
        return stats
    
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "by_type": {
                "audio": 0,
                "photo": 0,
                "unknown": 0
            },
            "processing_times": []
        }
        logger.info("Statistiques remises à zéro")
    
    def cleanup(self):
        """Nettoie les ressources du routeur."""
        try:
            # Nettoyer les modèles non utilisés
            self.model_manager.cleanup_unused_models()
            logger.info("Nettoyage du routeur terminé")
        except Exception as e:
            logger.error(f"Erreur nettoyage routeur: {e}")
    
    def preload_models(self, audio_model: bool = True, photo_model: bool = True) -> bool:
        """
        Précharge les modèles pour des prédictions plus rapides.
        
        Args:
            audio_model: Charger le modèle audio
            photo_model: Charger le modèle photo
            
        Returns:
            bool: True si tous les modèles demandés sont chargés
        """
        success = True
        
        if audio_model:
            if not self.model_manager.load_audio_model():
                logger.error("Échec du préchargement du modèle audio")
                success = False
            else:
                logger.info("Modèle audio préchargé")
        
        if photo_model:
            if not self.model_manager.load_photo_model():
                logger.error("Échec du préchargement du modèle photo")
                success = False
            else:
                logger.info("Modèle photo préchargé")
        
        return success


# Instance globale du routeur
_prediction_router = None


def get_prediction_router(config_path: Optional[Path] = None) -> PredictionRouter:
    """
    Retourne l'instance globale du routeur de prédiction.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        
    Returns:
        PredictionRouter: Instance du routeur
    """
    global _prediction_router
    if _prediction_router is None:
        _prediction_router = PredictionRouter(config_path)
    return _prediction_router


def predict_file(file_path: Union[str, Path], 
                model_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Fonction de commodité pour prédire un fichier.
    
    Args:
        file_path: Chemin vers le fichier
        model_id: ID du modèle spécifique (optionnel)
        
    Returns:
        Dict: Résultat de la prédiction
    """
    router = get_prediction_router()
    return router.predict_file(file_path, model_id)


def main():
    """Fonction de test et démonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test du routeur de prédiction NightScan")
    parser.add_argument("file_path", help="Chemin vers le fichier à analyser")
    parser.add_argument("--model-id", help="ID du modèle spécifique à utiliser")
    parser.add_argument("--stats", action="store_true", help="Afficher les statistiques")
    parser.add_argument("--preload", action="store_true", help="Précharger les modèles")
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Test du Routeur de Prédiction NightScan ===\n")
    
    # Créer le routeur
    router = get_prediction_router()
    
    # Précharger les modèles si demandé
    if args.preload:
        print("Préchargement des modèles...")
        router.preload_models()
        print()
    
    # Effectuer la prédiction
    print(f"Analyse du fichier: {args.file_path}")
    result = router.predict_file(args.file_path, args.model_id)
    
    # Afficher les résultats
    print("\n=== Résultat de la Prédiction ===")
    
    if result.get("success", True):
        print(f"Type de fichier: {result.get('file_type', 'inconnu')}")
        print(f"Modèle utilisé: {result.get('model_type', 'inconnu')}")
        print(f"Prédiction: {result.get('predicted_class', 'inconnu')}")
        print(f"Confiance: {result.get('confidence', 0):.2%}")
        print(f"Temps de traitement: {result.get('processing_time', 0):.3f}s")
        
        if "top_predictions" in result:
            print("\nTop 3 prédictions:")
            for i, pred in enumerate(result["top_predictions"][:3], 1):
                print(f"  {i}. {pred['class']} ({pred['confidence']:.2%})")
    else:
        print(f"Erreur: {result.get('error', 'erreur inconnue')}")
    
    # Afficher les statistiques si demandé
    if args.stats:
        print("\n=== Statistiques du Routeur ===")
        stats = router.get_stats()
        print(f"Prédictions totales: {stats['total_predictions']}")
        print(f"Succès: {stats['successful_predictions']}")
        print(f"Échecs: {stats['failed_predictions']}")
        print(f"Par type: {stats['by_type']}")
        
        if "avg_processing_time" in stats:
            print(f"Temps moyen: {stats['avg_processing_time']:.3f}s")


if __name__ == "__main__":
    main()