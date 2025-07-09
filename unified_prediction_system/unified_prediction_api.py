"""
API Unifiée de Prédiction pour NightScan

Cette API fournit un point d'entrée unique pour les prédictions
audio et photo avec aiguillage automatique.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import tempfile
import uuid

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .prediction_router import get_prediction_router, PredictionRouter
from .file_type_detector import FileType

# Configuration
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "nightscan_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.wav', '.npy', '.jpg', '.jpeg'}

logger = logging.getLogger(__name__)


class UnifiedPredictionAPI:
    """API Flask pour les prédictions unifiées NightScan."""
    
    def __init__(self, config_path: Optional[Path] = None, debug: bool = False):
        """
        Initialise l'API de prédiction.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            debug: Mode debug Flask
        """
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
        self.app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        
        # Configuration CORS
        CORS(self.app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])
        
        # Initialiser le routeur de prédiction
        self.router = get_prediction_router(config_path)
        
        # Précharger les modèles
        self.router.preload_models()
        
        # Configuration des routes
        self._setup_routes()
        
        # Configuration du logging
        if not debug:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        logger.info("API de prédiction unifiée initialisée")
    
    def _setup_routes(self):
        """Configure les routes Flask."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Point de santé de l'API."""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            })
        
        @self.app.route('/models/status', methods=['GET'])
        def models_status():
            """Statut des modèles chargés."""
            try:
                stats = self.router.get_stats()
                return jsonify({
                    "success": True,
                    "models": stats.get("model_stats", {}),
                    "router_stats": {
                        "total_predictions": stats["total_predictions"],
                        "successful_predictions": stats["successful_predictions"],
                        "failed_predictions": stats["failed_predictions"]
                    }
                })
            except Exception as e:
                logger.error(f"Erreur statut modèles: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/predict/upload', methods=['POST'])
        def predict_upload():
            """Prédiction par upload de fichier."""
            try:
                # Vérifier la présence du fichier
                if 'file' not in request.files:
                    return jsonify({
                        "success": False,
                        "error": "Aucun fichier fourni"
                    }), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({
                        "success": False,
                        "error": "Nom de fichier vide"
                    }), 400
                
                # Vérifier l'extension
                if not self._allowed_file(file.filename):
                    return jsonify({
                        "success": False,
                        "error": f"Extension non autorisée. Autorisées: {', '.join(ALLOWED_EXTENSIONS)}"
                    }), 400
                
                # Sauvegarder le fichier temporairement
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = UPLOAD_FOLDER / unique_filename
                
                file.save(filepath)
                
                try:
                    # Paramètres optionnels
                    model_id = request.form.get('model_id')
                    
                    # Effectuer la prédiction
                    result = self.router.predict_file(filepath, model_id)
                    
                    # Nettoyer le fichier temporaire
                    if filepath.exists():
                        filepath.unlink()
                    
                    return jsonify({
                        "success": True,
                        "prediction": result,
                        "upload_filename": filename
                    })
                    
                except Exception as e:
                    # Nettoyer le fichier en cas d'erreur
                    if filepath.exists():
                        filepath.unlink()
                    raise e
                    
            except Exception as e:
                logger.error(f"Erreur prédiction upload: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/predict/file', methods=['POST'])
        def predict_file():
            """Prédiction par chemin de fichier."""
            try:
                data = request.get_json()
                
                if not data or 'file_path' not in data:
                    return jsonify({
                        "success": False,
                        "error": "Chemin de fichier requis"
                    }), 400
                
                file_path = data['file_path']
                model_id = data.get('model_id')
                
                # Vérifier l'existence du fichier
                if not Path(file_path).exists():
                    return jsonify({
                        "success": False,
                        "error": f"Fichier non trouvé: {file_path}"
                    }), 404
                
                # Effectuer la prédiction
                result = self.router.predict_file(file_path, model_id)
                
                return jsonify({
                    "success": True,
                    "prediction": result
                })
                
            except Exception as e:
                logger.error(f"Erreur prédiction fichier: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/predict/batch', methods=['POST'])
        def predict_batch():
            """Prédiction en lot."""
            try:
                data = request.get_json()
                
                if not data or 'file_paths' not in data:
                    return jsonify({
                        "success": False,
                        "error": "Liste de fichiers requise"
                    }), 400
                
                file_paths = data['file_paths']
                model_ids = data.get('model_ids', {})
                
                # Vérifier la limite
                if len(file_paths) > 100:
                    return jsonify({
                        "success": False,
                        "error": "Maximum 100 fichiers par lot"
                    }), 400
                
                # Effectuer les prédictions
                results = self.router.batch_predict(file_paths, model_ids)
                
                return jsonify({
                    "success": True,
                    "results": results,
                    "total_files": len(file_paths)
                })
                
            except Exception as e:
                logger.error(f"Erreur prédiction lot: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/models/preload', methods=['POST'])
        def preload_models():
            """Précharge les modèles."""
            try:
                data = request.get_json() or {}
                
                audio_model = data.get('audio_model', True)
                photo_model = data.get('photo_model', True)
                
                success = self.router.preload_models(audio_model, photo_model)
                
                return jsonify({
                    "success": success,
                    "message": "Modèles préchargés" if success else "Erreur préchargement"
                })
                
            except Exception as e:
                logger.error(f"Erreur préchargement modèles: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Statistiques complètes du système."""
            try:
                stats = self.router.get_stats()
                return jsonify({
                    "success": True,
                    "stats": stats
                })
            except Exception as e:
                logger.error(f"Erreur statistiques: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/stats/reset', methods=['POST'])
        def reset_stats():
            """Remet à zéro les statistiques."""
            try:
                self.router.reset_stats()
                return jsonify({
                    "success": True,
                    "message": "Statistiques remises à zéro"
                })
            except Exception as e:
                logger.error(f"Erreur reset statistiques: {e}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        @self.app.route('/supported-formats', methods=['GET'])
        def supported_formats():
            """Formats de fichiers supportés."""
            return jsonify({
                "success": True,
                "formats": {
                    "audio_raw": [".wav"],
                    "audio_spectrogram": [".npy"],
                    "image": [".jpg", ".jpeg"],
                    "all_supported": list(ALLOWED_EXTENSIONS)
                },
                "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024)
            })
        
        @self.app.errorhandler(413)
        def too_large(e):
            """Gestionnaire d'erreur pour fichiers trop volumineux."""
            return jsonify({
                "success": False,
                "error": f"Fichier trop volumineux. Maximum: {MAX_FILE_SIZE / (1024 * 1024)}MB"
            }), 413
        
        @self.app.errorhandler(400)
        def bad_request(e):
            """Gestionnaire d'erreur pour requêtes malformées."""
            return jsonify({
                "success": False,
                "error": "Requête malformée"
            }), 400
        
        @self.app.errorhandler(500)
        def internal_error(e):
            """Gestionnaire d'erreur interne."""
            logger.error(f"Erreur interne: {e}")
            return jsonify({
                "success": False,
                "error": "Erreur interne du serveur"
            }), 500
    
    def _allowed_file(self, filename: str) -> bool:
        """Vérifie si l'extension du fichier est autorisée."""
        return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """
        Lance le serveur Flask.
        
        Args:
            host: Adresse d'écoute
            port: Port d'écoute
            debug: Mode debug
        """
        logger.info(f"Démarrage de l'API sur {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)
    
    def cleanup(self):
        """Nettoie les ressources."""
        try:
            self.router.cleanup()
            
            # Nettoyer les fichiers temporaires
            for temp_file in UPLOAD_FOLDER.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    
            logger.info("Nettoyage API terminé")
        except Exception as e:
            logger.error(f"Erreur nettoyage API: {e}")


def create_app(config_path: Optional[Path] = None, debug: bool = False) -> Flask:
    """
    Factory function pour créer l'application Flask.
    
    Args:
        config_path: Chemin vers le fichier de configuration
        debug: Mode debug
        
    Returns:
        Flask: Application Flask configurée
    """
    api = UnifiedPredictionAPI(config_path, debug)
    return api.app


def main():
    """Point d'entrée principal pour lancer l'API."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API de prédiction unifiée NightScan")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d'écoute")
    parser.add_argument("--port", type=int, default=5000, help="Port d'écoute")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    parser.add_argument("--config", type=Path, help="Fichier de configuration")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== API de Prédiction Unifiée NightScan ===")
    print(f"Démarrage sur {args.host}:{args.port}")
    print(f"Mode debug: {args.debug}")
    print(f"Dossier d'upload: {UPLOAD_FOLDER}")
    print(f"Taille max fichier: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB")
    print(f"Extensions autorisées: {', '.join(ALLOWED_EXTENSIONS)}")
    print()
    
    # Créer et lancer l'API
    api = UnifiedPredictionAPI(args.config, args.debug)
    
    try:
        api.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\nArrêt de l'API...")
        api.cleanup()
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        api.cleanup()
        raise


if __name__ == "__main__":
    main()