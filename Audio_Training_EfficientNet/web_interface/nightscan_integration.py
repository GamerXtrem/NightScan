"""
Intégration avec le Système NightScan Existant

Module d'intégration pour connecter le système d'entraînement EfficientNet
au système NightScan principal avec synchronisation des modèles et données.
"""

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import sqlite3
import requests

# Configuration logging
logger = logging.getLogger(__name__)

class NightScanIntegration:
    """Classe d'intégration avec le système NightScan principal."""
    
    def __init__(self, nightscan_root: str = None):
        """
        Initialise l'intégration NightScan.
        
        Args:
            nightscan_root: Chemin racine du système NightScan
        """
        self.nightscan_root = Path(nightscan_root) if nightscan_root else self._find_nightscan_root()
        self.config_path = self.nightscan_root / "config"
        self.models_path = self.nightscan_root / "models"
        self.data_path = self.nightscan_root / "data"
        self.logs_path = self.nightscan_root / "logs"
        
        # Chemin vers la base de données NightScan
        self.db_path = self.nightscan_root / "nightscan.db"
        
        # Configuration d'intégration
        self.integration_config = self._load_integration_config()
        
        logger.info(f"Intégration NightScan initialisée: {self.nightscan_root}")
    
    def _find_nightscan_root(self) -> Path:
        """Trouve automatiquement le répertoire racine NightScan."""
        current_dir = Path(__file__).parent
        
        # Chercher dans les répertoires parents
        for parent in current_dir.parents:
            if (parent / "NightScanPi").exists() or (parent / "nightscan.py").exists():
                return parent
        
        # Valeur par défaut
        return current_dir.parent.parent
    
    def _load_integration_config(self) -> Dict[str, Any]:
        """Charge la configuration d'intégration."""
        config_file = self.config_path / "integration_config.json"
        
        default_config = {
            "model_deployment": {
                "auto_deploy": True,
                "backup_previous": True,
                "validation_required": True
            },
            "data_sync": {
                "sync_training_data": True,
                "sync_results": True,
                "cleanup_old_data": False
            },
            "notification": {
                "training_complete": True,
                "deployment_success": True,
                "errors": True
            },
            "api": {
                "nightscan_endpoint": "http://localhost:8000",
                "auth_token": None
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Merge avec la config par défaut
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la config: {e}")
        
        return default_config
    
    def sync_training_data(self, csv_paths: List[str]) -> bool:
        """
        Synchronise les données d'entraînement avec le système NightScan.
        
        Args:
            csv_paths: Liste des chemins vers les fichiers CSV
            
        Returns:
            bool: True si la synchronisation a réussi
        """
        try:
            logger.info("Synchronisation des données d'entraînement...")
            
            # Créer le répertoire de destination
            training_data_dir = self.data_path / "training" / "efficientnet"
            training_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Copier les fichiers CSV
            for csv_path in csv_paths:
                if Path(csv_path).exists():
                    dest_path = training_data_dir / Path(csv_path).name
                    shutil.copy2(csv_path, dest_path)
                    logger.info(f"Copié: {csv_path} -> {dest_path}")
                else:
                    logger.warning(f"Fichier non trouvé: {csv_path}")
            
            # Mettre à jour la base de données
            self._update_training_data_db(csv_paths)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation des données: {e}")
            return False
    
    def deploy_trained_model(self, model_path: str, config: Dict[str, Any], 
                           metrics: Dict[str, Any]) -> bool:
        """
        Déploie un modèle entraîné dans le système NightScan.
        
        Args:
            model_path: Chemin vers le modèle entraîné
            config: Configuration du modèle
            metrics: Métriques de performance
            
        Returns:
            bool: True si le déploiement a réussi
        """
        try:
            logger.info(f"Déploiement du modèle: {model_path}")
            
            # Validation du modèle
            if not self._validate_model(model_path, metrics):
                logger.error("Validation du modèle échouée")
                return False
            
            # Sauvegarde du modèle précédent
            if self.integration_config["model_deployment"]["backup_previous"]:
                self._backup_current_model()
            
            # Copie du nouveau modèle
            model_dest = self.models_path / "efficientnet" / "current_model.pth"
            model_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, model_dest)
            
            # Sauvegarde de la configuration
            config_dest = self.models_path / "efficientnet" / "model_config.json"
            with open(config_dest, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Sauvegarde des métriques
            metrics_dest = self.models_path / "efficientnet" / "model_metrics.json"
            with open(metrics_dest, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Mise à jour de la base de données
            self._update_model_db(model_path, config, metrics)
            
            # Notification au système principal
            self._notify_nightscan_system("model_deployed", {
                "model_path": str(model_dest),
                "config": config,
                "metrics": metrics
            })
            
            logger.info("Modèle déployé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du déploiement: {e}")
            return False
    
    def get_nightscan_datasets(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des datasets disponibles dans NightScan.
        
        Returns:
            List[Dict]: Liste des datasets avec leurs métadonnées
        """
        try:
            datasets = []
            
            # Connexion à la base de données NightScan
            if self.db_path.exists():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Requête pour récupérer les datasets
                cursor.execute("""
                    SELECT name, path, created_at, size, description 
                    FROM datasets 
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                """)
                
                for row in cursor.fetchall():
                    datasets.append({
                        "name": row[0],
                        "path": row[1],
                        "created_at": row[2],
                        "size": row[3],
                        "description": row[4]
                    })
                
                conn.close()
            
            return datasets
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des datasets: {e}")
            return []
    
    def sync_training_results(self, session_id: str, results: Dict[str, Any]) -> bool:
        """
        Synchronise les résultats d'entraînement avec le système NightScan.
        
        Args:
            session_id: ID de la session d'entraînement
            results: Résultats détaillés de l'entraînement
            
        Returns:
            bool: True si la synchronisation a réussi
        """
        try:
            logger.info(f"Synchronisation des résultats pour la session: {session_id}")
            
            # Créer le répertoire de résultats
            results_dir = self.logs_path / "training" / "efficientnet" / session_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder les résultats
            results_file = results_dir / "training_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Sauvegarder l'historique des métriques
            if "history" in results:
                history_file = results_dir / "metrics_history.json"
                with open(history_file, 'w') as f:
                    json.dump(results["history"], f, indent=2)
            
            # Mise à jour de la base de données
            self._update_training_results_db(session_id, results)
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation des résultats: {e}")
            return False
    
    def get_model_performance_history(self) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des performances des modèles.
        
        Returns:
            List[Dict]: Historique des performances
        """
        try:
            history = []
            
            if self.db_path.exists():
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT session_id, model_type, accuracy, f1_score, 
                           training_time, deployed_at, metrics
                    FROM model_deployments 
                    WHERE model_type = 'efficientnet'
                    ORDER BY deployed_at DESC
                """)
                
                for row in cursor.fetchall():
                    history.append({
                        "session_id": row[0],
                        "model_type": row[1],
                        "accuracy": row[2],
                        "f1_score": row[3],
                        "training_time": row[4],
                        "deployed_at": row[5],
                        "metrics": json.loads(row[6]) if row[6] else {}
                    })
                
                conn.close()
            
            return history
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique: {e}")
            return []
    
    def _validate_model(self, model_path: str, metrics: Dict[str, Any]) -> bool:
        """Valide un modèle avant déploiement."""
        try:
            # Vérifier que le fichier existe
            if not Path(model_path).exists():
                return False
            
            # Vérifier les métriques minimales
            min_accuracy = self.integration_config.get("validation", {}).get("min_accuracy", 0.7)
            if metrics.get("accuracy", 0) < min_accuracy:
                logger.warning(f"Accuracy trop faible: {metrics.get('accuracy', 0)}")
                return False
            
            # Vérifier la taille du modèle
            model_size = Path(model_path).stat().st_size
            max_size = self.integration_config.get("validation", {}).get("max_model_size", 500 * 1024 * 1024)  # 500MB
            if model_size > max_size:
                logger.warning(f"Modèle trop volumineux: {model_size / (1024*1024):.1f}MB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {e}")
            return False
    
    def _backup_current_model(self):
        """Sauvegarde le modèle actuel."""
        try:
            current_model = self.models_path / "efficientnet" / "current_model.pth"
            if current_model.exists():
                backup_dir = self.models_path / "efficientnet" / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"model_backup_{timestamp}.pth"
                
                shutil.copy2(current_model, backup_path)
                logger.info(f"Modèle sauvegardé: {backup_path}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
    
    def _update_training_data_db(self, csv_paths: List[str]):
        """Met à jour la base de données avec les informations des données d'entraînement."""
        try:
            if not self.db_path.exists():
                self._create_database()
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for csv_path in csv_paths:
                if Path(csv_path).exists():
                    size = Path(csv_path).stat().st_size
                    cursor.execute("""
                        INSERT OR REPLACE INTO training_data 
                        (path, size, updated_at, status) 
                        VALUES (?, ?, ?, ?)
                    """, (csv_path, size, datetime.now().isoformat(), "active"))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la DB: {e}")
    
    def _update_model_db(self, model_path: str, config: Dict[str, Any], 
                        metrics: Dict[str, Any]):
        """Met à jour la base de données avec les informations du modèle."""
        try:
            if not self.db_path.exists():
                self._create_database()
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_deployments 
                (model_path, model_type, config, metrics, accuracy, f1_score, deployed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_path,
                "efficientnet",
                json.dumps(config),
                json.dumps(metrics),
                metrics.get("accuracy", 0),
                metrics.get("f1_score", 0),
                datetime.now().isoformat(),
                "active"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du modèle en DB: {e}")
    
    def _update_training_results_db(self, session_id: str, results: Dict[str, Any]):
        """Met à jour la base de données avec les résultats d'entraînement."""
        try:
            if not self.db_path.exists():
                self._create_database()
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO training_sessions 
                (session_id, model_type, results, final_accuracy, final_loss, 
                 training_time, completed_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                "efficientnet",
                json.dumps(results),
                results.get("final_accuracy", 0),
                results.get("final_loss", 0),
                results.get("training_time", 0),
                datetime.now().isoformat(),
                "completed"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des résultats: {e}")
    
    def _create_database(self):
        """Crée la base de données d'intégration."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Table pour les données d'entraînement
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    size INTEGER,
                    updated_at TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Table pour les déploiements de modèles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_deployments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_path TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    config TEXT,
                    metrics TEXT,
                    accuracy REAL,
                    f1_score REAL,
                    deployed_at TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            # Table pour les sessions d'entraînement
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    model_type TEXT NOT NULL,
                    results TEXT,
                    final_accuracy REAL,
                    final_loss REAL,
                    training_time REAL,
                    completed_at TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de la DB: {e}")
    
    def _notify_nightscan_system(self, event_type: str, data: Dict[str, Any]):
        """Envoie une notification au système NightScan principal."""
        try:
            if not self.integration_config["notification"].get(event_type.replace("_", ""), False):
                return
            
            api_endpoint = self.integration_config["api"]["nightscan_endpoint"]
            auth_token = self.integration_config["api"]["auth_token"]
            
            if not api_endpoint:
                logger.warning("Endpoint API NightScan non configuré")
                return
            
            headers = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
            
            payload = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            response = requests.post(
                f"{api_endpoint}/api/training/notifications",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Notification envoyée: {event_type}")
            else:
                logger.warning(f"Erreur notification: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de notification: {e}")


# Fonction d'aide pour l'initialisation
def initialize_nightscan_integration(nightscan_root: str = None) -> NightScanIntegration:
    """
    Initialise l'intégration NightScan.
    
    Args:
        nightscan_root: Chemin racine du système NightScan
        
    Returns:
        NightScanIntegration: Instance d'intégration configurée
    """
    return NightScanIntegration(nightscan_root)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialisation de l'intégration
    integration = initialize_nightscan_integration()
    
    # Test de synchronisation des données
    csv_files = [
        "data/processed/csv/train.csv",
        "data/processed/csv/val.csv",
        "data/processed/csv/test.csv"
    ]
    
    success = integration.sync_training_data(csv_files)
    if success:
        print("✅ Synchronisation des données réussie")
    else:
        print("❌ Échec de la synchronisation des données")
    
    # Récupération des datasets disponibles
    datasets = integration.get_nightscan_datasets()
    print(f"📊 Datasets disponibles: {len(datasets)}")
    
    # Affichage de l'historique des performances
    history = integration.get_model_performance_history()
    print(f"📈 Historique des modèles: {len(history)} entrées")