#!/usr/bin/env python3
"""
Script de démarrage pour le service de localisation NightScan Pi
Lance l'API de localisation sur le port 5001
"""

import os
import sys
import logging
import signal
import time
from pathlib import Path
from threading import Thread
import subprocess

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('location_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocationService:
    """Service de gestion de la localisation NightScan Pi"""
    
    def __init__(self):
        self.process = None
        self.running = False
        self.script_dir = Path(__file__).parent
        
    def start(self):
        """Démarre le service de localisation"""
        try:
            logger.info("🌍 Démarrage du service de localisation NightScan Pi...")
            
            # Vérifier que les dépendances sont installées
            self._check_dependencies()
            
            # Démarrer l'API de localisation
            api_script = self.script_dir / "location_api.py"
            if not api_script.exists():
                raise FileNotFoundError(f"Script API non trouvé: {api_script}")
            
            # Lancer l'API en arrière-plan
            self.process = subprocess.Popen([
                sys.executable, str(api_script)
            ], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.script_dir)
            )
            
            self.running = True
            logger.info("✅ Service de localisation démarré (PID: {})".format(self.process.pid))
            logger.info("📍 API disponible sur http://0.0.0.0:5001")
            
            # Configurer les gestionnaires de signaux
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Attendre et surveiller le processus
            self._monitor_process()
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage: {e}")
            self.stop()
            sys.exit(1)
    
    def stop(self):
        """Arrête le service de localisation"""
        if self.process and self.running:
            logger.info("🛑 Arrêt du service de localisation...")
            
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Arrêt forcé du processus")
                self.process.kill()
                self.process.wait()
            
            self.running = False
            logger.info("✅ Service de localisation arrêté")
    
    def _check_dependencies(self):
        """Vérifie que les dépendances sont installées"""
        required_packages = [
            'flask',
            'flask-cors',
            'requests',
            'sqlite3'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                if package == 'sqlite3':
                    import sqlite3
                else:
                    __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Packages manquants: {missing_packages}")
            logger.info("Installation des dépendances...")
            
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install'
                ] + missing_packages)
                logger.info("✅ Dépendances installées")
            except subprocess.CalledProcessError as e:
                raise Exception(f"Échec de l'installation des dépendances: {e}")
    
    def _monitor_process(self):
        """Surveille le processus de l'API"""
        while self.running:
            try:
                # Vérifier si le processus est toujours vivant
                if self.process.poll() is not None:
                    logger.error("❌ Le processus API s'est arrêté de manière inattendue")
                    self.running = False
                    break
                
                # Attendre un peu
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Interruption clavier détectée")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Erreur lors de la surveillance: {e}")
                time.sleep(5)
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire de signaux pour arrêt propre"""
        logger.info(f"Signal {signum} reçu, arrêt du service...")
        self.running = False
        self.stop()
    
    def status(self):
        """Retourne le statut du service"""
        if self.running and self.process and self.process.poll() is None:
            return {
                'running': True,
                'pid': self.process.pid,
                'uptime': 'N/A'  # Pourrait être calculé
            }
        else:
            return {
                'running': False,
                'pid': None,
                'uptime': None
            }


def main():
    """Fonction principale"""
    service = LocationService()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start':
            service.start()
        elif command == 'stop':
            service.stop()
        elif command == 'status':
            status = service.status()
            if status['running']:
                print(f"✅ Service en cours d'exécution (PID: {status['pid']})")
            else:
                print("❌ Service arrêté")
        elif command == 'restart':
            service.stop()
            time.sleep(2)
            service.start()
        else:
            print("Usage: python start_location_service.py [start|stop|status|restart]")
    else:
        # Démarrage par défaut
        service.start()


if __name__ == "__main__":
    main()