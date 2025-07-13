#!/usr/bin/env python3
"""
Script de démarrage pour le système de prédiction unifiée NightScan

Ce script lance l'API de prédiction et peut optionnellement ouvrir l'interface web.
"""

import os
import sys
import time
import argparse
import logging
import webbrowser
from pathlib import Path
from threading import Thread

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from unified_prediction_system.unified_prediction_api import UnifiedPredictionAPI


def check_dependencies():
    """Vérifie les dépendances requises."""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import flask
    except ImportError:
        missing_deps.append("flask")
    
    try:
        import flask_cors
    except ImportError:
        missing_deps.append("flask-cors")
    
    try:
        import PIL
    except ImportError:
        missing_deps.append("pillow")
    
    if missing_deps:
        print("❌ Dépendances manquantes:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstallez les dépendances avec:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True


def check_models():
    """Vérifie la présence des modèles requis."""
    audio_model_path = Path("audio_training_efficientnet/models/best_model.pth")
    photo_model_path = Path("picture_training_enhanced/models/best_model.pth")
    
    models_status = {
        "audio": audio_model_path.exists(),
        "photo": photo_model_path.exists()
    }
    
    print("📊 État des modèles:")
    print(f"   Audio: {'✅' if models_status['audio'] else '❌'} {audio_model_path}")
    print(f"   Photo: {'✅' if models_status['photo'] else '❌'} {photo_model_path}")
    
    if not any(models_status.values()):
        print("\n⚠️  Aucun modèle trouvé. Assurez-vous d'avoir entraîné au moins un modèle.")
        return False
    
    return True


def start_api_server(host, port, debug, config_path):
    """Lance le serveur API dans un thread séparé."""
    try:
        api = UnifiedPredictionAPI(config_path, debug)
        api.run(host=host, port=port, debug=debug)
    except Exception as e:
        print(f"❌ Erreur démarrage API: {e}")
        sys.exit(1)


def wait_for_api(host, port, timeout=30):
    """Attend que l'API soit disponible."""
    import requests
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    
    return False


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Système de prédiction unifiée NightScan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python start_prediction_system.py                    # Démarrage standard
  python start_prediction_system.py --debug           # Mode debug
  python start_prediction_system.py --no-browser      # Sans ouvrir le navigateur
  python start_prediction_system.py --port 8000       # Port personnalisé
        """
    )
    
    parser.add_argument("--host", default="127.0.0.1", 
                       help="Adresse d'écoute (défaut: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port d'écoute (défaut: 5000)")
    parser.add_argument("--debug", action="store_true",
                       help="Mode debug Flask")
    parser.add_argument("--no-browser", action="store_true",
                       help="Ne pas ouvrir automatiquement le navigateur")
    parser.add_argument("--config", type=Path,
                       help="Fichier de configuration des modèles")
    parser.add_argument("--check-only", action="store_true",
                       help="Vérifier les dépendances et modèles seulement")
    
    args = parser.parse_args()
    
    # Configuration du logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('prediction_system.log')
        ]
    )
    
    print("🌙 NightScan - Système de Prédiction Unifiée")
    print("=" * 50)
    
    # Vérifications préalables
    print("\n🔍 Vérification des dépendances...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Toutes les dépendances sont présentes")
    
    print("\n🔍 Vérification des modèles...")
    if not check_models():
        if not args.check_only:
            response = input("\nContinuer malgré les modèles manquants? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    if args.check_only:
        print("\n✅ Vérification terminée")
        return
    
    # Configuration
    web_interface_path = Path(__file__).parent / "web_interface.html"
    web_url = f"http://{args.host}:{args.port}"
    interface_url = f"file://{web_interface_path.absolute()}"
    
    print(f"\n🚀 Démarrage du système...")
    print(f"   API: {web_url}")
    print(f"   Interface: {interface_url}")
    print(f"   Mode debug: {args.debug}")
    print(f"   Logs: prediction_system.log")
    
    # Démarrer l'API dans un thread séparé
    api_thread = Thread(
        target=start_api_server,
        args=(args.host, args.port, args.debug, args.config),
        daemon=True
    )
    
    try:
        api_thread.start()
        
        # Attendre que l'API soit disponible
        print("\n⏳ Attente du démarrage de l'API...")
        if wait_for_api(args.host, args.port):
            print("✅ API démarrée avec succès")
            
            # Ouvrir le navigateur si demandé
            if not args.no_browser:
                print("🌐 Ouverture de l'interface web...")
                webbrowser.open(interface_url)
                print(f"💡 Interface accessible à: {interface_url}")
            
            print(f"\n🎯 API disponible à: {web_url}")
            print("📚 Documentation API:")
            print(f"   Health check: {web_url}/health")
            print(f"   Statut modèles: {web_url}/models/status")
            print(f"   Statistiques: {web_url}/stats")
            print(f"   Formats supportés: {web_url}/supported-formats")
            
            print("\n✨ Système prêt ! Appuyez sur Ctrl+C pour arrêter.")
            
            # Garder le programme en vie
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n🛑 Arrêt du système...")
                
        else:
            print("❌ Échec du démarrage de l'API")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt du système...")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)
    
    print("👋 Système arrêté")


if __name__ == "__main__":
    main()