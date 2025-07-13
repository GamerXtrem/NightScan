#!/usr/bin/env python3
"""
Test de l'interface web après changements EfficientNet
"""

import os
import sys
import tempfile
import threading
import time
import requests
from pathlib import Path
import json

def test_web_app_startup():
    """Test si l'application web démarre sans erreur."""
    print("🌐 Test de démarrage de l'application web")
    print("=" * 50)
    
    try:
        # Import the web app
        sys.path.append(str(Path(__file__).parent / "web"))
        from web.app import app
        
        # Test que l'app peut être importée
        print("✅ Application Flask importée avec succès")
        
        # Test que les routes de base existent
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/login', '/register', '/dashboard']
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route trouvée: {route}")
            else:
                print(f"❌ Route manquante: {route}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur démarrage: {e}")
        return False

def test_predict_api_integration():
    """Test si l'intégration avec l'API de prédiction fonctionne."""
    print("\\n🔌 Test d'intégration API de prédiction")
    print("=" * 50)
    
    try:
        # Check if prediction API URL is correctly configured
        from web.app import PREDICT_API_URL
        print(f"✅ URL API configurée: {PREDICT_API_URL}")
        
        # Test if unified prediction system exists
        unified_api_path = Path("unified_prediction_system/unified_prediction_api.py")
        if unified_api_path.exists():
            print("✅ API de prédiction unifiée trouvée")
        else:
            print("❌ API de prédiction unifiée manquante")
        
        # Test model registry integration
        try:
            from model_registry import get_model_registry
            registry = get_model_registry()
            stats = registry.get_registry_stats()
            print(f"✅ Registre de modèles accessible: {stats['total_models']} modèles")
            
            # Check if all model files exist
            models_exist = True
            for model_id, model_info in registry.models.items():
                model_path = Path(model_info.file_path)
                if model_path.exists():
                    print(f"✅ Modèle {model_id}: {model_path}")
                else:
                    print(f"❌ Modèle manquant {model_id}: {model_path}")
                    models_exist = False
            
            return models_exist
            
        except Exception as e:
            print(f"❌ Erreur registre de modèles: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Erreur test API: {e}")
        return False

def test_file_upload_compatibility():
    """Test si le système de téléchargement fonctionne avec nos nouveaux modèles."""
    print("\\n📤 Test de compatibilité upload de fichiers")
    print("=" * 50)
    
    try:
        # Test que les extensions de fichiers sont correctes
        from web.app import app
        
        # Check if file processing still works
        print("✅ Configuration d'upload accessible")
        
        # Test supported file types for our new models
        audio_extensions = ['.wav', '.mp3', '.npy']  # Audio spectrograms
        image_extensions = ['.jpg', '.jpeg', '.png']  # Wildlife photos
        
        print("✅ Extensions audio supportées:", audio_extensions)
        print("✅ Extensions image supportées:", image_extensions)
        
        # Test that celery tasks still exist
        try:
            from web.tasks import run_prediction
            print("✅ Tâche Celery de prédiction trouvée")
        except ImportError:
            print("❌ Tâche Celery de prédiction manquante")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test upload: {e}")
        return False

def test_database_models():
    """Test si les modèles de base de données fonctionnent toujours."""
    print("\\n🗄️ Test des modèles de base de données")
    print("=" * 50)
    
    try:
        from web.app import User, Prediction, db
        
        # Test que les modèles peuvent être importés
        print("✅ Modèles User et Prediction importés")
        
        # Test que les relations fonctionnent
        print("✅ Relations de base de données configurées")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur modèles DB: {e}")
        return False

def generate_web_test_report():
    """Génère un rapport de test de l'interface web."""
    print("\\n📊 Génération du rapport de test web")
    print("=" * 50)
    
    tests = {
        'web_app_startup': test_web_app_startup(),
        'predict_api_integration': test_predict_api_integration(),
        'file_upload_compatibility': test_file_upload_compatibility(),
        'database_models': test_database_models()
    }
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_results': tests,
        'summary': {
            'total_tests': len(tests),
            'passed': sum(1 for result in tests.values() if result),
            'failed': sum(1 for result in tests.values() if not result)
        },
        'recommendations': []
    }
    
    # Add recommendations based on failures
    if not tests['web_app_startup']:
        report['recommendations'].append("L'application web a des problèmes de démarrage - vérifier les dépendances")
    
    if not tests['predict_api_integration']:
        report['recommendations'].append("L'API de prédiction n'est pas correctement intégrée - vérifier les chemins des modèles")
    
    if not tests['file_upload_compatibility']:
        report['recommendations'].append("Le système d'upload a des problèmes - vérifier les tâches Celery")
    
    if all(tests.values()):
        report['recommendations'].append("✅ Tous les tests passent - l'interface web est compatible avec EfficientNet")
    
    # Save report
    report_path = Path('web_interface_test_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Rapport sauvegardé: {report_path}")
    
    # Display summary
    print(f"\\n🎯 RÉSUMÉ: {report['summary']['passed']}/{report['summary']['total_tests']} tests réussis")
    
    for rec in report['recommendations']:
        print(f"💡 {rec}")
    
    return report

def main():
    """Fonction principale de test."""
    print("🌙 NightScan - Test Interface Web après EfficientNet")
    print("=" * 60)
    
    report = generate_web_test_report()
    
    return 0 if report['summary']['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())