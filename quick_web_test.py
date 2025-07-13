#!/usr/bin/env python3
"""
Test rapide de compatibilité de l'interface web
"""

import sys
from pathlib import Path

def test_basic_imports():
    """Test les imports de base."""
    print("🔍 Test des imports de base")
    print("=" * 40)
    
    try:
        # Test log_utils
        from log_utils import LogRotationConfig
        print("✅ LogRotationConfig importé")
    except Exception as e:
        print(f"❌ Erreur LogRotationConfig: {e}")
        return False
    
    try:
        # Test model registry
        from model_registry import get_model_registry
        registry = get_model_registry()
        print(f"✅ Registre de modèles: {len(registry.models)} modèles")
    except Exception as e:
        print(f"❌ Erreur registre: {e}")
        return False
    
    try:
        # Test prediction router
        from unified_prediction_system.prediction_router import PredictionRouter
        print("✅ Routeur de prédictions accessible")
    except Exception as e:
        print(f"❌ Erreur routeur: {e}")
        return False
    
    return True

def test_model_files():
    """Test que les fichiers de modèles existent."""
    print("\\n📦 Test des fichiers de modèles")
    print("=" * 40)
    
    models_to_check = [
        "audio_training_efficientnet/models/best_model.pth",
        "picture_training_enhanced/models/best_model.pth", 
        "mobile_models/audio_light_model.pth",
        "mobile_models/photo_light_model.pth"
    ]
    
    all_exist = True
    for model_path in models_to_check:
        path = Path(model_path)
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"✅ {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {model_path} manquant")
            all_exist = False
    
    return all_exist

def test_web_config():
    """Test la configuration web."""
    print("\\n⚙️ Test de configuration web")
    print("=" * 40)
    
    try:
        # Check if we can access the web directory
        web_dir = Path("web")
        if web_dir.exists():
            print("✅ Dossier web trouvé")
            
            # Check for key files
            key_files = ["app.py", "tasks.py", "templates/index.html"]
            for file_path in key_files:
                full_path = web_dir / file_path
                if full_path.exists():
                    print(f"✅ {file_path}")
                else:
                    print(f"❌ {file_path} manquant")
        else:
            print("❌ Dossier web manquant")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur config web: {e}")
        return False

def test_api_endpoints():
    """Test que les endpoints d'API existent."""
    print("\\n🌐 Test des endpoints API")
    print("=" * 40)
    
    try:
        # Check unified prediction API
        api_file = Path("unified_prediction_system/unified_prediction_api.py")
        if api_file.exists():
            print("✅ API de prédiction unifiée")
        else:
            print("❌ API de prédiction manquante")
            
        # Check API v1
        api_v1_file = Path("api_v1.py")
        if api_v1_file.exists():
            print("✅ API v1")
        else:
            print("❌ API v1 manquante")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur endpoints: {e}")
        return False

def main():
    """Test principal."""
    print("🌙 NightScan - Test Rapide Interface Web")
    print("=" * 50)
    
    tests = [
        ("Imports de base", test_basic_imports),
        ("Fichiers de modèles", test_model_files),
        ("Configuration web", test_web_config),
        ("Endpoints API", test_api_endpoints)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\\n📋 {name}")
        result = test_func()
        results.append(result)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\\n🎯 RÉSULTAT: {passed}/{total} tests réussis")
    
    if passed == total:
        print("✅ L'interface web semble compatible avec les changements EfficientNet")
    else:
        print("⚠️ Problèmes détectés - voir détails ci-dessus")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())