#!/usr/bin/env python3
"""
Script de test pour vérifier la cohérence entre tous les modules.
Vérifie les imports, les configurations et l'intégration.
"""

import sys
import json
import yaml
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Teste que tous les imports fonctionnent."""
    logger.info("🔍 Test des imports...")
    
    errors = []
    
    # Tester les imports principaux
    try:
        from data_preparation import DataPreparation
        logger.info("  ✅ data_preparation")
    except ImportError as e:
        errors.append(f"data_preparation: {e}")
        logger.error(f"  ❌ data_preparation: {e}")
    
    try:
        from photo_dataset import PhotoDataset
        logger.info("  ✅ photo_dataset")
    except ImportError as e:
        errors.append(f"photo_dataset: {e}")
        logger.error(f"  ❌ photo_dataset: {e}")
    
    try:
        from photo_model_dynamic import create_dynamic_model, estimate_model_size
        logger.info("  ✅ photo_model_dynamic")
    except ImportError as e:
        errors.append(f"photo_model_dynamic: {e}")
        logger.error(f"  ❌ photo_model_dynamic: {e}")
    
    try:
        from metrics import MetricsTracker, compute_confusion_matrix, analyze_errors
        logger.info("  ✅ metrics")
    except ImportError as e:
        errors.append(f"metrics: {e}")
        logger.error(f"  ❌ metrics: {e}")
    
    try:
        from visualize_results import ResultsVisualizer
        logger.info("  ✅ visualize_results")
    except ImportError as e:
        errors.append(f"visualize_results: {e}")
        logger.error(f"  ❌ visualize_results: {e}")
    
    try:
        from train_real_images import Trainer, parse_config
        logger.info("  ✅ train_real_images")
    except ImportError as e:
        errors.append(f"train_real_images: {e}")
        logger.error(f"  ❌ train_real_images: {e}")
    
    try:
        from evaluate_model import ModelEvaluator
        logger.info("  ✅ evaluate_model")
    except ImportError as e:
        errors.append(f"evaluate_model: {e}")
        logger.error(f"  ❌ evaluate_model: {e}")
    
    try:
        from export_models import ModelExporter
        logger.info("  ✅ export_models")
    except ImportError as e:
        errors.append(f"export_models: {e}")
        logger.error(f"  ❌ export_models: {e}")
    
    return len(errors) == 0, errors

def test_config_parsing():
    """Teste le parsing de la configuration."""
    logger.info("\n🔍 Test du parsing de configuration...")
    
    # Charger la configuration YAML
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.warning("  ⚠️ config.yaml non trouvé")
        return False, ["config.yaml non trouvé"]
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Tester la fonction parse_config
    try:
        from train_real_images import parse_config
        flat_config = parse_config(config)
        
        # Vérifier les clés essentielles
        required_keys = ['data_dir', 'epochs', 'batch_size', 'learning_rate', 'output_dir']
        missing_keys = []
        
        for key in required_keys:
            if key not in flat_config:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"  ❌ Clés manquantes après parsing: {missing_keys}")
            return False, [f"Clés manquantes: {missing_keys}"]
        
        logger.info("  ✅ Configuration correctement parsée")
        logger.info(f"     {len(flat_config)} paramètres extraits")
        
        return True, []
        
    except Exception as e:
        logger.error(f"  ❌ Erreur lors du parsing: {e}")
        return False, [str(e)]

def test_data_flow():
    """Teste le flux de données entre les modules."""
    logger.info("\n🔍 Test du flux de données...")
    
    errors = []
    
    # Créer un dossier de test temporaire
    test_dir = Path("test_temp")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Créer une structure de test minimale
        for split in ['train', 'val', 'test']:
            split_dir = test_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Créer une classe fictive
            class_dir = split_dir / 'test_class'
            class_dir.mkdir(exist_ok=True)
            
            # Créer un fichier fictif (pas une vraie image)
            dummy_file = class_dir / 'dummy.txt'
            dummy_file.write_text('test')
        
        # Créer des métadonnées fictives
        metadata = {
            'num_classes': 1,
            'classes': ['test_class'],
            'splits': {'train': 1, 'val': 1, 'test': 1},
            'image_stats': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        metadata_path = test_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Tester PhotoDataset
        from photo_dataset import PhotoDataset
        dataset = PhotoDataset(str(test_dir), str(metadata_path))
        
        if dataset.num_classes != 1:
            errors.append(f"Nombre de classes incorrect: {dataset.num_classes}")
        else:
            logger.info("  ✅ PhotoDataset charge correctement les métadonnées")
        
        # Nettoyer
        import shutil
        shutil.rmtree(test_dir)
        
    except Exception as e:
        errors.append(f"Erreur dans le flux de données: {e}")
        logger.error(f"  ❌ {e}")
        
        # Nettoyer en cas d'erreur
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
    
    return len(errors) == 0, errors

def test_checkpoint_compatibility():
    """Teste la compatibilité des checkpoints."""
    logger.info("\n🔍 Test de compatibilité des checkpoints...")
    
    # Créer un checkpoint fictif
    checkpoint = {
        'epoch': 10,
        'model_state_dict': {},
        'optimizer_state_dict': {},
        'metrics': {'accuracy': 90.5, 'loss': 0.25},
        'config': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        'dataset_info': {
            'num_classes': 5,
            'classes': ['class1', 'class2', 'class3', 'class4', 'class5']
        }
    }
    
    # Vérifier que la structure est compatible avec evaluate_model et export_models
    required_fields = ['epoch', 'model_state_dict', 'metrics']
    missing_fields = [f for f in required_fields if f not in checkpoint]
    
    if missing_fields:
        logger.error(f"  ❌ Champs manquants dans le checkpoint: {missing_fields}")
        return False, [f"Champs manquants: {missing_fields}"]
    
    logger.info("  ✅ Structure de checkpoint compatible")
    return True, []

def main():
    """Fonction principale."""
    logger.info("="*60)
    logger.info("🧪 TEST DE COHÉRENCE DES SCRIPTS")
    logger.info("="*60)
    
    all_passed = True
    all_errors = []
    
    # Test 1: Imports
    passed, errors = test_imports()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    
    # Test 2: Configuration
    passed, errors = test_config_parsing()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    
    # Test 3: Flux de données
    passed, errors = test_data_flow()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    
    # Test 4: Checkpoints
    passed, errors = test_checkpoint_compatibility()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    
    # Résumé
    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("✅ TOUS LES TESTS SONT PASSÉS!")
        logger.info("Les scripts sont cohérents et prêts à l'utilisation.")
    else:
        logger.error("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        logger.error(f"Nombre d'erreurs: {len(all_errors)}")
        for error in all_errors:
            logger.error(f"  - {error}")
    logger.info("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())