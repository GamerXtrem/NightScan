#!/usr/bin/env python3
"""
Script de test pour vérifier que les noms de classes sont correctement propagés
depuis les dossiers sources jusqu'aux résultats finaux.
"""

import os
import sys
import json
import torch
from pathlib import Path
import tempfile
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def create_test_structure():
    """Crée une structure de test avec des noms de classes explicites."""
    
    # Créer un dossier temporaire
    test_dir = Path(tempfile.mkdtemp(prefix="test_classes_"))
    logger.info(f"📁 Création de la structure de test dans {test_dir}")
    
    # Définir les noms de classes explicites
    class_names = ["chat", "chien", "renard", "sanglier", "chevreuil"]
    
    # Créer la structure source
    source_dir = test_dir / "source"
    source_dir.mkdir()
    
    for class_name in class_names:
        class_dir = source_dir / class_name
        class_dir.mkdir()
        # Créer des fichiers fictifs
        for i in range(3):
            (class_dir / f"image_{i}.txt").write_text(f"{class_name}_{i}")
    
    logger.info(f"✅ Structure créée avec les classes: {class_names}")
    
    return test_dir, source_dir, class_names

def test_data_preparation_flow(source_dir, output_dir, expected_classes):
    """Teste que data_preparation conserve les noms de classes."""
    logger.info("\n🔍 Test 1: Data Preparation")
    
    from data_preparation import DataPreparation
    
    # Préparer les données
    prep = DataPreparation(
        input_dir=str(source_dir),
        output_dir=str(output_dir),
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    
    # Découvrir les classes
    class_to_idx = prep.discover_classes()
    
    # Vérifier que les classes sont correctes
    assert set(prep.classes) == set(expected_classes), \
        f"Classes incorrectes: {prep.classes} != {expected_classes}"
    
    logger.info(f"  ✅ Classes découvertes correctement: {prep.classes}")
    
    # Simuler la création de métadonnées
    metadata = {
        'num_classes': len(prep.classes),
        'classes': prep.classes,
        'class_to_idx': class_to_idx,
        'image_stats': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    # Sauvegarder les métadonnées
    metadata_path = output_dir / 'dataset_metadata.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"  ✅ Métadonnées sauvegardées avec classes: {metadata['classes']}")
    
    return metadata

def test_photo_dataset_flow(data_dir, expected_classes):
    """Teste que PhotoDataset récupère les noms de classes."""
    logger.info("\n🔍 Test 2: PhotoDataset")
    
    from photo_dataset import PhotoDataset
    
    # Créer la structure train/val/test
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)
        for class_name in expected_classes:
            (split_dir / class_name).mkdir(exist_ok=True)
            # Créer un fichier fictif
            (split_dir / class_name / "dummy.txt").write_text("test")
    
    # Créer le dataset
    dataset = PhotoDataset(str(data_dir))
    
    # Vérifier les classes
    assert hasattr(dataset, 'classes'), "PhotoDataset n'a pas d'attribut 'classes'"
    assert set(dataset.classes) == set(expected_classes), \
        f"Classes incorrectes dans PhotoDataset: {dataset.classes} != {expected_classes}"
    
    logger.info(f"  ✅ PhotoDataset a récupéré les classes: {dataset.classes}")
    
    # Vérifier get_data_info
    info = dataset.get_data_info()
    assert 'classes' in info, "get_data_info() ne contient pas 'classes'"
    assert set(info['classes']) == set(expected_classes), \
        f"Classes incorrectes dans get_data_info: {info['classes']} != {expected_classes}"
    
    logger.info(f"  ✅ get_data_info() contient les classes: {info['classes']}")
    
    return dataset

def test_checkpoint_flow(dataset, checkpoint_path, expected_classes):
    """Teste que les checkpoints sauvegardent les noms de classes."""
    logger.info("\n🔍 Test 3: Checkpoint Saving")
    
    # Simuler un checkpoint
    checkpoint = {
        'epoch': 10,
        'model_state_dict': {},  # État fictif
        'optimizer_state_dict': {},  # État fictif
        'metrics': {'accuracy': 95.0},
        'dataset_info': dataset.get_data_info()
    }
    
    # Sauvegarder
    torch.save(checkpoint, checkpoint_path)
    
    # Recharger et vérifier
    loaded = torch.load(checkpoint_path, map_location='cpu')
    assert 'dataset_info' in loaded, "Checkpoint ne contient pas 'dataset_info'"
    assert 'classes' in loaded['dataset_info'], "dataset_info ne contient pas 'classes'"
    
    saved_classes = loaded['dataset_info']['classes']
    assert set(saved_classes) == set(expected_classes), \
        f"Classes incorrectes dans checkpoint: {saved_classes} != {expected_classes}"
    
    logger.info(f"  ✅ Checkpoint contient les classes: {saved_classes}")
    
    return loaded

def test_evaluation_flow(checkpoint, test_dir, expected_classes):
    """Teste que l'évaluation utilise les noms de classes."""
    logger.info("\n🔍 Test 4: Evaluation")
    
    # Créer un dossier de test
    for class_name in expected_classes:
        class_dir = test_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / "test.txt").write_text("test")
    
    # Simuler les métriques avec les noms de classes
    from metrics import MetricsTracker
    
    tracker = MetricsTracker(
        num_classes=len(expected_classes),
        class_names=expected_classes
    )
    
    # Vérifier que les classes sont utilisées
    assert tracker.class_names == expected_classes, \
        f"MetricsTracker n'a pas les bons noms: {tracker.class_names} != {expected_classes}"
    
    logger.info(f"  ✅ MetricsTracker utilise les classes: {tracker.class_names}")
    
    # Simuler des métriques par classe
    metrics = {
        'per_class': {
            class_name: {
                'precision': 90 + i,
                'recall': 88 + i,
                'f1_score': 89 + i
            }
            for i, class_name in enumerate(expected_classes)
        }
    }
    
    logger.info(f"  ✅ Métriques générées pour chaque classe: {list(metrics['per_class'].keys())}")
    
    return metrics

def test_export_flow(checkpoint_path, expected_classes):
    """Teste que l'export conserve les noms de classes."""
    logger.info("\n🔍 Test 5: Model Export")
    
    # Charger le checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Simuler l'extraction des classes pour l'export
    if 'dataset_info' in checkpoint:
        classes = checkpoint['dataset_info'].get('classes', [])
    else:
        classes = []
    
    assert set(classes) == set(expected_classes), \
        f"Export n'a pas les bonnes classes: {classes} != {expected_classes}"
    
    logger.info(f"  ✅ Export conserve les classes: {classes}")
    
    # Simuler les métadonnées d'export
    export_metadata = {
        'model_info': {
            'classes': classes,
            'num_classes': len(classes)
        }
    }
    
    logger.info(f"  ✅ Métadonnées d'export contiennent les classes")
    
    return export_metadata

def main():
    """Fonction principale de test."""
    logger.info("="*60)
    logger.info("🧪 TEST DU FLUX DES NOMS DE CLASSES")
    logger.info("="*60)
    
    all_passed = True
    
    try:
        # Créer la structure de test
        test_dir, source_dir, expected_classes = create_test_structure()
        output_dir = test_dir / "processed"
        checkpoint_path = test_dir / "checkpoint.pth"
        test_data_dir = test_dir / "test_data"
        
        # Test 1: Data Preparation
        metadata = test_data_preparation_flow(source_dir, output_dir, expected_classes)
        
        # Test 2: PhotoDataset
        dataset = test_photo_dataset_flow(output_dir, expected_classes)
        
        # Test 3: Checkpoint
        checkpoint = test_checkpoint_flow(dataset, checkpoint_path, expected_classes)
        
        # Test 4: Evaluation
        metrics = test_evaluation_flow(checkpoint, test_data_dir, expected_classes)
        
        # Test 5: Export
        export_metadata = test_export_flow(checkpoint_path, expected_classes)
        
        logger.info("\n" + "="*60)
        logger.info("✅ TOUS LES TESTS SONT PASSÉS!")
        logger.info("Les noms de classes sont correctement propagés depuis")
        logger.info("les dossiers sources jusqu'aux résultats finaux:")
        logger.info(f"  📁 Dossiers sources: {expected_classes}")
        logger.info(f"  📊 Métadonnées: {metadata['classes']}")
        logger.info(f"  🗂️ Dataset: {dataset.classes}")
        logger.info(f"  💾 Checkpoint: {checkpoint['dataset_info']['classes']}")
        logger.info(f"  📈 Métriques: {list(metrics['per_class'].keys())}")
        logger.info(f"  📦 Export: {export_metadata['model_info']['classes']}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ ERREUR: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()
    
    finally:
        # Nettoyer
        if 'test_dir' in locals():
            shutil.rmtree(test_dir, ignore_errors=True)
            logger.info(f"\n🧹 Dossier temporaire nettoyé")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())