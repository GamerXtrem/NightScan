#!/usr/bin/env python3
"""
Script de test pour vérifier que toute la chaîne de traitement audio fonctionne correctement
avant de lancer l'entraînement réel.
"""

import sys
import os
from pathlib import Path
import tempfile
import torch
import torchaudio
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent.parent))

# Imports du projet
try:
    from audio_segmentation import AudioSegmenter
    from prepare_audio_data import scan_audio_directory, create_dataset_splits, save_datasets, save_class_names
    from audio_dataset import AudioSpectrogramDataset, create_data_loaders
    from spectrogram_config import get_config_for_animal
    from models.efficientnet_config import create_audio_model
    print("✅ Tous les imports ont réussi")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)


def create_test_audio_files(base_dir: Path, num_classes: int = 3, files_per_class: int = 5) -> dict:
    """Crée des fichiers audio de test pour simuler un dataset."""
    print("\n📁 Création des fichiers audio de test...")
    
    class_files = {}
    sample_rate = 22050
    
    for i in range(num_classes):
        class_name = f"animal_{i+1}"
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_files[class_name] = []
        
        for j in range(files_per_class):
            # Créer des fichiers de durées variables
            durations = [3.0, 7.0, 15.0, 20.0, 5.0]
            duration = durations[j % len(durations)]
            
            # Générer un signal audio simple
            t = torch.linspace(0, duration, int(sample_rate * duration))
            freq = 440 * (i + 1)  # Fréquence différente par classe
            signal = 0.5 * torch.sin(2 * np.pi * freq * t).unsqueeze(0)
            
            # Ajouter du bruit
            signal += 0.1 * torch.randn_like(signal)
            
            # Sauvegarder
            file_path = class_dir / f"audio_{j+1}.wav"
            torchaudio.save(str(file_path), signal, sample_rate)
            class_files[class_name].append(file_path)
            
    print(f"✅ Créé {num_classes} classes avec {files_per_class} fichiers chacune")
    return class_files


def test_segmentation(audio_dir: Path, segmented_dir: Path) -> bool:
    """Test la segmentation audio."""
    print("\n🔪 Test de la segmentation audio...")
    
    try:
        segmenter = AudioSegmenter(
            segment_duration=8.0,
            overlap=2.0,
            min_segment_duration=3.0
        )
        
        segments_info = segmenter.segment_directory(
            audio_dir,
            segmented_dir,
            preserve_structure=True
        )
        
        if segments_info:
            total_segments = sum(len(segs) for segs in segments_info.values())
            print(f"✅ Segmentation réussie: {len(segments_info)} fichiers → {total_segments} segments")
            
            # Vérifier qu'un segment a bien 8 secondes
            for original_file, segments in segments_info.items():
                if segments:
                    first_segment = segments[0]
                    segment_path = segmented_dir / Path(original_file).parent.name / first_segment['filename']
                    if segment_path.exists():
                        info = torchaudio.info(str(segment_path))
                        duration = info.num_frames / info.sample_rate
                        print(f"   - Premier segment: {duration:.1f}s (attendu: 8.0s)")
                        assert abs(duration - 8.0) < 0.1, f"Durée incorrecte: {duration}s"
                    break
                    
            return True
        else:
            print("⚠️  Aucun fichier segmenté (tous les fichiers sont courts)")
            return True
            
    except Exception as e:
        print(f"❌ Erreur segmentation: {e}")
        return False


def test_data_preparation(audio_dir: Path, csv_dir: Path) -> bool:
    """Test la préparation des données CSV."""
    print("\n📋 Test de la préparation des données...")
    
    try:
        # Scanner les fichiers
        class_files = scan_audio_directory(audio_dir)
        print(f"✅ Classes trouvées: {list(class_files.keys())}")
        
        # Créer les splits
        train_df, val_df, test_df = create_dataset_splits(
            class_files,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Sauvegarder
        csv_dir.mkdir(parents=True, exist_ok=True)
        save_datasets(train_df, val_df, test_df, csv_dir, relative_to=audio_dir)
        
        class_names = list(class_files.keys())
        save_class_names(class_names, csv_dir / 'classes.json')
        
        print(f"✅ Fichiers CSV créés dans {csv_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur préparation données: {e}")
        return False


def test_dataset_loading(csv_dir: Path, audio_dir: Path) -> bool:
    """Test le chargement du dataset PyTorch."""
    print("\n🎵 Test du dataset PyTorch...")
    
    try:
        # Créer un dataset
        dataset = AudioSpectrogramDataset(
            csv_file=csv_dir / 'train.csv',
            audio_dir=audio_dir,
            classes_json=csv_dir / 'classes.json',
            animal_type='general',
            augment=False
        )
        
        print(f"✅ Dataset créé: {len(dataset)} échantillons")
        
        # Charger un échantillon
        if len(dataset) > 0:
            try:
                spec, label = dataset[0]
                print(f"   - Forme spectrogramme: {spec.shape} (attendu: [3, 128, X])")
                print(f"   - Type: {spec.dtype}")
                print(f"   - Min/Max: {spec.min():.2f}/{spec.max():.2f}")
                
                assert spec.shape[0] == 3, "Le spectrogramme doit avoir 3 canaux"
                assert spec.shape[1] == 128, "Le spectrogramme doit avoir 128 bandes mel"
            except Exception as e:
                print(f"   - Erreur lors du chargement: {e}")
                import traceback
                traceback.print_exc()
                raise
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur dataset: {e}")
        return False


def test_data_loader(csv_dir: Path, audio_dir: Path) -> bool:
    """Test le DataLoader."""
    print("\n📦 Test du DataLoader...")
    
    try:
        loaders = create_data_loaders(
            csv_dir=csv_dir,
            audio_dir=audio_dir,
            batch_size=4,
            num_workers=0,  # 0 pour éviter les problèmes en test
            augment_train=True
        )
        
        if 'train' in loaders:
            train_loader = loaders['train']
            print(f"✅ DataLoader créé: {len(train_loader)} batches")
            
            # Charger un batch
            for batch_data, batch_labels in train_loader:
                print(f"   - Batch shape: {batch_data.shape}")
                print(f"   - Labels shape: {batch_labels.shape}")
                assert batch_data.shape[1] == 3, "3 canaux attendus"
                assert batch_data.shape[2] == 128, "128 mels attendus"
                break
                
        return True
        
    except Exception as e:
        print(f"❌ Erreur DataLoader: {e}")
        return False


def test_model_creation() -> bool:
    """Test la création du modèle."""
    print("\n🤖 Test du modèle EfficientNet...")
    
    try:
        model = create_audio_model(
            num_classes=3,
            model_name='efficientnet-b1',
            pretrained=True,
            dropout_rate=0.3
        )
        
        # Test sur CPU
        device = torch.device('cpu')
        model.to(device)
        model.eval()
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"✅ Modèle créé et testé")
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output shape: {output.shape}")
        
        assert output.shape == (2, 3), f"Output shape incorrect: {output.shape}"
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return False


def main():
    """Fonction principale de test."""
    print("🧪 Test complet de la chaîne de traitement audio NightScan")
    print("=" * 60)
    
    # Créer un répertoire temporaire pour les tests
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_dir = temp_path / "audio_data"
        segmented_dir = temp_path / "audio_segmented"
        csv_dir = temp_path / "csv"
        
        # Tests
        tests_passed = []
        
        # 1. Créer des fichiers audio de test
        create_test_audio_files(audio_dir)
        
        # 2. Tester la segmentation
        tests_passed.append(("Segmentation", test_segmentation(audio_dir, segmented_dir)))
        
        # 3. Tester la préparation des données
        # Utiliser le répertoire segmenté s'il existe, sinon l'original
        data_dir = segmented_dir if segmented_dir.exists() else audio_dir
        tests_passed.append(("Préparation données", test_data_preparation(data_dir, csv_dir)))
        
        # 4. Tester le dataset
        tests_passed.append(("Dataset PyTorch", test_dataset_loading(csv_dir, data_dir)))
        
        # 5. Tester le DataLoader
        tests_passed.append(("DataLoader", test_data_loader(csv_dir, data_dir)))
        
        # 6. Tester le modèle
        tests_passed.append(("Modèle EfficientNet", test_model_creation()))
        
        # Résumé
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DES TESTS:")
        print("=" * 60)
        
        all_passed = True
        for test_name, passed in tests_passed:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:.<40} {status}")
            if not passed:
                all_passed = False
        
        print("=" * 60)
        
        if all_passed:
            print("\n🎉 Tous les tests sont passés ! Vous pouvez lancer l'entraînement.")
            print("\nCommandes suggérées:")
            print("1. Préparer vos vraies données:")
            print("   python prepare_audio_data.py /chemin/vers/audio --segment")
            print("\n2. Lancer l'entraînement:")
            print("   python train_audio.py --data-dir /chemin/vers/audio_segmented --epochs 50")
            return 0
        else:
            print("\n⚠️  Certains tests ont échoué. Corrigez les erreurs avant de lancer l'entraînement.")
            return 1


if __name__ == "__main__":
    sys.exit(main())