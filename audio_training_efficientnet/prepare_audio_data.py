#!/usr/bin/env python3
"""
Script de préparation des données audio pour NightScan
Scanne les dossiers d'audio et crée les fichiers CSV pour l'entraînement
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import json
from typing import List, Tuple, Dict
import logging

# Importer le module de segmentation
from audio_segmentation import AudioSegmenter

logger = logging.getLogger(__name__)

def scan_audio_directory(audio_dir: Path) -> Dict[str, List[Path]]:
    """
    Scanne le répertoire audio et retourne un dictionnaire {classe: [fichiers]}.
    
    Args:
        audio_dir: Répertoire contenant les sous-dossiers de classes
        
    Returns:
        Dict mapping classe -> liste de fichiers WAV
    """
    class_files = {}
    
    # Scanner chaque sous-dossier
    for class_dir in audio_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        
        # Ignorer les dossiers cachés
        if class_name.startswith('.'):
            continue
            
        # Collecter tous les fichiers WAV dans ce dossier
        wav_files = list(class_dir.rglob('*.wav'))
        
        # Filtrer les fichiers cachés macOS (commençant par ._)
        wav_files = [f for f in wav_files if not f.name.startswith('._')]
        
        if wav_files:
            # Limiter à 500 fichiers maximum par classe
            if len(wav_files) > 500:
                # Échantillonner aléatoirement 500 fichiers
                import random
                random.seed(42)  # Pour la reproductibilité
                wav_files = random.sample(wav_files, 500)
                print(f"Classe '{class_name}': {len(wav_files)} fichiers WAV sélectionnés (limité à 500)")
            else:
                print(f"Classe '{class_name}': {len(wav_files)} fichiers WAV trouvés")
            
            class_files[class_name] = wav_files
    
    return class_files

def create_dataset_splits(class_files: Dict[str, List[Path]], 
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Crée les splits train/val/test avec stratification par classe.
    
    Args:
        class_files: Dictionnaire {classe: [fichiers]}
        train_ratio: Proportion pour l'entraînement
        val_ratio: Proportion pour la validation
        test_ratio: Proportion pour le test
        random_state: Seed pour la reproductibilité
        
    Returns:
        Tuple de DataFrames (train, val, test)
    """
    all_files = []
    all_labels = []
    
    # Rassembler tous les fichiers et labels
    for class_name, files in class_files.items():
        for file_path in files:
            all_files.append(str(file_path))
            all_labels.append(class_name)
    
    # Créer un DataFrame
    df = pd.DataFrame({
        'filename': all_files,
        'label': all_labels
    })
    
    # Premier split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['label'],
        random_state=random_state
    )
    
    print(f"\nSplits créés:")
    print(f"- Train: {len(train_df)} échantillons")
    print(f"- Val: {len(val_df)} échantillons")
    print(f"- Test: {len(test_df)} échantillons")
    
    # Afficher la distribution par classe
    print("\nDistribution par classe:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n{split_name}:")
        print(split_df['label'].value_counts().to_string())
    
    return train_df, val_df, test_df

def save_datasets(train_df: pd.DataFrame, 
                  val_df: pd.DataFrame, 
                  test_df: pd.DataFrame,
                  output_dir: Path,
                  relative_to: Path = None):
    """
    Sauvegarde les datasets en CSV.
    
    Args:
        train_df, val_df, test_df: DataFrames à sauvegarder
        output_dir: Répertoire de sortie pour les CSV
        relative_to: Si fourni, les chemins seront relatifs à ce répertoire
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convertir les chemins en relatifs si demandé
    if relative_to:
        for df in [train_df, val_df, test_df]:
            df['filename'] = df['filename'].apply(
                lambda p: str(Path(p).relative_to(relative_to))
            )
    
    # Sauvegarder les CSV
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    print(f"\nFichiers CSV sauvegardés dans: {output_dir}")

def save_class_names(class_names: List[str], output_path: Path):
    """
    Sauvegarde la liste des classes dans un fichier JSON.
    
    Args:
        class_names: Liste des noms de classes
        output_path: Chemin du fichier JSON de sortie
    """
    class_info = {
        'num_classes': len(class_names),
        'class_names': sorted(class_names),
        'class_to_idx': {name: idx for idx, name in enumerate(sorted(class_names))}
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)
    
    print(f"\nInformations sur les classes sauvegardées dans: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Prépare les données audio pour l'entraînement NightScan"
    )
    parser.add_argument(
        'audio_dir',
        type=Path,
        help="Répertoire contenant les sous-dossiers de classes audio"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed/csv'),
        help="Répertoire de sortie pour les CSV (défaut: data/processed/csv)"
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help="Proportion pour l'entraînement (défaut: 0.7)"
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help="Proportion pour la validation (défaut: 0.15)"
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help="Proportion pour le test (défaut: 0.15)"
    )
    parser.add_argument(
        '--relative-paths',
        action='store_true',
        help="Utiliser des chemins relatifs dans les CSV"
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help="Nombre minimum d'échantillons par classe (défaut: 10)"
    )
    
    # Nouveaux arguments pour la segmentation
    parser.add_argument(
        '--segment',
        action='store_true',
        help="Segmenter les fichiers audio longs avant la préparation"
    )
    parser.add_argument(
        '--segment-duration',
        type=float,
        default=8.0,
        help="Durée des segments en secondes (défaut: 8)"
    )
    parser.add_argument(
        '--segment-overlap',
        type=float,
        default=2.0,
        help="Chevauchement entre segments en secondes (défaut: 2)"
    )
    parser.add_argument(
        '--segment-dir',
        type=Path,
        default=None,
        help="Répertoire pour les fichiers segmentés (défaut: audio_dir_segmented)"
    )
    
    args = parser.parse_args()
    
    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Vérifier que le répertoire existe
    if not args.audio_dir.exists():
        print(f"Erreur: Le répertoire {args.audio_dir} n'existe pas!")
        return 1
    
    # Gérer la segmentation si demandée
    working_dir = args.audio_dir
    
    if args.segment:
        print(f"\n🔪 Segmentation des fichiers audio longs...")
        
        # Déterminer le répertoire de sortie pour les segments
        if args.segment_dir is None:
            segment_dir = args.audio_dir.parent / f"{args.audio_dir.name}_segmented"
        else:
            segment_dir = args.segment_dir
        
        # Créer le segmenteur
        segmenter = AudioSegmenter(
            segment_duration=args.segment_duration,
            overlap=args.segment_overlap,
            min_segment_duration=3.0
        )
        
        # Segmenter les fichiers
        segments_info = segmenter.segment_directory(
            args.audio_dir,
            segment_dir,
            preserve_structure=True
        )
        
        if not segments_info:
            print("Aucun fichier n'a été segmenté (tous les fichiers sont déjà courts)")
        else:
            print(f"✅ Segmentation terminée. Segments créés dans: {segment_dir}")
            working_dir = segment_dir
    
    # Scanner le répertoire (original ou segmenté)
    print(f"\nScan du répertoire: {working_dir}")
    class_files = scan_audio_directory(working_dir)
    
    if not class_files:
        print("Erreur: Aucune classe trouvée!")
        return 1
    
    # Filtrer les classes avec trop peu d'échantillons
    filtered_classes = {}
    for class_name, files in class_files.items():
        if len(files) >= args.min_samples:
            filtered_classes[class_name] = files
        else:
            print(f"⚠️  Classe '{class_name}' ignorée (seulement {len(files)} échantillons, minimum requis: {args.min_samples})")
    
    if not filtered_classes:
        print("Erreur: Aucune classe n'a assez d'échantillons!")
        return 1
    
    print(f"\n{len(filtered_classes)} classes retenues pour l'entraînement")
    
    # Créer les splits
    train_df, val_df, test_df = create_dataset_splits(
        filtered_classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Sauvegarder les CSV
    relative_to = working_dir if args.relative_paths else None
    save_datasets(train_df, val_df, test_df, args.output_dir, relative_to)
    
    # Sauvegarder les informations sur les classes
    class_names = list(filtered_classes.keys())
    save_class_names(class_names, args.output_dir / 'classes.json')
    
    print("\n✅ Préparation des données terminée!")
    print(f"Classes détectées: {', '.join(sorted(class_names))}")
    
    if args.segment and segments_info:
        print(f"\n💡 Conseil: Utilisez le répertoire segmenté pour l'entraînement:")
        print(f"   python train_audio.py --data-dir {segment_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())