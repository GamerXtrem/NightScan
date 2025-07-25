#!/usr/bin/env python3
"""
Crée un fichier classes.json à partir de la structure du dataset
"""

import json
from pathlib import Path
import argparse

def create_classes_json(audio_dir: Path, output_file: Path):
    """Crée le fichier classes.json depuis la structure des dossiers."""
    
    # Scanner les dossiers pour trouver les classes
    classes = []
    for class_dir in sorted(audio_dir.iterdir()):
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            classes.append(class_dir.name)
    
    # Créer le mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    
    # Créer le dictionnaire
    class_info = {
        'num_classes': len(classes),
        'class_names': classes,
        'class_to_idx': class_to_idx
    }
    
    # Sauvegarder
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)
    
    print(f"Fichier créé: {output_file}")
    print(f"Nombre de classes: {len(classes)}")
    print(f"Classes: {', '.join(classes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Créer classes.json")
    parser.add_argument("audio_dir", type=Path, help="Répertoire du dataset")
    parser.add_argument("--output", type=Path, default=Path("data/processed/csv/classes.json"),
                       help="Fichier de sortie")
    args = parser.parse_args()
    
    # Créer le répertoire si nécessaire
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    create_classes_json(args.audio_dir, args.output)