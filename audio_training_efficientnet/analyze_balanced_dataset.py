#!/usr/bin/env python3
"""
Analyse le dataset équilibré pour détecter d'éventuels problèmes
"""

import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_dataset(db_path: str):
    """Analyse la base de données pour vérifier l'équilibre et détecter les problèmes."""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=== Analyse du dataset équilibré ===\n")
    
    # 1. Nombre total d'échantillons par split
    print("1. Distribution par split:")
    cursor.execute("""
        SELECT split, COUNT(*) as count 
        FROM audio_samples 
        GROUP BY split
    """)
    split_counts = {}
    for row in cursor.fetchall():
        split_counts[row[0]] = row[1]
        print(f"   {row[0]}: {row[1]} échantillons")
    
    # 2. Distribution par classe et split
    print("\n2. Distribution par classe et split:")
    cursor.execute("""
        SELECT class_name, split, COUNT(*) as count 
        FROM audio_samples 
        GROUP BY class_name, split 
        ORDER BY class_name, split
    """)
    
    class_split_counts = defaultdict(dict)
    for row in cursor.fetchall():
        class_split_counts[row[0]][row[1]] = row[2]
    
    # Afficher sous forme de tableau
    print(f"   {'Classe':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("   " + "-" * 50)
    for class_name in sorted(class_split_counts.keys()):
        counts = class_split_counts[class_name]
        print(f"   {class_name:<20} {counts.get('train', 0):<10} {counts.get('val', 0):<10} {counts.get('test', 0):<10}")
    
    # 3. Ratio originaux/augmentés
    print("\n3. Ratio originaux/augmentés par split:")
    cursor.execute("""
        SELECT split, is_augmented, COUNT(*) as count 
        FROM audio_samples 
        GROUP BY split, is_augmented 
        ORDER BY split, is_augmented
    """)
    
    for split in ['train', 'val', 'test']:
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN is_augmented = 0 THEN 1 ELSE 0 END) as original,
                SUM(CASE WHEN is_augmented = 1 THEN 1 ELSE 0 END) as augmented
            FROM audio_samples 
            WHERE split = ?
        """, (split,))
        result = cursor.fetchone()
        if result and result[0] is not None:
            orig, aug = result
            total = orig + aug
            print(f"   {split}: {orig} originaux ({orig/total*100:.1f}%), {aug} augmentés ({aug/total*100:.1f}%)")
    
    # 4. Vérifier les fichiers en double entre splits
    print("\n4. Vérification des fuites de données:")
    
    # Vérifier si des fichiers originaux sont dans plusieurs splits
    cursor.execute("""
        SELECT original_file, COUNT(DISTINCT split) as split_count
        FROM audio_samples
        WHERE original_file IS NOT NULL
        GROUP BY original_file
        HAVING split_count > 1
    """)
    
    duplicates = cursor.fetchall()
    if duplicates:
        print(f"   ⚠️  ATTENTION: {len(duplicates)} fichiers originaux apparaissent dans plusieurs splits!")
        for dup in duplicates[:5]:  # Afficher les 5 premiers
            print(f"      - {dup[0]}")
    else:
        print("   ✅ Aucune fuite détectée: chaque original et ses augmentations sont dans un seul split")
    
    # 5. Statistiques sur les types d'augmentation
    print("\n5. Types d'augmentation utilisés:")
    cursor.execute("""
        SELECT augmentation_type, COUNT(*) as count 
        FROM audio_samples 
        WHERE augmentation_type IS NOT NULL 
        GROUP BY augmentation_type 
        ORDER BY count DESC
    """)
    
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]}")
    
    # 6. Calcul du nombre de batches attendus
    print("\n6. Estimation du nombre de batches:")
    for split, count in split_counts.items():
        for batch_size in [32, 64, 128, 256]:
            n_batches = count // batch_size
            print(f"   {split} avec batch_size={batch_size}: {n_batches} batches")
    
    # 7. Vérifier l'équilibre des classes
    print("\n7. Analyse de l'équilibre des classes:")
    for split in ['train', 'val', 'test']:
        counts = [class_split_counts[cls].get(split, 0) for cls in class_split_counts]
        if counts:
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            min_count = min(counts)
            max_count = max(counts)
            print(f"   {split}: moyenne={mean_count:.1f}, std={std_count:.1f}, min={min_count}, max={max_count}")
            if std_count > mean_count * 0.1:
                print(f"      ⚠️  Déséquilibre détecté!")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Analyser le dataset équilibré")
    parser.add_argument('--index-db', type=str, default='balanced_audio_index.db',
                       help='Base SQLite à analyser')
    
    args = parser.parse_args()
    
    if not Path(args.index_db).exists():
        print(f"Erreur: {args.index_db} n'existe pas")
        return
    
    analyze_dataset(args.index_db)


if __name__ == "__main__":
    main()