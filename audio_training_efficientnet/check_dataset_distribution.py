#!/usr/bin/env python3
"""
Script pour vérifier la distribution des classes dans la base de données SQLite
"""

import sqlite3
import argparse
from pathlib import Path
import pandas as pd

def check_distribution(db_path: str):
    """Vérifie la distribution des classes par split."""
    conn = sqlite3.connect(db_path)
    
    # Distribution globale
    print("\n=== Distribution globale ===")
    query = """
    SELECT class_name, COUNT(*) as count 
    FROM audio_samples 
    GROUP BY class_name 
    ORDER BY count DESC
    """
    df = pd.read_sql_query(query, conn)
    print(df.to_string())
    print(f"\nTotal: {df['count'].sum()} échantillons, {len(df)} classes")
    
    # Distribution par split
    print("\n=== Distribution par split ===")
    query = """
    SELECT split, class_name, COUNT(*) as count 
    FROM audio_samples 
    GROUP BY split, class_name 
    ORDER BY split, class_name
    """
    df = pd.read_sql_query(query, conn)
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) > 0:
            print(f"\n{split.upper()}: {split_df['count'].sum()} échantillons")
            print(split_df[['class_name', 'count']].to_string(index=False))
            
            # Statistiques
            counts = split_df['count'].values
            print(f"  Min: {counts.min()}, Max: {counts.max()}, Moyenne: {counts.mean():.1f}")
            
            # Classes avec très peu d'échantillons
            low_count = split_df[split_df['count'] < 10]
            if len(low_count) > 0:
                print(f"  ⚠️  {len(low_count)} classes avec < 10 échantillons!")
    
    # Vérifier les classes vides
    print("\n=== Classes vides par split ===")
    query = """
    SELECT a.class_name, 
           COALESCE(t.count, 0) as train_count,
           COALESCE(v.count, 0) as val_count,
           COALESCE(te.count, 0) as test_count
    FROM (SELECT DISTINCT class_name FROM audio_samples) a
    LEFT JOIN (SELECT class_name, COUNT(*) as count FROM audio_samples WHERE split='train' GROUP BY class_name) t ON a.class_name = t.class_name
    LEFT JOIN (SELECT class_name, COUNT(*) as count FROM audio_samples WHERE split='val' GROUP BY class_name) v ON a.class_name = v.class_name
    LEFT JOIN (SELECT class_name, COUNT(*) as count FROM audio_samples WHERE split='test' GROUP BY class_name) te ON a.class_name = te.class_name
    WHERE t.count IS NULL OR v.count IS NULL OR te.count IS NULL
    """
    df = pd.read_sql_query(query, conn)
    if len(df) > 0:
        print("⚠️  Classes avec des splits vides:")
        print(df.to_string())
    else:
        print("✅ Toutes les classes sont présentes dans tous les splits")
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérifier la distribution du dataset")
    parser.add_argument("db_path", help="Chemin vers la base SQLite")
    args = parser.parse_args()
    
    if not Path(args.db_path).exists():
        print(f"Erreur: {args.db_path} n'existe pas")
        exit(1)
    
    check_distribution(args.db_path)