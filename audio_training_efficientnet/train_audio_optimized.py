#!/usr/bin/env python3
"""
Script d'entraînement optimisé pour NightScan Audio
Utilise toutes les optimisations de performance par défaut
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse
import torch

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent.parent))


def main():
    """Script wrapper pour lancer l'entraînement avec les paramètres optimisés."""
    parser = argparse.ArgumentParser(
        description="Entraînement optimisé du modèle audio NightScan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Entraînement standard optimisé
  python train_audio_optimized.py --data-dir /chemin/vers/audio_data

  # Entraînement avec prégénération des spectrogrammes (recommandé)
  python train_audio_optimized.py --data-dir /chemin/vers/audio_data --pregenerate

  # Entraînement rapide pour tests
  python train_audio_optimized.py --data-dir /chemin/vers/audio_data --epochs 10 --batch-size 64
        """
    )
    
    # Arguments essentiels
    parser.add_argument('--data-dir', type=str, required=True,
                       help="Répertoire contenant les fichiers audio")
    parser.add_argument('--csv-dir', type=str, default='data/processed/csv',
                       help="Répertoire contenant les CSV (défaut: data/processed/csv)")
    parser.add_argument('--output-dir', type=str, default='audio_training_efficientnet/models',
                       help="Répertoire de sortie pour le modèle")
    
    # Paramètres d'entraînement
    parser.add_argument('--epochs', type=int, default=50,
                       help="Nombre d'epochs (défaut: 50)")
    parser.add_argument('--batch-size', type=int, default=None,
                       help="Taille du batch (auto-détecté si non spécifié)")
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help="Learning rate (défaut: 0.001)")
    
    # Options d'optimisation
    parser.add_argument('--pregenerate', action='store_true',
                       help="Prégénérer tous les spectrogrammes avant l'entraînement (recommandé)")
    parser.add_argument('--no-optimization', action='store_true',
                       help="Désactiver toutes les optimisations (pour comparaison)")
    parser.add_argument('--num-workers', type=int, default=None,
                       help="Nombre de workers (auto-détecté si non spécifié)")
    
    args = parser.parse_args()
    
    # Auto-détecter les paramètres optimaux
    if args.num_workers is None:
        # Utiliser le nombre de CPU - 2, avec un minimum de 4 et maximum de 16
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        args.num_workers = min(max(4, cpu_count - 2), 16)
        print(f"Nombre de workers auto-détecté: {args.num_workers}")
    
    if args.batch_size is None:
        # Auto-détecter la taille de batch selon la mémoire GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 16:
                args.batch_size = 64
            elif gpu_memory >= 8:
                args.batch_size = 32
            else:
                args.batch_size = 16
            print(f"Batch size auto-détecté: {args.batch_size} (GPU: {gpu_memory:.1f} GB)")
        else:
            args.batch_size = 32
            print(f"Batch size par défaut: {args.batch_size} (pas de GPU)")
    
    # Construire la commande d'entraînement
    cmd = [
        sys.executable,
        "train_audio.py",
        "--data-dir", args.data_dir,
        "--csv-dir", args.csv_dir,
        "--output-dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--num-workers", str(args.num_workers),
    ]
    
    # Ajouter les optimisations par défaut
    if not args.no_optimization:
        # Répertoire de cache pour les spectrogrammes
        spectrogram_cache = Path("data/spectrograms_cache")
        cmd.extend(["--spectrogram-dir", str(spectrogram_cache)])
        
        # Activer la prégénération si demandé
        if args.pregenerate:
            cmd.append("--pregenerate-spectrograms")
        
        # Utiliser les workers persistants
        cmd.append("--persistent-workers")
        
        # Prefetch factor optimisé
        cmd.extend(["--prefetch-factor", "4"])
        
        print("\n🚀 Optimisations activées:")
        print(f"  - Mixed Precision Training (AMP)")
        print(f"  - Workers persistants: {args.num_workers}")
        print(f"  - Prefetch factor: 4")
        print(f"  - Cache spectrogrammes: {spectrogram_cache}")
        if args.pregenerate:
            print(f"  - Prégénération des spectrogrammes")
    else:
        cmd.append("--no-amp")
        cmd.extend(["--num-workers", "4"])
        print("\n⚠️  Mode sans optimisation (pour comparaison)")
    
    print(f"\n📊 Configuration:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Epochs: {args.epochs}")
    
    # Afficher des conseils
    print("\n💡 Conseils pour accélérer encore plus l'entraînement:")
    if not args.pregenerate:
        print("  - Utilisez --pregenerate pour prégénérer les spectrogrammes (gain ~2-3x)")
    if args.batch_size < 64 and torch.cuda.is_available():
        print(f"  - Essayez un batch size plus grand (ex: --batch-size 64)")
    if args.num_workers < 8:
        print(f"  - Augmentez le nombre de workers si vous avez plus de CPU")
    
    # Exécuter la commande
    print(f"\n🎯 Lancement de l'entraînement...")
    print(f"Commande: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n⚠️  Entraînement interrompu par l'utilisateur")
        return 1
    
    print(f"\n✅ Entraînement terminé avec succès!")
    print(f"Modèle sauvegardé dans: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())