#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier une classe audio spécifique.
Utile pour identifier les fichiers problématiques.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import psutil
import torchaudio

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_memory(context: str):
    """Log l'utilisation mémoire actuelle."""
    memory_info = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    process_memory = process.memory_info().rss / (1024 ** 3)  # GB
    
    logger.info(f"[MEM {context}] Système: {memory_info.percent:.1f}% "
                f"({memory_info.used / (1024**3):.1f}/{memory_info.total / (1024**3):.1f} GB) | "
                f"Processus: {process_memory:.2f} GB")


def diagnose_class(class_dir: Path):
    """Diagnostique une classe audio."""
    logger.info(f"=== Diagnostic de {class_dir.name} ===")
    log_memory("Début")
    
    # 1. Lister les fichiers
    logger.info("\n1. Listage des fichiers...")
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        files = list(class_dir.glob(f"*{ext}"))
        if files:
            logger.info(f"  Format {ext}: {len(files)} fichiers")
            audio_files.extend(files)
    
    if not audio_files:
        logger.error("Aucun fichier audio trouvé!")
        return
    
    logger.info(f"  Total: {len(audio_files)} fichiers audio")
    log_memory("Après listage")
    
    # 2. Analyser les tailles de fichiers
    logger.info("\n2. Analyse des tailles de fichiers...")
    sizes = []
    for f in audio_files:
        try:
            size = f.stat().st_size / (1024 * 1024)  # MB
            sizes.append((size, f))
        except Exception as e:
            logger.error(f"  Erreur stat {f.name}: {e}")
    
    sizes.sort(reverse=True)
    logger.info(f"  Plus gros fichier: {sizes[0][1].name} ({sizes[0][0]:.2f} MB)")
    logger.info(f"  Plus petit fichier: {sizes[-1][1].name} ({sizes[-1][0]:.2f} MB)")
    logger.info(f"  Taille moyenne: {sum(s[0] for s in sizes) / len(sizes):.2f} MB")
    
    # Afficher les 5 plus gros
    logger.info("  Top 5 plus gros fichiers:")
    for size, f in sizes[:5]:
        logger.info(f"    {f.name}: {size:.2f} MB")
    
    log_memory("Après analyse tailles")
    
    # 3. Tester le chargement du premier fichier
    logger.info("\n3. Test de chargement du premier fichier...")
    test_file = audio_files[0]
    logger.info(f"  Fichier test: {test_file.name}")
    
    try:
        log_memory("Avant chargement")
        waveform, sr = torchaudio.load(str(test_file))
        logger.info(f"  ✓ Chargé avec succès: shape={waveform.shape}, sr={sr}")
        logger.info(f"    Durée: {waveform.shape[1]/sr:.2f} secondes")
        logger.info(f"    Canaux: {waveform.shape[0]}")
        
        # Tester la conversion mono
        if waveform.shape[0] > 1:
            waveform_mono = waveform.mean(dim=0, keepdim=True)
            logger.info(f"    Conversion mono: shape={waveform_mono.shape}")
            del waveform_mono
        
        del waveform
        log_memory("Après libération")
        
    except Exception as e:
        logger.error(f"  ✗ Erreur chargement: {type(e).__name__}: {e}")
    
    # 4. Tester le chargement du plus gros fichier
    if len(sizes) > 1:
        logger.info("\n4. Test de chargement du plus gros fichier...")
        biggest_file = sizes[0][1]
        logger.info(f"  Fichier: {biggest_file.name} ({sizes[0][0]:.2f} MB)")
        
        try:
            log_memory("Avant chargement gros fichier")
            waveform, sr = torchaudio.load(str(biggest_file))
            logger.info(f"  ✓ Chargé: shape={waveform.shape}, durée={waveform.shape[1]/sr:.2f}s")
            del waveform
            log_memory("Après libération gros fichier")
        except Exception as e:
            logger.error(f"  ✗ Erreur: {type(e).__name__}: {e}")
    
    # 5. Test rapide sur tous les fichiers
    logger.info("\n5. Test rapide sur tous les fichiers...")
    problematic = []
    
    for i, f in enumerate(audio_files):
        try:
            # Juste obtenir les infos sans charger
            info = torchaudio.info(str(f))
            if i < 5:  # Afficher les 5 premiers
                logger.info(f"  {f.name}: {info.sample_rate}Hz, {info.num_frames} frames")
        except Exception as e:
            problematic.append((f, str(e)))
            logger.warning(f"  ✗ {f.name}: {e}")
    
    if problematic:
        logger.warning(f"\n  {len(problematic)} fichiers problématiques détectés!")
        for f, err in problematic[:5]:
            logger.warning(f"    {f.name}: {err}")
    else:
        logger.info("  ✓ Tous les fichiers semblent valides")
    
    log_memory("Fin diagnostic")
    
    # Résumé
    logger.info("\n=== RÉSUMÉ ===")
    logger.info(f"Classe: {class_dir.name}")
    logger.info(f"Fichiers: {len(audio_files)}")
    logger.info(f"Taille totale: {sum(s[0] for s in sizes):.2f} MB")
    logger.info(f"Fichiers problématiques: {len(problematic)}")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic d'une classe audio")
    parser.add_argument('--audio-root', type=Path, required=True,
                       help='Répertoire racine des fichiers audio')
    parser.add_argument('--class-name', type=str, required=True,
                       help='Nom de la classe à diagnostiquer')
    
    args = parser.parse_args()
    
    class_dir = args.audio_root / args.class_name
    if not class_dir.exists() or not class_dir.is_dir():
        logger.error(f"Classe '{args.class_name}' non trouvée dans {args.audio_root}")
        return
    
    diagnose_class(class_dir)


if __name__ == "__main__":
    main()