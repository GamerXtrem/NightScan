#!/usr/bin/env python3
"""
Script de test pour vérifier que la génération du cache fonctionne correctement.
"""

import sys
import subprocess
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cache_generation():
    """Test la génération du cache avec un petit ensemble de données."""
    
    # Définir les chemins
    csv_dir = Path("data/processed/csv")
    audio_dir = Path("data/audio")  # À ajuster selon votre structure
    cache_dir = Path("data/test_spectrograms_cache")
    
    # Nettoyer le cache de test s'il existe
    if cache_dir.exists():
        logger.info(f"Suppression du cache existant: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # Commande pour générer le cache (seulement val pour tester rapidement)
    cmd = [
        sys.executable,
        "pregenerate_spectrograms.py",
        "--csv-dir", str(csv_dir),
        "--audio-dir", str(audio_dir),
        "--output-dir", str(cache_dir),
        "--splits", "val",
        "--skip-augmented"  # Pour accélérer le test
    ]
    
    logger.info(f"Exécution de la commande: {' '.join(cmd)}")
    
    try:
        # Exécuter la commande
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Sortie standard:")
        logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("Sortie d'erreur:")
            logger.warning(result.stderr)
        
        # Vérifier les résultats
        if cache_dir.exists():
            npy_files = list(cache_dir.rglob("*.npy"))
            logger.info(f"\n✅ Test réussi!")
            logger.info(f"Nombre de fichiers .npy générés: {len(npy_files)}")
            
            if npy_files:
                logger.info("\nExemples de fichiers générés:")
                for f in npy_files[:10]:
                    size_mb = f.stat().st_size / 1024 / 1024
                    logger.info(f"  - {f.relative_to(cache_dir)} ({size_mb:.2f} MB)")
            else:
                logger.error("❌ Aucun fichier .npy généré!")
                return False
        else:
            logger.error(f"❌ Le répertoire de cache n'a pas été créé: {cache_dir}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur lors de l'exécution: {e}")
        logger.error(f"Sortie standard: {e.stdout}")
        logger.error(f"Sortie d'erreur: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    # Vérifier que nous sommes dans le bon répertoire
    if not Path("pregenerate_spectrograms.py").exists():
        logger.error("Ce script doit être exécuté depuis le répertoire audio_training_efficientnet/")
        sys.exit(1)
    
    success = test_cache_generation()
    sys.exit(0 if success else 1)