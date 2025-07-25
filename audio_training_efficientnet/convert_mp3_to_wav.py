#!/usr/bin/env python3
"""
Script pour convertir les fichiers MP3 en WAV
Compatible avec les structures de dossiers de NightScan
"""

import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import subprocess
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_ffmpeg():
    """Vérifie si ffmpeg est installé."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, sample_rate: int = 22050) -> bool:
    """
    Convertit un fichier MP3 en WAV en utilisant ffmpeg.
    
    Args:
        mp3_path: Chemin du fichier MP3
        wav_path: Chemin de sortie pour le fichier WAV
        sample_rate: Taux d'échantillonnage cible (défaut: 22050 Hz)
        
    Returns:
        True si la conversion a réussi, False sinon
    """
    try:
        # Créer le répertoire parent si nécessaire
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Commande ffmpeg pour la conversion
        cmd = [
            'ffmpeg',
            '-i', str(mp3_path),
            '-acodec', 'pcm_s16le',  # Format PCM 16-bit
            '-ar', str(sample_rate),  # Taux d'échantillonnage
            '-ac', '1',  # Mono
            '-y',  # Écraser si le fichier existe
            str(wav_path)
        ]
        
        # Exécuter la conversion
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Erreur lors de la conversion de {mp3_path}: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Exception lors de la conversion de {mp3_path}: {e}")
        return False


def scan_mp3_files(input_dir: Path) -> List[Tuple[Path, str]]:
    """
    Scanne un répertoire pour trouver tous les fichiers MP3.
    
    Args:
        input_dir: Répertoire à scanner
        
    Returns:
        Liste de tuples (chemin_mp3, nom_classe)
    """
    mp3_files = []
    
    for class_dir in input_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        
        # Ignorer les dossiers cachés
        if class_name.startswith('.'):
            continue
            
        # Trouver tous les fichiers MP3 dans ce dossier (insensible à la casse)
        for mp3_file in class_dir.rglob('*.[mM][pP]3'):
            mp3_files.append((mp3_file, class_name))
            
    return mp3_files


def convert_directory(input_dir: Path, output_dir: Path, sample_rate: int = 22050) -> Tuple[int, int]:
    """
    Convertit tous les fichiers MP3 d'un répertoire en WAV.
    
    Args:
        input_dir: Répertoire contenant les fichiers MP3
        output_dir: Répertoire de sortie pour les fichiers WAV
        sample_rate: Taux d'échantillonnage cible
        
    Returns:
        Tuple (nombre de conversions réussies, nombre total de fichiers)
    """
    # Scanner les fichiers MP3
    mp3_files = scan_mp3_files(input_dir)
    
    if not mp3_files:
        logger.info("Aucun fichier MP3 trouvé.")
        return 0, 0
        
    logger.info(f"Trouvé {len(mp3_files)} fichiers MP3 à convertir")
    
    # Compter les conversions par classe
    class_counts = {}
    for _, class_name in mp3_files:
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    logger.info("Distribution par classe:")
    for class_name, count in sorted(class_counts.items()):
        logger.info(f"  {class_name}: {count} fichiers")
    
    # Convertir les fichiers
    success_count = 0
    
    with tqdm(total=len(mp3_files), desc="Conversion MP3 → WAV") as pbar:
        for mp3_path, class_name in mp3_files:
            # Construire le chemin de sortie
            relative_path = mp3_path.relative_to(input_dir)
            wav_path = output_dir / relative_path.with_suffix('.wav')
            
            # Convertir le fichier
            if convert_mp3_to_wav(mp3_path, wav_path, sample_rate):
                success_count += 1
            
            pbar.update(1)
            pbar.set_postfix({'réussi': success_count, 'échoué': pbar.n - success_count})
    
    return success_count, len(mp3_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convertit les fichiers MP3 en WAV pour l'entraînement NightScan"
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help="Répertoire contenant les fichiers MP3 organisés par classe"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help="Répertoire de sortie (défaut: input_dir avec '_wav' ajouté)"
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help="Taux d'échantillonnage cible en Hz (défaut: 22050)"
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help="Convertir les fichiers dans le même répertoire (garde les MP3)"
    )
    
    args = parser.parse_args()
    
    # Vérifier que ffmpeg est installé
    if not check_ffmpeg():
        print("❌ Erreur: ffmpeg n'est pas installé!")
        print("Installez-le avec: brew install ffmpeg")
        return 1
    
    # Vérifier que le répertoire d'entrée existe
    if not args.input_dir.exists():
        print(f"❌ Erreur: Le répertoire {args.input_dir} n'existe pas!")
        return 1
    
    # Déterminer le répertoire de sortie
    if args.in_place:
        output_dir = args.input_dir
        logger.info("Mode in-place: les fichiers WAV seront créés à côté des MP3")
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.input_dir.parent / f"{args.input_dir.name}_wav"
    
    logger.info(f"Répertoire d'entrée: {args.input_dir}")
    logger.info(f"Répertoire de sortie: {output_dir}")
    logger.info(f"Taux d'échantillonnage: {args.sample_rate} Hz")
    
    # Effectuer la conversion
    success, total = convert_directory(args.input_dir, output_dir, args.sample_rate)
    
    # Afficher le résumé
    print(f"\n✅ Conversion terminée!")
    print(f"Fichiers convertis avec succès: {success}/{total}")
    
    if success < total:
        print(f"⚠️  {total - success} fichiers ont échoué")
        return 1
    
    if not args.in_place:
        print(f"\nLes fichiers WAV ont été sauvegardés dans: {output_dir}")
        print(f"\nProchaines étapes:")
        print(f"1. Segmenter les nouveaux fichiers WAV:")
        print(f"   python prepare_audio_data.py {output_dir} --segment")
        print(f"2. Fusionner avec les données existantes si nécessaire")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())