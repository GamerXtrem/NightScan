#!/usr/bin/env python3
"""
Script principal pour le workflow NightScan en 2 passes (comme BirdNET).
Combine l'analyse et l'extraction des segments.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_analysis(audio_input: Path, 
                output: Path,
                model: str,
                training_db: str,
                min_conf: float = 0.25,
                seg_length: float = 3.0,
                threads: int = 1,
                device: str = None,
                verbose: bool = False,
                limit_files: int = None,
                single_file: Path = None) -> bool:
    """
    Exécute la première passe : analyse des fichiers audio.
    
    Returns:
        True si succès, False sinon
    """
    logger.info("=== PASSE 1: Analyse des fichiers audio ===")
    
    cmd = [
        sys.executable,
        "analyze_audio_files.py",
        "--audio-input", str(audio_input),
        "--output", str(output / "results"),
        "--model", model,
        "--training-db", training_db,
        "--min-conf", str(min_conf),
        "--seg-length", str(seg_length),
        "--threads", str(threads)
    ]
    
    if device:
        cmd.extend(["--device", device])
    if verbose:
        cmd.append("--verbose")
    if limit_files:
        cmd.extend(["--limit-files", str(limit_files)])
    if single_file:
        cmd.extend(["--single-file", str(single_file)])
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'analyse : {e}")
        return False


def run_extraction(audio_input: Path,
                  results: Path,
                  output: Path,
                  max_segments: int = 500,
                  threads: int = 1,
                  verbose: bool = False,
                  dry_run: bool = False) -> bool:
    """
    Exécute la deuxième passe : extraction des segments.
    
    Returns:
        True si succès, False sinon
    """
    logger.info("=== PASSE 2: Extraction des segments ===")
    
    cmd = [
        sys.executable,
        "extract_segments_from_results.py",
        "--audio-input", str(audio_input),
        "--results", str(results),
        "--output", str(output / "segments"),
        "--max-segments", str(max_segments),
        "--threads", str(threads)
    ]
    
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'extraction : {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="NightScan Segments - Workflow en 2 passes comme BirdNET"
    )
    
    # Commandes
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande "all" - Exécute les deux passes
    all_parser = subparsers.add_parser('all', help='Exécute l\'analyse et l\'extraction')
    all_parser.add_argument('--audio-input', type=Path, required=True,
                           help='Répertoire contenant les fichiers audio')
    all_parser.add_argument('--output', type=Path, required=True,
                           help='Répertoire de sortie principal')
    all_parser.add_argument('--model', type=str, required=True,
                           help='Chemin vers le modèle (.pth)')
    all_parser.add_argument('--training-db', type=str, required=True,
                           help='Base SQLite d\'entraînement')
    all_parser.add_argument('--min-conf', type=float, default=0.25,
                           help='Confiance minimale (défaut: 0.25)')
    all_parser.add_argument('--max-segments', type=int, default=500,
                           help='Nombre max de segments par fichier (défaut: 500)')
    all_parser.add_argument('--seg-length', type=float, default=3.0,
                           help='Longueur des segments en secondes (défaut: 3.0)')
    all_parser.add_argument('--threads', type=int, default=1,
                           help='Nombre de threads CPU (défaut: 1)')
    all_parser.add_argument('--device', type=str, default=None,
                           help='Device (cuda/cpu, auto si non spécifié)')
    all_parser.add_argument('--verbose', action='store_true',
                           help='Mode verbose - affiche les détails')
    all_parser.add_argument('--dry-run', action='store_true',
                           help='Mode simulation - ne pas extraire les fichiers')
    all_parser.add_argument('--limit-files', type=int, default=None,
                           help='Limiter aux N premiers fichiers')
    all_parser.add_argument('--single-file', type=Path, default=None,
                           help='Traiter un seul fichier spécifique')
    
    # Commande "analyze" - Analyse seulement
    analyze_parser = subparsers.add_parser('analyze', help='Analyse seulement (passe 1)')
    analyze_parser.add_argument('--audio-input', type=Path, required=True,
                               help='Répertoire contenant les fichiers audio')
    analyze_parser.add_argument('--output', type=Path, required=True,
                               help='Répertoire de sortie pour les résultats')
    analyze_parser.add_argument('--model', type=str, required=True,
                               help='Chemin vers le modèle (.pth)')
    analyze_parser.add_argument('--training-db', type=str, required=True,
                               help='Base SQLite d\'entraînement')
    analyze_parser.add_argument('--min-conf', type=float, default=0.25,
                               help='Confiance minimale (défaut: 0.25)')
    analyze_parser.add_argument('--seg-length', type=float, default=3.0,
                               help='Longueur des segments en secondes (défaut: 3.0)')
    analyze_parser.add_argument('--threads', type=int, default=1,
                               help='Nombre de threads CPU (défaut: 1)')
    analyze_parser.add_argument('--device', type=str, default=None,
                               help='Device (cuda/cpu, auto si non spécifié)')
    analyze_parser.add_argument('--verbose', action='store_true',
                               help='Mode verbose - affiche les détails')
    analyze_parser.add_argument('--limit-files', type=int, default=None,
                               help='Limiter aux N premiers fichiers')
    analyze_parser.add_argument('--single-file', type=Path, default=None,
                               help='Traiter un seul fichier spécifique')
    
    # Commande "extract" - Extraction seulement
    extract_parser = subparsers.add_parser('extract', help='Extraction seulement (passe 2)')
    extract_parser.add_argument('--audio-input', type=Path, required=True,
                               help='Répertoire contenant les fichiers audio originaux')
    extract_parser.add_argument('--results', type=Path, required=True,
                               help='Répertoire contenant les résultats CSV')
    extract_parser.add_argument('--output', type=Path, required=True,
                               help='Répertoire de sortie pour les segments')
    extract_parser.add_argument('--max-segments', type=int, default=500,
                               help='Nombre max de segments par fichier (défaut: 500)')
    extract_parser.add_argument('--threads', type=int, default=1,
                               help='Nombre de threads CPU (défaut: 1)')
    extract_parser.add_argument('--verbose', action='store_true',
                               help='Mode verbose - affiche les détails')
    extract_parser.add_argument('--dry-run', action='store_true',
                               help='Mode simulation - ne pas extraire les fichiers')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Exécuter la commande appropriée
    if args.command == 'all':
        # Créer le répertoire de sortie
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Passe 1: Analyse
        success = run_analysis(
            args.audio_input,
            args.output,
            args.model,
            args.training_db,
            args.min_conf,
            args.seg_length,
            args.threads,
            args.device,
            args.verbose,
            args.limit_files,
            args.single_file
        )
        
        if not success:
            logger.error("L'analyse a échoué. Arrêt du processus.")
            sys.exit(1)
        
        # Passe 2: Extraction
        results_dir = args.output / "results"
        success = run_extraction(
            args.audio_input,
            results_dir,
            args.output,
            args.max_segments,
            args.threads,
            args.verbose,
            args.dry_run
        )
        
        if success:
            logger.info("✅ Workflow terminé avec succès!")
            logger.info(f"Segments extraits dans : {args.output / 'segments'}")
        else:
            logger.error("L'extraction a échoué.")
            sys.exit(1)
            
    elif args.command == 'analyze':
        # Analyse seulement
        args.output.mkdir(parents=True, exist_ok=True)
        success = run_analysis(
            args.audio_input,
            args.output,
            args.model,
            args.training_db,
            args.min_conf,
            args.seg_length,
            args.threads,
            args.device,
            args.verbose,
            args.limit_files,
            args.single_file
        )
        
        if success:
            logger.info("✅ Analyse terminée avec succès!")
            logger.info(f"Résultats dans : {args.output}")
        else:
            sys.exit(1)
            
    elif args.command == 'extract':
        # Extraction seulement
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Appeler directement le script d'extraction
        cmd = [
            sys.executable,
            "extract_segments_from_results.py",
            "--audio-input", str(args.audio_input),
            "--results", str(args.results),
            "--output", str(args.output),
            "--max-segments", str(args.max_segments),
            "--threads", str(args.threads)
        ]
        
        if args.verbose:
            cmd.append("--verbose")
        if args.dry_run:
            cmd.append("--dry-run")
        
        try:
            subprocess.run(cmd, check=True)
            logger.info("✅ Extraction terminée avec succès!")
            logger.info(f"Segments dans : {args.output}")
        except subprocess.CalledProcessError:
            sys.exit(1)


if __name__ == "__main__":
    main()