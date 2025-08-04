#!/usr/bin/env python3
"""
Première passe : Analyse des fichiers audio pour générer des fichiers de résultats.
Basé sur l'approche BirdNET - détection uniquement, l'extraction se fait séparément.
"""

import os
import sys
import torch
import argparse
import csv
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager, Queue, Value, Lock
import json
import time
from datetime import datetime, timedelta
import threading
import queue
import signal

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_config import create_audio_model
from selection_detection_audio_data import ModelBasedDetector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variables globales pour le multiprocessing
_global_analyzer = None
_global_output_dir = None
_global_window_size = None
_global_hop_size = None
_global_min_conf = None
_global_energy_threshold = None


class AudioAnalyzer:
    """Analyseur audio basé sur le modèle ML pour générer des résultats de détection."""
    
    def __init__(self, 
                 model_path: str,
                 training_db: str,
                 device: Optional[str] = None,
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 fmin: int = 50,
                 fmax: int = 11000):
        """
        Initialise l'analyseur.
        
        Args:
            model_path: Chemin vers le checkpoint du modèle
            training_db: Base SQLite utilisée pendant l'entraînement
            device: Device à utiliser (auto-détection si None)
            sample_rate: Taux d'échantillonnage (22050 par défaut)
            n_fft: Taille FFT (2048 par défaut)
            hop_length: Hop length pour le spectrogramme (512 par défaut)
            n_mels: Nombre de bandes mel (128 par défaut)
            fmin: Fréquence minimale (50 par défaut)
            fmax: Fréquence maximale (11000 par défaut)
        """
        # Créer le détecteur avec les bons paramètres
        self.detector = ModelBasedDetector(
            model_path=model_path,
            device=device,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            training_db=training_db
        )
        
        logger.info(f"Analyseur initialisé avec {self.detector.num_classes} classes")
    
    def analyze_file(self, 
                    audio_path: Path,
                    window_size: float = 3.0,
                    hop_size: float = 1.5,
                    min_confidence: float = 0.25,
                    energy_threshold: float = -50.0) -> List[Dict]:
        """
        Analyse un fichier audio et retourne toutes les détections.
        
        Args:
            audio_path: Chemin du fichier audio
            window_size: Taille de la fenêtre en secondes (3.0 par défaut)
            hop_size: Décalage entre fenêtres en secondes (1.5 par défaut)
            min_confidence: Confiance minimale (0.25 par défaut, comme BirdNET)
            energy_threshold: Seuil d'énergie en dB
            
        Returns:
            Liste des détections avec toutes les informations
        """
        # Utiliser la méthode existante du détecteur
        detections = self.detector.process_file(
            audio_path=audio_path,
            window_size=window_size,
            hop_size=hop_size,
            min_confidence=min_confidence,
            target_classes=None,  # Toutes les classes
            energy_threshold=energy_threshold
        )
        
        return detections
    
    def save_results(self, detections: List[Dict], output_path: Path, audio_file: Path):
        """
        Sauvegarde les résultats de détection dans un fichier CSV.
        Format compatible avec BirdNET.
        
        Args:
            detections: Liste des détections
            output_path: Chemin du fichier de sortie CSV
            audio_file: Fichier audio source
        """
        # Créer le répertoire si nécessaire
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # En-tête compatible BirdNET
            writer.writerow(['Start (s)', 'End (s)', 'Scientific name', 'Common name', 'Confidence'])
            
            # Écrire les détections
            for detection in detections:
                writer.writerow([
                    f"{detection['start_time']:.1f}",
                    f"{detection['end_time']:.1f}",
                    detection['class'],  # Nom scientifique
                    detection['class'],  # Nom commun (identique pour l'instant)
                    f"{detection['confidence']:.4f}"
                ])
        
        logger.debug(f"Résultats sauvegardés : {output_path} ({len(detections)} détections)")


# Classes et fonctions supprimées pour simplification
# L'ancien système avec ProgressTracker et result_collector a été enlevé


def init_worker(model_path: str, training_db: str, device: str, output_dir: Path,
                window_size: float, hop_size: float, min_conf: float, energy_threshold: float):
    """
    Initialise un worker pour le multiprocessing.
    Charge le modèle une seule fois par processus.
    
    Args:
        model_path: Chemin vers le modèle
        training_db: Base SQLite d'entraînement
        device: Device à utiliser
        output_dir: Répertoire de sortie
        window_size: Taille de fenêtre
        hop_size: Décalage entre fenêtres
        min_conf: Confiance minimale
        energy_threshold: Seuil d'énergie
    """
    global _global_analyzer, _global_output_dir, _global_window_size
    global _global_hop_size, _global_min_conf, _global_energy_threshold
    
    # Créer l'analyseur une seule fois
    _global_analyzer = AudioAnalyzer(
        model_path=model_path,
        training_db=training_db,
        device=device
    )
    
    # Stocker les paramètres
    _global_output_dir = output_dir
    _global_window_size = window_size
    _global_hop_size = hop_size
    _global_min_conf = min_conf
    _global_energy_threshold = energy_threshold
    
    logger.info(f"Worker initialisé (PID: {os.getpid()})")


def process_single_file_mp(audio_file: Path, verbose: bool = False) -> Dict:
    """
    Version multiprocessing de process_single_file.
    Utilise les variables globales initialisées par init_worker.
    
    Args:
        audio_file: Fichier audio à traiter
        verbose: Mode verbose
        
    Returns:
        Dict avec les statistiques du traitement
    """
    try:
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyse de : {audio_file}")
            logger.info(f"Classe : {audio_file.parent.name}")
            logger.info(f"Paramètres : fenêtre={_global_window_size}s, hop={_global_hop_size}s, min_conf={_global_min_conf}")
        
        # Analyser le fichier avec l'analyzer global
        detections = _global_analyzer.analyze_file(
            audio_path=audio_file,
            window_size=_global_window_size,
            hop_size=_global_hop_size,
            min_confidence=_global_min_conf,
            energy_threshold=_global_energy_threshold
        )
        
        if verbose and detections:
            logger.info(f"\nDétections trouvées : {len(detections)}")
            # Afficher les top 5 détections
            sorted_detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
            for i, det in enumerate(sorted_detections[:5]):
                logger.info(f"  [{i+1}] {det['start_time']:.1f}-{det['end_time']:.1f}s : "
                          f"{det['class']} (conf: {det['confidence']:.3f})")
            if len(detections) > 5:
                logger.info(f"  ... et {len(detections)-5} autres détections")
            
            # Statistiques par classe
            class_counts = {}
            for det in detections:
                class_counts[det['class']] = class_counts.get(det['class'], 0) + 1
            logger.info(f"\nRépartition par espèce détectée :")
            for species, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {species}: {count}")
        
        # Générer le nom du fichier de résultats
        class_name = audio_file.parent.name
        output_subdir = _global_output_dir / class_name
        output_file = output_subdir / f"{audio_file.stem}.csv"
        
        # Sauvegarder les résultats
        _global_analyzer.save_results(detections, output_file, audio_file)
        
        return {
            'status': 'success',
            'file': str(audio_file),
            'detections': len(detections),
            'class': class_name
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {audio_file}: {e}")
        return {
            'status': 'error',
            'file': str(audio_file),
            'error': str(e)
        }


def process_single_file(args: Tuple[Path, AudioAnalyzer, Path, float, float, float, float, bool]) -> Dict:
    """
    Traite un seul fichier audio (pour multiprocessing).
    
    Args:
        args: Tuple contenant (audio_file, analyzer, output_dir, window_size, hop_size, min_conf, energy_threshold, verbose)
        
    Returns:
        Dict avec les statistiques du traitement
    """
    audio_file, analyzer, output_dir, window_size, hop_size, min_conf, energy_threshold, verbose = args
    
    try:
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Analyse de : {audio_file}")
            logger.info(f"Classe : {audio_file.parent.name}")
            logger.info(f"Paramètres : fenêtre={window_size}s, hop={hop_size}s, min_conf={min_conf}")
        
        # Analyser le fichier
        detections = analyzer.analyze_file(
            audio_path=audio_file,
            window_size=window_size,
            hop_size=hop_size,
            min_confidence=min_conf,
            energy_threshold=energy_threshold
        )
        
        if verbose and detections:
            logger.info(f"\nDétections trouvées : {len(detections)}")
            # Afficher les top 5 détections
            sorted_detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
            for i, det in enumerate(sorted_detections[:5]):
                logger.info(f"  [{i+1}] {det['start_time']:.1f}-{det['end_time']:.1f}s : "
                          f"{det['class']} (conf: {det['confidence']:.3f})")
            if len(detections) > 5:
                logger.info(f"  ... et {len(detections)-5} autres détections")
            
            # Statistiques par classe
            class_counts = {}
            for det in detections:
                class_counts[det['class']] = class_counts.get(det['class'], 0) + 1
            logger.info(f"\nRépartition par espèce détectée :")
            for species, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {species}: {count}")
        
        # Générer le nom du fichier de résultats
        # Structure : output_dir/class_name/audio_file_name.csv
        class_name = audio_file.parent.name
        output_subdir = output_dir / class_name
        output_file = output_subdir / f"{audio_file.stem}.csv"
        
        # Sauvegarder les résultats
        analyzer.save_results(detections, output_file, audio_file)
        
        return {
            'status': 'success',
            'file': str(audio_file),
            'detections': len(detections),
            'class': class_name
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {audio_file}: {e}")
        return {
            'status': 'error',
            'file': str(audio_file),
            'error': str(e)
        }


def simple_progress_monitor(output_dir: Path, total_files: int, log_file: Optional[Path] = None,
                           update_interval: int = 5, stop_event: threading.Event = None):
    """
    Moniteur de progression simple qui compte les fichiers CSV créés.
    
    Args:
        output_dir: Répertoire de sortie où sont créés les CSV
        total_files: Nombre total de fichiers à traiter  
        log_file: Fichier de log optionnel
        update_interval: Intervalle de mise à jour en secondes
        stop_event: Event pour arrêter le thread
    """
    start_time = time.time()
    last_count = 0
    last_time = start_time
    
    while not stop_event.is_set():
        time.sleep(update_interval)
        
        # Compter les fichiers CSV créés
        csv_files = list(output_dir.rglob('*.csv'))
        processed = len([f for f in csv_files if f.name != 'analysis_summary.json'])
        
        if processed == 0:
            continue
            
        # Calculer les statistiques
        current_time = time.time()
        elapsed = current_time - start_time
        instant_rate = (processed - last_count) / (current_time - last_time) if current_time > last_time else 0
        avg_rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total_files - processed) / avg_rate if avg_rate > 0 else 0
        
        last_count = processed
        last_time = current_time
        
        # Créer le message
        msg = (
            f"\n{'='*60}\n"
            f"PROGRESSION: {processed}/{total_files} fichiers ({processed/total_files*100:.1f}%)\n"
            f"Vitesse: {avg_rate:.1f} fichiers/sec (instant: {instant_rate:.1f})\n"
            f"Temps écoulé: {timedelta(seconds=int(elapsed))}\n"
            f"Temps restant estimé: {timedelta(seconds=int(remaining))}\n"
            f"{'='*60}"
        )
        
        print(msg, flush=True)
        
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")
                
        if processed >= total_files:
            break


def parse_folders(audio_dir: Path, extensions: List[str] = ['.wav', '.mp3', '.flac']) -> List[Path]:
    """
    Parse le dossier audio pour trouver tous les fichiers à analyser.
    
    Args:
        audio_dir: Répertoire contenant les fichiers audio
        extensions: Extensions de fichiers à traiter
        
    Returns:
        Liste des chemins de fichiers audio
    """
    audio_files = []
    
    for ext in extensions:
        files = list(audio_dir.rglob(f'*{ext}'))
        # Filtrer les fichiers cachés macOS
        files = [f for f in files if not f.name.startswith('._')]
        audio_files.extend(files)
    
    logger.info(f"Trouvé {len(audio_files)} fichiers audio dans {audio_dir}")
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse des fichiers audio pour générer des résultats de détection (Passe 1)"
    )
    
    # Entrées/Sorties
    parser.add_argument('--audio-input', type=Path, required=True,
                       help="Répertoire contenant les fichiers audio")
    parser.add_argument('--output', type=Path, required=True,
                       help="Répertoire de sortie pour les fichiers de résultats")
    
    # Modèle
    parser.add_argument('--model', type=str, required=True,
                       help="Chemin vers le checkpoint du modèle (.pth)")
    parser.add_argument('--training-db', type=str, required=True,
                       help="Base SQLite utilisée pendant l'entraînement")
    parser.add_argument('--device', type=str, default=None,
                       help="Device à utiliser (cuda/cpu, auto si non spécifié)")
    
    # Paramètres de détection
    parser.add_argument('--min-conf', type=float, default=0.25,
                       help="Confiance minimale (défaut: 0.25, comme BirdNET)")
    parser.add_argument('--energy-threshold', type=float, default=-50.0,
                       help="Seuil d'énergie en dB (défaut: -50)")
    
    # Paramètres de fenêtrage
    parser.add_argument('--seg-length', type=float, default=3.0,
                       help="Taille de la fenêtre/segment en secondes (défaut: 3.0)")
    parser.add_argument('--hop-size', type=float, default=1.5,
                       help="Décalage entre fenêtres en secondes (défaut: 1.5)")
    
    # Paramètres de performance
    parser.add_argument('--threads', type=int, default=1,
                       help="Nombre de threads CPU (défaut: 1)")
    
    # Options
    parser.add_argument('--extensions', nargs='+', default=['.wav', '.mp3', '.flac'],
                       help="Extensions de fichiers à traiter")
    parser.add_argument('--verbose', action='store_true',
                       help="Mode verbose - affiche les détails de chaque détection")
    parser.add_argument('--limit-files', type=int, default=None,
                       help="Limiter le traitement aux N premiers fichiers (pour debug)")
    parser.add_argument('--single-file', type=Path, default=None,
                       help="Traiter un seul fichier spécifique en mode détaillé")
    parser.add_argument('--log-file', type=Path, default=None,
                       help='Fichier de log pour la progression (utile en multiprocessing)')
    parser.add_argument('--progress-interval', type=int, default=10,
                       help='Intervalle de mise à jour de la progression en secondes (défaut: 10)')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Scanner les fichiers audio
    audio_files = parse_folders(args.audio_input, args.extensions)
    
    if not audio_files:
        logger.error("Aucun fichier audio trouvé!")
        return
    
    # Créer l'analyseur uniquement pour le mode single-thread
    analyzer = None
    if args.threads < 2:
        analyzer = AudioAnalyzer(
            model_path=args.model,
            training_db=args.training_db,
            device=args.device
        )
    
    # Statistiques globales
    total_detections = 0
    successful_files = 0
    error_files = 0
    detections_by_class = {}
    
    # Filtrer si single-file ou limit-files
    if args.single_file:
        if args.single_file in audio_files:
            audio_files = [args.single_file]
            logger.info(f"Mode single-file : traitement de {args.single_file}")
        else:
            logger.error(f"Fichier non trouvé : {args.single_file}")
            return
    elif args.limit_files:
        audio_files = audio_files[:args.limit_files]
        logger.info(f"Limitation à {args.limit_files} fichiers")
    
    logger.info(f"Début de l'analyse de {len(audio_files)} fichiers...")
    
    if args.threads < 2:
        # Mode single-thread
        results = []
        for audio_file in tqdm(audio_files, desc="Analyse des fichiers", disable=args.verbose):
            result = process_single_file((
                audio_file, analyzer, args.output, args.seg_length, 
                args.hop_size, args.min_conf, args.energy_threshold, args.verbose
            ))
            results.append(result)
    else:
        # Mode multi-thread avec Pool
        logger.info(f"Utilisation du multiprocessing avec {args.threads} threads")
        
        # Event pour arrêter le monitoring
        stop_event = threading.Event()
        
        # Démarrer le thread de monitoring simple
        monitor_thread = threading.Thread(
            target=simple_progress_monitor,
            args=(args.output, len(audio_files), args.log_file, args.progress_interval, stop_event),
            daemon=True
        )
        monitor_thread.start()
        
        # Calculer la taille de chunk optimale
        chunksize = max(1, len(audio_files) // (args.threads * 10))
        logger.info(f"Chunksize: {chunksize}")
        
        try:
            # Créer le pool avec initializer
            with Pool(
                processes=args.threads,
                initializer=init_worker,
                initargs=(
                    args.model,
                    args.training_db,
                    args.device,
                    args.output,
                    args.seg_length,
                    args.hop_size,
                    args.min_conf,
                    args.energy_threshold
                )
            ) as pool:
                # Utiliser imap_unordered pour un traitement asynchrone
                results = []
                
                # Créer un générateur pour les arguments
                def arg_generator():
                    for f in audio_files:
                        yield (f, args.verbose)
                
                # Traiter de manière asynchrone
                logger.info("Traitement en cours...")
                
                for result in pool.imap_unordered(
                    lambda args: process_single_file_mp(*args),
                    arg_generator(),
                    chunksize=chunksize
                ):
                    results.append(result)
                    
        finally:
            # Arrêter le monitoring
            stop_event.set()
            monitor_thread.join(timeout=2)
    
    # Compiler les statistiques
    for result in results:
        if result['status'] == 'success':
            successful_files += 1
            total_detections += result['detections']
            class_name = result.get('class', 'unknown')
            if class_name not in detections_by_class:
                detections_by_class[class_name] = 0
            detections_by_class[class_name] += result['detections']
        else:
            error_files += 1
    
    # Sauvegarder un résumé global
    summary_path = args.output / 'analysis_summary.json'
    summary = {
        'total_files': len(audio_files),
        'successful_files': successful_files,
        'error_files': error_files,
        'total_detections': total_detections,
        'detections_by_class': detections_by_class,
        'parameters': {
            'min_confidence': args.min_conf,
            'segment_length': args.seg_length,
            'hop_size': args.hop_size,
            'energy_threshold': args.energy_threshold
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Afficher le résumé
    # Afficher un résumé final plus détaillé
    print(f"\n{'='*60}")
    print("RÉSUMÉ FINAL DE L'ANALYSE")
    print(f"{'='*60}")
    print(f"Fichiers analysés : {successful_files}/{len(audio_files)}")
    print(f"Erreurs : {error_files}")
    print(f"Total détections : {total_detections}")
    
    if args.threads >= 2:
        # Calculer le temps écoulé depuis le début
        # Note: on n'a plus progress_tracker, mais on peut estimer
        csv_files = list(args.output.rglob('*.csv'))
        actual_processed = len([f for f in csv_files if f.name != 'analysis_summary.json'])
        if actual_processed > 0:
            print(f"\nFichiers réellement traités : {actual_processed}")
    
    if total_detections > 0:
        print(f"Moyenne détections/fichier : {total_detections/successful_files:.1f}")
        print(f"\nDétections par classe :")
        # Trier par nombre de détections
        sorted_classes = sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes[:10]:  # Top 10
            percentage = (count / total_detections) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        if len(sorted_classes) > 10:
            print(f"  ... et {len(sorted_classes)-10} autres classes")
    
    print(f"\nRésultats sauvegardés dans : {args.output}")
    print(f"Résumé : {summary_path}")
    
    if args.verbose:
        print(f"\nParamètres utilisés :")
        print(f"  Confiance minimale : {args.min_conf}")
        print(f"  Longueur segments : {args.seg_length}s")
        print(f"  Hop size : {args.hop_size}s")
        print(f"  Seuil énergie : {args.energy_threshold}dB")


if __name__ == "__main__":
    main()