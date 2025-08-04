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
from multiprocessing import Pool, cpu_count, Manager, Queue
import json
import time
from datetime import datetime, timedelta
import threading

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
_global_progress_dict = None
_global_start_time = None


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


def init_worker(model_path: str, training_db: str, device: str, output_dir: Path,
                window_size: float, hop_size: float, min_conf: float, energy_threshold: float,
                progress_dict: dict, start_time: float):
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
        progress_dict: Dictionnaire partagé pour la progression
        start_time: Temps de démarrage
    """
    global _global_analyzer, _global_output_dir, _global_window_size
    global _global_hop_size, _global_min_conf, _global_energy_threshold
    global _global_progress_dict, _global_start_time
    
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
    _global_progress_dict = progress_dict
    _global_start_time = start_time
    
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
        
        # Mettre à jour la progression
        _global_progress_dict['processed'] += 1
        _global_progress_dict['successful'] += 1
        _global_progress_dict['total_detections'] += len(detections)
        
        return {
            'status': 'success',
            'file': str(audio_file),
            'detections': len(detections),
            'class': class_name
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {audio_file}: {e}")
        
        # Mettre à jour la progression
        _global_progress_dict['processed'] += 1
        _global_progress_dict['errors'] += 1
        
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


def progress_monitor(progress_dict: dict, total_files: int, log_file: Optional[Path] = None, 
                    update_interval: int = 5):
    """
    Thread de monitoring qui affiche la progression périodiquement.
    
    Args:
        progress_dict: Dictionnaire partagé contenant la progression
        total_files: Nombre total de fichiers à traiter
        log_file: Fichier de log optionnel
        update_interval: Intervalle de mise à jour en secondes
    """
    start_time = progress_dict['start_time']
    
    while progress_dict['processed'] < total_files:
        time.sleep(update_interval)
        
        # Calculer les statistiques
        processed = progress_dict['processed']
        if processed == 0:
            continue
            
        elapsed = time.time() - start_time
        rate = processed / elapsed
        remaining = (total_files - processed) / rate if rate > 0 else 0
        
        # Créer le message de progression
        msg = (
            f"\n{'='*60}\n"
            f"PROGRESSION: {processed}/{total_files} fichiers ({processed/total_files*100:.1f}%)\n"
            f"Réussis: {progress_dict['successful']} | Erreurs: {progress_dict['errors']}\n"
            f"Détections totales: {progress_dict['total_detections']}\n"
            f"Vitesse: {rate:.1f} fichiers/sec\n"
            f"Temps écoulé: {timedelta(seconds=int(elapsed))}\n"
            f"Temps restant estimé: {timedelta(seconds=int(remaining))}\n"
            f"{'='*60}"
        )
        
        # Afficher et logger
        print(msg)
        
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")


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
        
        # Créer un manager pour partager des données entre processus
        manager = Manager()
        progress_dict = manager.dict({
            'processed': 0,
            'successful': 0,
            'errors': 0,
            'total_detections': 0,
            'start_time': time.time()
        })
        
        # Démarrer le thread de monitoring
        monitor_thread = threading.Thread(
            target=progress_monitor,
            args=(progress_dict, len(audio_files), args.log_file, args.progress_interval),
            daemon=True
        )
        monitor_thread.start()
        
        # Calculer la taille de chunk optimale
        chunksize = max(1, len(audio_files) // (args.threads * 10))
        logger.info(f"Chunksize: {chunksize}")
        
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
                args.energy_threshold,
                progress_dict,
                time.time()
            )
        ) as pool:
            # Préparer les arguments pour chaque fichier
            file_args = [(f, args.verbose) for f in audio_files]
            
            # Traiter les fichiers en parallèle (sans tqdm pour éviter les conflits)
            if args.verbose:
                logger.info("Traitement en cours... Surveillez la progression ci-dessous.")
            
            results = pool.starmap(process_single_file_mp, file_args, chunksize=chunksize)
    
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
        elapsed = time.time() - progress_dict['start_time']
        print(f"\nTemps total : {timedelta(seconds=int(elapsed))}")
        print(f"Vitesse moyenne : {len(audio_files)/elapsed:.2f} fichiers/sec")
    
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