#!/usr/bin/env python3
"""
Deuxième passe : Extraction des segments audio basée sur les fichiers de résultats.
Basé sur l'approche BirdNET - lit les résultats CSV et extrait les meilleurs segments.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
import librosa
import soundfile as sf
from collections import defaultdict
import numpy as np

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Detection:
    """Représente une détection avec toutes ses métadonnées."""
    
    def __init__(self, start_time: float, end_time: float, species: str, 
                 confidence: float, audio_file: Path, result_file: Path):
        self.start_time = start_time
        self.end_time = end_time
        self.species = species
        self.confidence = confidence
        self.audio_file = audio_file
        self.result_file = result_file
        self.duration = end_time - start_time
    
    def __repr__(self):
        return f"Detection({self.species}, {self.confidence:.2f}, {self.start_time:.1f}-{self.end_time:.1f}s)"


def parse_result_file(result_file: Path, audio_root: Path) -> List[Detection]:
    """
    Parse un fichier de résultats CSV et retourne les détections.
    
    Args:
        result_file: Chemin du fichier CSV de résultats
        audio_root: Répertoire racine des fichiers audio
        
    Returns:
        Liste des détections
    """
    detections = []
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Reconstruire le chemin du fichier audio
            # Structure attendue : results/class_name/audio_file.csv
            class_name = result_file.parent.name
            audio_filename = result_file.stem + '.wav'  # Assumer .wav par défaut
            audio_file = audio_root / class_name / audio_filename
            
            # Vérifier d'autres extensions si .wav n'existe pas
            if not audio_file.exists():
                for ext in ['.mp3', '.flac', '.ogg']:
                    alt_file = audio_root / class_name / (result_file.stem + ext)
                    if alt_file.exists():
                        audio_file = alt_file
                        break
            
            for row in reader:
                detection = Detection(
                    start_time=float(row['Start (s)']),
                    end_time=float(row['End (s)']),
                    species=row['Scientific name'],
                    confidence=float(row['Confidence']),
                    audio_file=audio_file,
                    result_file=result_file
                )
                detections.append(detection)
                
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de {result_file}: {e}")
    
    return detections


def parse_folders(audio_root: Path, results_dir: Path) -> Dict[Path, List[Detection]]:
    """
    Parse tous les fichiers de résultats et groupe les détections par fichier audio.
    
    Args:
        audio_root: Répertoire racine des fichiers audio
        results_dir: Répertoire contenant les fichiers de résultats
        
    Returns:
        Dict mappant chaque fichier audio à ses détections
    """
    all_detections = defaultdict(list)
    
    # Trouver tous les fichiers CSV de résultats
    result_files = list(results_dir.rglob('*.csv'))
    # Filtrer le fichier summary
    result_files = [f for f in result_files if f.name != 'analysis_summary.json']
    
    logger.info(f"Trouvé {len(result_files)} fichiers de résultats")
    
    # Parser chaque fichier
    for result_file in tqdm(result_files, desc="Lecture des résultats"):
        detections = parse_result_file(result_file, audio_root)
        
        # Grouper par fichier audio
        for detection in detections:
            all_detections[detection.audio_file].append(detection)
    
    # Trier les détections par confiance pour chaque fichier
    for audio_file in all_detections:
        all_detections[audio_file].sort(key=lambda d: d.confidence, reverse=True)
    
    return dict(all_detections)


def select_best_segments(detections: List[Detection], max_segments: int = 500,
                        min_overlap: float = 0.5, verbose: bool = False) -> List[Detection]:
    """
    Sélectionne les meilleurs segments en évitant trop de chevauchement.
    
    Args:
        detections: Liste des détections triées par confiance
        max_segments: Nombre maximum de segments à garder
        min_overlap: Chevauchement minimum pour considérer deux segments comme identiques
        
    Returns:
        Liste des meilleures détections sans trop de chevauchement
    """
    selected = []
    
    for detection in detections:
        # Vérifier le chevauchement avec les segments déjà sélectionnés
        overlaps = False
        
        for selected_det in selected:
            # Calculer le chevauchement
            overlap_start = max(detection.start_time, selected_det.start_time)
            overlap_end = min(detection.end_time, selected_det.end_time)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Ratio de chevauchement
            overlap_ratio = overlap_duration / detection.duration
            
            if overlap_ratio > min_overlap:
                overlaps = True
                if verbose:
                    logger.debug(f"  Segment {detection.start_time:.1f}-{detection.end_time:.1f}s "
                              f"chevauche avec {selected_det.start_time:.1f}-{selected_det.end_time:.1f}s")
                break
        
        if not overlaps:
            selected.append(detection)
            if verbose:
                logger.info(f"  Sélectionné [{len(selected)}] : {detection.start_time:.1f}-{detection.end_time:.1f}s "
                          f"{detection.species} (conf: {detection.confidence:.3f})")
            
        if len(selected) >= max_segments:
            break
    
    return selected


def extract_segment(detection: Detection, output_dir: Path, sample_rate: int = 22050) -> Optional[Path]:
    """
    Extrait un segment audio basé sur une détection.
    
    Args:
        detection: Objet Detection contenant les infos du segment
        output_dir: Répertoire de sortie
        sample_rate: Taux d'échantillonnage
        
    Returns:
        Chemin du fichier sauvegardé ou None si erreur
    """
    try:
        # Vérifier que le fichier existe
        if not detection.audio_file.exists():
            logger.error(f"Fichier audio introuvable : {detection.audio_file}")
            return None
        
        # Charger l'audio
        audio, sr = librosa.load(str(detection.audio_file), sr=sample_rate, mono=True)
        
        # Extraire le segment
        start_sample = int(detection.start_time * sample_rate)
        end_sample = int(detection.end_time * sample_rate)
        segment = audio[start_sample:end_sample]
        
        # Créer le répertoire de sortie par espèce
        species_dir = output_dir / detection.species
        species_dir.mkdir(parents=True, exist_ok=True)
        
        # Générer le nom du fichier
        timestamp = f"{int(detection.start_time*1000):06d}"
        confidence = int(detection.confidence * 100)
        output_name = f"{detection.audio_file.stem}_seg{timestamp}_conf{confidence}.wav"
        output_path = species_dir / output_name
        
        # Sauvegarder
        sf.write(str(output_path), segment, sample_rate)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction du segment : {e}")
        return None


def process_audio_file(args: Tuple[Path, List[Detection], Path, int, int, bool, bool]) -> Dict:
    """
    Traite un fichier audio et extrait ses segments (pour multiprocessing).
    
    Args:
        args: Tuple (audio_file, detections, output_dir, max_segments, sample_rate, verbose, dry_run)
        
    Returns:
        Dict avec les statistiques
    """
    audio_file, detections, output_dir, max_segments, sample_rate, verbose, dry_run = args
    
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Traitement de : {audio_file}")
        logger.info(f"Détections totales : {len(detections)}")
        logger.info(f"Sélection des {max_segments} meilleurs segments...")
    
    # Sélectionner les meilleurs segments
    selected_detections = select_best_segments(detections, max_segments, verbose=verbose)
    
    if verbose:
        logger.info(f"Segments sélectionnés : {len(selected_detections)}")
    
    # Extraire les segments
    extracted = 0
    errors = 0
    
    for i, detection in enumerate(selected_detections):
        if verbose and i < 5:  # Afficher les 5 premiers
            logger.info(f"\nExtraction [{i+1}/{len(selected_detections)}] : "
                      f"{detection.start_time:.1f}-{detection.end_time:.1f}s "
                      f"{detection.species} (conf: {detection.confidence:.3f})")
        
        if not dry_run:
            output_path = extract_segment(detection, output_dir, sample_rate)
            if output_path:
                extracted += 1
                if verbose and i < 5:
                    logger.info(f"  → Sauvegardé : {output_path.name}")
            else:
                errors += 1
        else:
            if verbose:
                logger.info(f"  [DRY-RUN] Extrairait : {detection.audio_file.stem}_"
                          f"seg{int(detection.start_time*1000):06d}_"
                          f"conf{int(detection.confidence*100)}.wav")
            extracted += 1  # Compter comme extrait en dry-run
    
    return {
        'audio_file': str(audio_file),
        'total_detections': len(detections),
        'selected': len(selected_detections),
        'extracted': extracted,
        'errors': errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extraction des segments audio basée sur les fichiers de résultats (Passe 2)"
    )
    
    # Entrées/Sorties
    parser.add_argument('--audio-input', type=Path, required=True,
                       help="Répertoire racine contenant les fichiers audio originaux")
    parser.add_argument('--results', type=Path, required=True,
                       help="Répertoire contenant les fichiers de résultats CSV")
    parser.add_argument('--output', type=Path, required=True,
                       help="Répertoire de sortie pour les segments extraits")
    
    # Paramètres de sélection
    parser.add_argument('--max-segments', type=int, default=500,
                       help="Nombre maximum de segments par fichier audio (défaut: 500)")
    parser.add_argument('--min-overlap', type=float, default=0.5,
                       help="Chevauchement minimum pour filtrer les doublons (défaut: 0.5)")
    
    # Paramètres audio
    parser.add_argument('--sample-rate', type=int, default=22050,
                       help="Taux d'échantillonnage (défaut: 22050)")
    
    # Performance
    parser.add_argument('--threads', type=int, default=1,
                       help="Nombre de threads CPU (défaut: 1)")
    
    # Options
    parser.add_argument('--verbose', action='store_true',
                       help="Mode verbose - affiche les détails de l'extraction")
    parser.add_argument('--dry-run', action='store_true',
                       help="Mode simulation - ne pas extraire les fichiers")
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Parser tous les résultats
    logger.info("Lecture des fichiers de résultats...")
    detections_by_file = parse_folders(args.audio_input, args.results)
    
    if not detections_by_file:
        logger.error("Aucune détection trouvée dans les fichiers de résultats!")
        return
    
    logger.info(f"Trouvé des détections pour {len(detections_by_file)} fichiers audio")
    
    # Préparer les arguments pour le traitement
    process_args = [
        (audio_file, detections, args.output, args.max_segments, args.sample_rate, 
         args.verbose, args.dry_run)
        for audio_file, detections in detections_by_file.items()
    ]
    
    # Traiter les fichiers
    if args.dry_run:
        logger.info(f"MODE DRY-RUN : Simulation de l'extraction des segments...")
    else:
        logger.info(f"Extraction des segments...")
    
    if args.threads < 2:
        # Mode single-thread
        results = []
        for args_tuple in tqdm(process_args, desc="Extraction des segments", disable=args.verbose):
            result = process_audio_file(args_tuple)
            results.append(result)
    else:
        # Mode multi-thread
        with Pool(args.threads) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_audio_file, process_args),
                total=len(process_args),
                desc="Extraction des segments"
            ))
    
    # Compiler les statistiques
    total_detections = sum(r['total_detections'] for r in results)
    total_selected = sum(r['selected'] for r in results)
    total_extracted = sum(r['extracted'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    
    # Sauvegarder le résumé
    summary_path = args.output / 'extraction_summary.json'
    summary = {
        'files_processed': len(results),
        'total_detections': total_detections,
        'total_selected': total_selected,
        'total_extracted': total_extracted,
        'total_errors': total_errors,
        'parameters': {
            'max_segments': args.max_segments,
            'min_overlap': args.min_overlap,
            'sample_rate': args.sample_rate
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Afficher le résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ DE L'EXTRACTION")
    print(f"{'='*60}")
    print(f"Fichiers traités : {len(results)}")
    print(f"Détections totales : {total_detections}")
    print(f"Segments sélectionnés : {total_selected}")
    print(f"Segments extraits : {total_extracted}")
    print(f"Erreurs : {total_errors}")
    
    if args.dry_run:
        print(f"\n⚠️  MODE DRY-RUN - Aucun fichier n'a été créé")
    else:
        print(f"\nSegments sauvegardés dans : {args.output}")
    print(f"Résumé : {summary_path}")
    
    if args.verbose:
        print(f"\nParamètres utilisés :")
        print(f"  Max segments/fichier : {args.max_segments}")
        print(f"  Chevauchement min : {args.min_overlap}")
        print(f"  Taux d'échantillonnage : {args.sample_rate}Hz")


if __name__ == "__main__":
    main()