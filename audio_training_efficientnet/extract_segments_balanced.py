#!/usr/bin/env python3
"""
Script d'extraction équilibrée des segments audio.
Limite le nombre de segments par classe ET par fichier audio source.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Set
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
        return f"Detection({self.species}, {self.confidence:.2f}, {self.start_time:.1f}-{self.end_time:.1f}s, {self.audio_file.name})"


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


def parse_all_results(audio_root: Path, results_dir: Path) -> Dict[str, List[Detection]]:
    """
    Parse tous les fichiers de résultats et groupe les détections par espèce.
    
    Args:
        audio_root: Répertoire racine des fichiers audio
        results_dir: Répertoire contenant les fichiers de résultats
        
    Returns:
        Dict mappant chaque espèce à ses détections
    """
    detections_by_species = defaultdict(list)
    
    # Trouver tous les fichiers CSV de résultats
    result_files = list(results_dir.rglob('*.csv'))
    # Filtrer le fichier summary
    result_files = [f for f in result_files if f.name != 'analysis_summary.json']
    
    logger.info(f"Trouvé {len(result_files)} fichiers de résultats")
    
    # Parser chaque fichier
    for result_file in tqdm(result_files, desc="Lecture des résultats"):
        detections = parse_result_file(result_file, audio_root)
        
        # Grouper par espèce
        for detection in detections:
            detections_by_species[detection.species].append(detection)
    
    # Trier les détections par confiance pour chaque espèce
    for species in detections_by_species:
        detections_by_species[species].sort(key=lambda d: d.confidence, reverse=True)
    
    return dict(detections_by_species)


def select_balanced_segments(detections: List[Detection], 
                           max_per_class: int = 500,
                           max_per_file: int = 5,
                           min_overlap: float = 0.5,
                           verbose: bool = False) -> List[Detection]:
    """
    Sélectionne les meilleurs segments avec limitation par classe ET par fichier.
    
    Args:
        detections: Liste des détections triées par confiance
        max_per_class: Nombre maximum de segments pour cette classe
        max_per_file: Nombre maximum de segments par fichier audio
        min_overlap: Chevauchement minimum pour considérer deux segments comme identiques
        verbose: Afficher les détails
        
    Returns:
        Liste des détections sélectionnées
    """
    selected = []
    segments_per_file = defaultdict(int)
    
    if verbose:
        logger.info(f"  Sélection parmi {len(detections)} détections...")
        logger.info(f"  Limites : max {max_per_class}/classe, max {max_per_file}/fichier")
    
    for detection in detections:
        # Vérifier la limite par fichier
        if segments_per_file[detection.audio_file] >= max_per_file:
            if verbose and len(selected) < 10:  # Log les 10 premiers rejets
                logger.debug(f"    Rejeté (limite fichier atteinte): {detection}")
            continue
        
        # Vérifier le chevauchement avec les segments déjà sélectionnés du même fichier
        overlaps = False
        for selected_det in selected:
            if selected_det.audio_file == detection.audio_file:
                # Calculer le chevauchement
                overlap_start = max(detection.start_time, selected_det.start_time)
                overlap_end = min(detection.end_time, selected_det.end_time)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                # Ratio de chevauchement
                overlap_ratio = overlap_duration / detection.duration
                
                if overlap_ratio > min_overlap:
                    overlaps = True
                    break
        
        if not overlaps:
            selected.append(detection)
            segments_per_file[detection.audio_file] += 1
            
            if verbose and len(selected) <= 10:
                logger.info(f"    [{len(selected)}] Sélectionné : {detection}")
        
        if len(selected) >= max_per_class:
            break
    
    if verbose:
        logger.info(f"  Total sélectionné : {len(selected)}")
        # Statistiques par fichier
        file_counts = list(segments_per_file.values())
        if file_counts:
            logger.info(f"  Fichiers utilisés : {len(file_counts)}")
            logger.info(f"  Segments/fichier : min={min(file_counts)}, max={max(file_counts)}, moy={sum(file_counts)/len(file_counts):.1f}")
    
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


def process_species(args: Tuple[str, List[Detection], Path, int, int, float, int, bool, bool]) -> Dict:
    """
    Traite toutes les détections d'une espèce (pour multiprocessing).
    
    Args:
        args: Tuple (species, detections, output_dir, max_per_class, max_per_file, 
                    min_overlap, sample_rate, verbose, dry_run)
        
    Returns:
        Dict avec les statistiques
    """
    species, detections, output_dir, max_per_class, max_per_file, min_overlap, sample_rate, verbose, dry_run = args
    
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info(f"Traitement espèce : {species}")
        logger.info(f"Détections totales : {len(detections)}")
    
    # Sélectionner les segments équilibrés
    selected_detections = select_balanced_segments(
        detections, 
        max_per_class=max_per_class,
        max_per_file=max_per_file,
        min_overlap=min_overlap,
        verbose=verbose
    )
    
    # Extraire les segments
    extracted = 0
    errors = 0
    
    for detection in tqdm(selected_detections, desc=f"Extraction {species}", disable=not verbose):
        if not dry_run:
            output_path = extract_segment(detection, output_dir, sample_rate)
            if output_path:
                extracted += 1
            else:
                errors += 1
        else:
            extracted += 1  # Compter comme extrait en dry-run
    
    return {
        'species': species,
        'total_detections': len(detections),
        'selected': len(selected_detections),
        'extracted': extracted,
        'errors': errors,
        'files_used': len(set(d.audio_file for d in selected_detections))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extraction équilibrée des segments audio (limite par classe ET par fichier)"
    )
    
    # Entrées/Sorties
    parser.add_argument('--audio-input', type=Path, required=True,
                       help="Répertoire racine contenant les fichiers audio originaux")
    parser.add_argument('--results', type=Path, required=True,
                       help="Répertoire contenant les fichiers de résultats CSV")
    parser.add_argument('--output', type=Path, required=True,
                       help="Répertoire de sortie pour les segments extraits")
    
    # Paramètres de sélection
    parser.add_argument('--max-per-class', type=int, default=500,
                       help="Nombre maximum de segments par classe/espèce (défaut: 500)")
    parser.add_argument('--max-per-file', type=int, default=5,
                       help="Nombre maximum de segments par fichier audio (défaut: 5)")
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
    parser.add_argument('--limit-species', type=int, default=None,
                       help="Limiter au N premières espèces (pour tests)")
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Parser tous les résultats groupés par espèce
    logger.info("Lecture des fichiers de résultats...")
    detections_by_species = parse_all_results(args.audio_input, args.results)
    
    if not detections_by_species:
        logger.error("Aucune détection trouvée dans les fichiers de résultats!")
        return
    
    logger.info(f"Trouvé {len(detections_by_species)} espèces avec détections")
    
    # Trier les espèces par nombre de détections (pour traiter les plus importantes en premier)
    species_list = sorted(detections_by_species.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Limiter si demandé
    if args.limit_species:
        species_list = species_list[:args.limit_species]
        logger.info(f"Limitation aux {args.limit_species} premières espèces")
    
    # Afficher un aperçu
    logger.info("\nAperçu des espèces à traiter :")
    for species, detections in species_list[:10]:
        logger.info(f"  {species}: {len(detections)} détections")
    if len(species_list) > 10:
        logger.info(f"  ... et {len(species_list)-10} autres espèces")
    
    # Préparer les arguments pour le traitement
    process_args = [
        (species, detections, args.output, args.max_per_class, args.max_per_file,
         args.min_overlap, args.sample_rate, args.verbose, args.dry_run)
        for species, detections in species_list
    ]
    
    # Traiter les espèces
    if args.dry_run:
        logger.info(f"\nMODE DRY-RUN : Simulation de l'extraction équilibrée...")
    else:
        logger.info(f"\nExtraction équilibrée des segments...")
    
    if args.threads < 2:
        # Mode single-thread
        results = []
        for args_tuple in process_args:
            result = process_species(args_tuple)
            results.append(result)
    else:
        # Mode multi-thread
        with Pool(args.threads) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_species, process_args),
                total=len(process_args),
                desc="Traitement des espèces"
            ))
    
    # Compiler les statistiques
    total_detections = sum(r['total_detections'] for r in results)
    total_selected = sum(r['selected'] for r in results)
    total_extracted = sum(r['extracted'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    
    # Sauvegarder le résumé
    summary_path = args.output / 'extraction_balanced_summary.json'
    summary = {
        'species_processed': len(results),
        'total_detections': total_detections,
        'total_selected': total_selected,
        'total_extracted': total_extracted,
        'total_errors': total_errors,
        'parameters': {
            'max_per_class': args.max_per_class,
            'max_per_file': args.max_per_file,
            'min_overlap': args.min_overlap,
            'sample_rate': args.sample_rate
        },
        'species_details': sorted(results, key=lambda x: x['extracted'], reverse=True)
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Afficher le résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ DE L'EXTRACTION ÉQUILIBRÉE")
    print(f"{'='*60}")
    print(f"Espèces traitées : {len(results)}")
    print(f"Détections totales : {total_detections}")
    print(f"Segments sélectionnés : {total_selected}")
    print(f"Segments extraits : {total_extracted}")
    print(f"Erreurs : {total_errors}")
    
    print(f"\nParamètres utilisés :")
    print(f"  Max segments/classe : {args.max_per_class}")
    print(f"  Max segments/fichier : {args.max_per_file}")
    print(f"  Chevauchement min : {args.min_overlap}")
    
    # Top 10 des espèces
    print(f"\nTop 10 espèces par nombre de segments extraits :")
    for i, result in enumerate(sorted(results, key=lambda x: x['extracted'], reverse=True)[:10]):
        print(f"  {i+1}. {result['species']}: {result['extracted']} segments "
              f"(depuis {result['files_used']} fichiers)")
    
    if args.dry_run:
        print(f"\n⚠️  MODE DRY-RUN - Aucun fichier n'a été créé")
    else:
        print(f"\nSegments sauvegardés dans : {args.output}")
    print(f"Résumé détaillé : {summary_path}")


if __name__ == "__main__":
    main()