"""
Module de segmentation audio pour NightScan
Découpe les fichiers audio longs en segments de taille fixe pour l'entraînement
"""

import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from typing import List, Tuple, Optional, Dict
import logging
from tqdm import tqdm
import json
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class AudioSegmenter:
    """
    Classe pour segmenter des fichiers audio longs en segments plus courts.
    """
    
    def __init__(self, 
                 segment_duration: float = 8.0,
                 overlap: float = 0.0,
                 min_segment_duration: float = 3.0,
                 sample_rate: Optional[int] = None):
        """
        Initialise le segmenteur audio.
        
        Args:
            segment_duration: Durée cible des segments en secondes
            overlap: Chevauchement entre segments en secondes (0 = pas de chevauchement)
            min_segment_duration: Durée minimale d'un segment pour être gardé
            sample_rate: Taux d'échantillonnage cible (None = garder l'original)
        """
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.min_segment_duration = min_segment_duration
        self.sample_rate = sample_rate
        self.hop_duration = segment_duration - overlap
        
        if self.hop_duration <= 0:
            raise ValueError("Le chevauchement doit être inférieur à la durée du segment")
        
        logger.info(f"Segmenteur initialisé: durée={segment_duration}s, overlap={overlap}s")
    
    def get_audio_info(self, audio_path: Path) -> Dict:
        """
        Obtient les informations d'un fichier audio.
        
        Args:
            audio_path: Chemin du fichier audio
            
        Returns:
            Dict avec durée, sample_rate, channels
        """
        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
        
        return {
            'duration': duration,
            'sample_rate': info.sample_rate,
            'channels': info.num_channels,
            'num_frames': info.num_frames
        }
    
    def segment_audio_file(self, 
                          input_path: Path, 
                          output_dir: Path,
                          prefix: Optional[str] = None) -> List[Dict]:
        """
        Segmente un fichier audio en plusieurs segments.
        
        Args:
            input_path: Chemin du fichier audio à segmenter
            output_dir: Répertoire de sortie pour les segments
            prefix: Préfixe pour les noms de fichiers de sortie
            
        Returns:
            Liste des informations sur les segments créés
        """
        # Obtenir les infos du fichier
        audio_info = self.get_audio_info(input_path)
        duration = audio_info['duration']
        
        # Si le fichier est déjà court, pas besoin de segmenter
        # On ne segmente que si on peut créer au moins 2 segments valides
        min_duration_for_segmentation = self.segment_duration + self.hop_duration
        if duration < min_duration_for_segmentation:
            logger.info(f"Fichier {input_path.name} trop court pour segmentation ({duration:.1f}s < {min_duration_for_segmentation:.1f}s)")
            
            # Copier directement le fichier au lieu de l'ignorer
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / input_path.name
            
            # Copier le fichier
            shutil.copy2(str(input_path), str(output_path))
            logger.info(f"Fichier {input_path.name} copié sans segmentation")
            
            # Retourner les métadonnées pour ce fichier non segmenté
            return [{
                'filename': input_path.name,
                'original_file': input_path.name,
                'start_time': 0,
                'end_time': duration,
                'actual_duration': duration,
                'padded_duration': duration,
                'segment_index': 0,
                'sample_rate': audio_info['sample_rate'],
                'has_padding': False,
                'copied_without_segmentation': True
            }]
        
        # Créer le répertoire de sortie
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger l'audio
        waveform, orig_sr = torchaudio.load(str(input_path))
        
        # Rééchantillonner si nécessaire
        if self.sample_rate and orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)
            sr = self.sample_rate
        else:
            sr = orig_sr
        
        # Calculer les segments
        segments = []
        segment_samples = int(self.segment_duration * sr)
        hop_samples = int(self.hop_duration * sr)
        
        # Générer le préfixe si non fourni
        if prefix is None:
            prefix = input_path.stem
        
        # Découper en segments
        start_sample = 0
        segment_idx = 0
        
        while start_sample < waveform.shape[1]:
            end_sample = min(start_sample + segment_samples, waveform.shape[1])
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # Calculer la durée réelle du segment (avant padding)
            actual_duration = (end_sample - start_sample) / sr
            
            # Sauvegarder seulement si le segment contient assez de signal audio
            if actual_duration >= self.min_segment_duration:
                # Appliquer le padding si nécessaire pour atteindre segment_duration
                if segment_waveform.shape[1] < segment_samples:
                    pad_length = segment_samples - segment_waveform.shape[1]
                    segment_waveform = torch.nn.functional.pad(
                        segment_waveform, 
                        (0, pad_length), 
                        mode='constant', 
                        value=0
                    )
                    logger.debug(f"Segment {segment_idx}: padding ajouté ({actual_duration:.2f}s → {self.segment_duration}s)")
                
                # Nom du fichier de sortie
                output_filename = f"{prefix}_seg{segment_idx:03d}.wav"
                output_path = output_dir / output_filename
                
                # Sauvegarder le segment (toujours de durée segment_duration)
                torchaudio.save(str(output_path), segment_waveform, sr)
                
                # Enregistrer les métadonnées
                segment_info = {
                    'filename': output_filename,
                    'original_file': input_path.name,
                    'start_time': start_sample / sr,
                    'end_time': end_sample / sr,
                    'actual_duration': actual_duration,  # Durée réelle avant padding
                    'padded_duration': self.segment_duration,  # Durée après padding
                    'segment_index': segment_idx,
                    'sample_rate': sr,
                    'has_padding': segment_waveform.shape[1] == segment_samples and actual_duration < self.segment_duration
                }
                segments.append(segment_info)
                
                logger.debug(f"Segment {segment_idx}: {actual_duration:.2f}s (padded to {self.segment_duration}s) sauvé dans {output_filename}")
                segment_idx += 1
            else:
                logger.debug(f"Segment ignoré (trop court: {actual_duration:.2f}s < {self.min_segment_duration}s)")
            
            # Passer au segment suivant
            start_sample += hop_samples
            
            # Arrêter si on a dépassé la fin
            if start_sample >= waveform.shape[1]:
                break
        
        logger.info(f"Fichier {input_path.name} segmenté en {len(segments)} segments")
        return segments
    
    def segment_directory(self, 
                         input_dir: Path,
                         output_dir: Path,
                         preserve_structure: bool = True,
                         file_pattern: str = "*.wav") -> Dict[str, List[Dict]]:
        """
        Segmente tous les fichiers audio d'un répertoire.
        
        Args:
            input_dir: Répertoire contenant les fichiers audio
            output_dir: Répertoire de sortie pour les segments
            preserve_structure: Si True, préserve la structure des sous-dossiers
            file_pattern: Pattern pour les fichiers à traiter
            
        Returns:
            Dict mapping fichier original -> liste des segments
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Collecter tous les fichiers audio
        audio_files = list(input_dir.rglob(file_pattern))
        
        if not audio_files:
            logger.warning(f"Aucun fichier trouvé avec le pattern '{file_pattern}' dans {input_dir}")
            return {}
        
        logger.info(f"Segmentation de {len(audio_files)} fichiers audio...")
        
        all_segments = {}
        stats = {'total_files': 0, 'segmented_files': 0, 'copied_files': 0, 'total_segments': 0}
        
        # Traiter chaque fichier
        for audio_file in tqdm(audio_files, desc="Segmentation"):
            stats['total_files'] += 1
            
            # Déterminer le répertoire de sortie
            if preserve_structure:
                # Préserver la structure relative
                relative_path = audio_file.relative_to(input_dir)
                segment_output_dir = output_dir / relative_path.parent
            else:
                # Tout dans le même répertoire
                segment_output_dir = output_dir
            
            # Segmenter le fichier
            try:
                segments = self.segment_audio_file(
                    audio_file,
                    segment_output_dir,
                    prefix=audio_file.stem
                )
                
                if segments:
                    all_segments[str(audio_file)] = segments
                    # Vérifier si c'est un fichier copié ou segmenté
                    if len(segments) == 1 and segments[0].get('copied_without_segmentation', False):
                        stats['copied_files'] += 1
                    else:
                        stats['segmented_files'] += 1
                        stats['total_segments'] += len(segments)
                    
            except Exception as e:
                logger.error(f"Erreur lors de la segmentation de {audio_file}: {e}")
        
        # Afficher les statistiques
        logger.info(f"\nStatistiques de segmentation:")
        logger.info(f"- Fichiers traités: {stats['total_files']}")
        logger.info(f"- Fichiers segmentés: {stats['segmented_files']}")
        logger.info(f"- Fichiers copiés (trop courts): {stats['copied_files']}")
        logger.info(f"- Total segments créés: {stats['total_segments']}")
        
        if stats['segmented_files'] > 0:
            avg_segments = stats['total_segments'] / stats['segmented_files']
            logger.info(f"- Moyenne segments/fichier segmenté: {avg_segments:.1f}")
        
        # Sauvegarder les métadonnées
        metadata_path = output_dir / "segmentation_metadata.json"
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'segment_duration': self.segment_duration,
                'overlap': self.overlap,
                'min_segment_duration': self.min_segment_duration,
                'sample_rate': self.sample_rate
            },
            'statistics': stats,
            'segments': all_segments
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Métadonnées sauvegardées dans {metadata_path}")
        
        return all_segments


def segment_for_training(input_dir: Path,
                        output_dir: Path,
                        segment_duration: float = 8.0,
                        overlap: float = 2.0) -> None:
    """
    Fonction utilitaire pour segmenter un dataset pour l'entraînement.
    
    Args:
        input_dir: Répertoire contenant les fichiers audio originaux
        output_dir: Répertoire de sortie pour les segments
        segment_duration: Durée des segments en secondes
        overlap: Chevauchement entre segments en secondes
    """
    segmenter = AudioSegmenter(
        segment_duration=segment_duration,
        overlap=overlap,
        min_segment_duration=3.0
    )
    
    # Segmenter en préservant la structure des classes
    segmenter.segment_directory(
        input_dir,
        output_dir,
        preserve_structure=True
    )


def main():
    """Fonction principale pour utilisation en ligne de commande."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Segmente des fichiers audio longs en segments plus courts"
    )
    parser.add_argument('input_dir', type=Path,
                       help="Répertoire contenant les fichiers audio")
    parser.add_argument('output_dir', type=Path,
                       help="Répertoire de sortie pour les segments")
    parser.add_argument('--duration', type=float, default=8.0,
                       help="Durée des segments en secondes (défaut: 8)")
    parser.add_argument('--overlap', type=float, default=2.0,
                       help="Chevauchement entre segments en secondes (défaut: 2)")
    parser.add_argument('--min-duration', type=float, default=3.0,
                       help="Durée minimale d'un segment (défaut: 3)")
    parser.add_argument('--sample-rate', type=int, default=None,
                       help="Taux d'échantillonnage cible (défaut: garder l'original)")
    parser.add_argument('--no-preserve-structure', action='store_true',
                       help="Ne pas préserver la structure des dossiers")
    
    args = parser.parse_args()
    
    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Créer le segmenteur
    segmenter = AudioSegmenter(
        segment_duration=args.duration,
        overlap=args.overlap,
        min_segment_duration=args.min_duration,
        sample_rate=args.sample_rate
    )
    
    # Segmenter le répertoire
    segmenter.segment_directory(
        args.input_dir,
        args.output_dir,
        preserve_structure=not args.no_preserve_structure
    )
    
    print(f"\n✅ Segmentation terminée! Les segments sont dans: {args.output_dir}")


if __name__ == "__main__":
    main()