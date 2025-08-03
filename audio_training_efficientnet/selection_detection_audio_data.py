#!/usr/bin/env python3
"""
Script de sélection des données audio basé sur la détection par modèle ML
Utilise un modèle EfficientNet entraîné pour filtrer les segments contenant de vraies détections
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
import argparse
import json
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import librosa
import soundfile as sf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ajouter le chemin parent pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_config import create_audio_model, get_audio_classes
from spectrogram_config import SpectrogramConfig, get_config_for_animal

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBasedDetector:
    """Détecteur basé sur un modèle ML pour filtrer les segments audio."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = None,
                 sample_rate: int = 22050,  # Changé de 48000 à 22050 (comme l'entraînement)
                 n_fft: int = 2048,         # Changé de 1024 à 2048 (comme l'entraînement)
                 hop_length: int = 512,     # Changé de 320 à 512 (comme l'entraînement)
                 n_mels: int = 128,
                 fmin: int = 50,            # Changé de 0 à 50 (comme l'entraînement)
                 fmax: int = 11000,         # Changé de 24000 à 11000 (comme l'entraînement)
                 index_db: Optional[str] = None,
                 training_db: Optional[str] = None,
                 class_list_file: Optional[str] = None):
        """
        Initialise le détecteur.
        
        Args:
            model_path: Chemin vers le checkpoint du modèle
            device: Device à utiliser (auto-détection si None)
            sample_rate: Taux d'échantillonnage (défaut: 22050 comme l'entraînement)
            n_fft: Taille FFT (défaut: 2048 comme l'entraînement)
            hop_length: Hop length pour le spectrogramme (défaut: 512 comme l'entraînement)
            n_mels: Nombre de bandes mel (défaut: 128)
            fmin: Fréquence minimale (défaut: 50 comme l'entraînement)
            fmax: Fréquence maximale (défaut: 11000 comme l'entraînement)
            index_db: Base SQLite contenant les fichiers à traiter (optionnel)
            training_db: Base SQLite utilisée pendant l'entraînement (IMPORTANT: pour l'ordre des classes)
            class_list_file: Fichier texte avec les noms de classes
        """
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.sample_rate = sample_rate
        
        # Paramètres spectrogramme
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        # Charger le modèle
        logger.info(f"Chargement du modèle depuis {model_path}")
        # PyTorch 2.6+ nécessite weights_only=False pour charger les checkpoints avec des objets Python
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Vérifier si les classes sont dans le checkpoint
        self.checkpoint_class_names = checkpoint.get('class_names', None)
        if self.checkpoint_class_names:
            logger.info(f"✅ Classes trouvées dans le checkpoint: {len(self.checkpoint_class_names)} classes")
            
        # Extraire les informations du modèle
        if 'args' in checkpoint:
            self.num_classes = checkpoint['args'].num_classes
            model_name = getattr(checkpoint['args'], 'model', 'efficientnet-b1')
        else:
            self.num_classes = checkpoint.get('num_classes', 10)
            model_name = checkpoint.get('model_name', 'efficientnet-b1')
        
        # Créer et charger le modèle
        self.model = create_audio_model(
            num_classes=self.num_classes,
            model_name=model_name,
            pretrained=False,
            dropout_rate=0.0
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Obtenir les noms de classes
        self.index_db = index_db
        self.training_db = training_db
        self.class_list_file = class_list_file
        self.class_names = self._get_class_names()
        
        logger.info(f"Modèle chargé: {model_name} avec {self.num_classes} classes")
        logger.info(f"Device: {self.device}")
        if len(self.class_names) <= 20:
            logger.info(f"Classes: {', '.join(self.class_names)}")
        else:
            logger.info(f"Classes: {', '.join(self.class_names[:10])} ... et {len(self.class_names)-10} autres")
        
        # Transform pour mel spectrogramme
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax
        ).to(self.device)
        
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80).to(self.device)
    
    def _get_class_names(self) -> List[str]:
        """Récupère les noms de classes depuis la base de données ou utilise les défauts."""
        import sqlite3
        
        # Option 0: Si les classes sont dans le checkpoint, les utiliser (PLUS FIABLE)
        if self.checkpoint_class_names:
            if len(self.checkpoint_class_names) == self.num_classes:
                logger.info(f"✅ Utilisation des classes du checkpoint (plus fiable)")
                logger.info(f"Ordre des classes (premières 10): {self.checkpoint_class_names[:10]}")
                return self.checkpoint_class_names
            else:
                logger.warning(f"Classes dans checkpoint ({len(self.checkpoint_class_names)}) != num_classes ({self.num_classes})")
        
        # Option 1: Charger depuis la base d'entraînement (PRIORITAIRE)
        if self.training_db and Path(self.training_db).exists():
            try:
                logger.info(f"Chargement des classes depuis la base d'entraînement: {self.training_db}")
                conn = sqlite3.connect(self.training_db)
                cursor = conn.cursor()
                
                # Récupérer les classes distinctes dans le MÊME ORDRE que pendant l'entraînement
                # C'est CRITIQUE: l'ordre doit être identique !
                cursor.execute("""
                    SELECT DISTINCT class_name 
                    FROM audio_samples 
                    WHERE split = 'train'
                    ORDER BY class_name
                """)
                class_names = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                if len(class_names) == self.num_classes:
                    logger.info(f"✅ Classes chargées depuis la base d'entraînement: {len(class_names)} classes")
                    logger.info(f"Ordre des classes (premières 10): {class_names[:10]}")
                    return class_names
                else:
                    logger.error(f"❌ ERREUR CRITIQUE: Nombre de classes dans la base d'entraînement ({len(class_names)}) != num_classes du modèle ({self.num_classes})")
                    logger.error(f"Classes trouvées: {class_names}")
                    raise ValueError(f"Incohérence du nombre de classes: {len(class_names)} != {self.num_classes}")
            except Exception as e:
                logger.error(f"Erreur lecture base d'entraînement: {e}")
                raise
                
        # Option 2: Charger depuis la base d'index (moins fiable)
        elif self.index_db and Path(self.index_db).exists():
            logger.warning("⚠️  Utilisation de la base d'index au lieu de la base d'entraînement - risque d'ordre incorrect!")
            try:
                conn = sqlite3.connect(self.index_db)
                cursor = conn.cursor()
                
                # Récupérer les classes distinctes triées par ordre alphabétique
                cursor.execute("SELECT DISTINCT class_name FROM audio_samples ORDER BY class_name")
                class_names = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                if len(class_names) == self.num_classes:
                    logger.info(f"Classes chargées depuis la base d'index: {len(class_names)} classes")
                    return class_names
                else:
                    logger.warning(f"Nombre de classes dans la base ({len(class_names)}) != num_classes du modèle ({self.num_classes})")
            except Exception as e:
                logger.error(f"Erreur lecture SQLite: {e}")
        
        # Option 3: Charger depuis un fichier texte
        elif self.class_list_file and Path(self.class_list_file).exists():
            try:
                with open(self.class_list_file, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                
                if len(class_names) == self.num_classes:
                    logger.info(f"Classes chargées depuis {self.class_list_file}: {len(class_names)} classes")
                    return class_names
                else:
                    logger.warning(f"Nombre de classes dans le fichier ({len(class_names)}) != num_classes du modèle ({self.num_classes})")
            except Exception as e:
                logger.error(f"Erreur lecture fichier classes: {e}")
        
        # Option 4: Utiliser les classes par défaut ou générer
        default_classes = get_audio_classes()
        
        if self.num_classes <= len(default_classes):
            logger.error("❌ ATTENTION: Utilisation des classes par défaut - les détections seront INCORRECTES!")
            logger.error("Utilisez --training-db pour spécifier la base SQLite d'entraînement")
            return default_classes[:self.num_classes]
        else:
            logger.error(f"❌ ERREUR: Impossible de déterminer les noms de classes pour {self.num_classes} classes")
            logger.error("Utilisez --training-db pour spécifier la base SQLite d'entraînement")
            raise ValueError("Impossible de charger les noms de classes - spécifiez --training-db")
    
    def _generate_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Génère un spectrogramme mel à partir d'un signal audio.
        
        Args:
            audio: Signal audio (1, samples)
            
        Returns:
            Spectrogramme mel en dB (3, n_mels, time)
        """
        # Calculer le spectrogramme mel
        mel_spec = self.mel_transform(audio)
        
        # Convertir en dB
        mel_spec_db = self.db_transform(mel_spec)
        
        # Normaliser avec Z-score (comme pendant l'entraînement)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        # Convertir en 3 canaux pour le modèle
        if mel_spec_db.dim() == 3:
            mel_spec_db = mel_spec_db.squeeze(0)
        mel_spec_db = mel_spec_db.unsqueeze(0).repeat(3, 1, 1)
        
        return mel_spec_db
    
    def process_audio_window(self, audio: torch.Tensor, min_confidence: float = 0.3) -> Dict:
        """
        Process une fenêtre audio et retourne les prédictions.
        
        Args:
            audio: Signal audio (1, samples)
            min_confidence: Confiance minimale pour considérer une détection
            
        Returns:
            Dict avec les résultats de détection
        """
        with torch.no_grad():
            # Générer le spectrogramme
            spectrogram = self._generate_spectrogram(audio)
            spectrogram = spectrogram.unsqueeze(0).to(self.device)
            
            # Prédiction
            outputs = self.model(spectrogram)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Obtenir la classe prédite et la confiance
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            
            # Obtenir les top-3 predictions
            top3_probs, top3_classes = torch.topk(probabilities, k=min(3, self.num_classes), dim=1)
            
            result = {
                'detected': confidence >= min_confidence,
                'class': self.class_names[predicted_class],
                'class_id': predicted_class,
                'confidence': confidence,
                'top3_classes': [self.class_names[idx] for idx in top3_classes[0].cpu().numpy()],
                'top3_confidences': top3_probs[0].cpu().numpy().tolist()
            }
            
            return result
    
    def process_file(self, 
                    audio_path: Path,
                    window_size: float = 3.0,
                    hop_size: float = 1.5,
                    min_confidence: float = 0.3,
                    target_classes: Optional[List[str]] = None,
                    energy_threshold: float = -50.0) -> List[Dict]:
        """
        Process un fichier audio complet avec fenêtre glissante.
        
        Args:
            audio_path: Chemin du fichier audio
            window_size: Taille de la fenêtre en secondes
            hop_size: Décalage entre fenêtres en secondes
            min_confidence: Confiance minimale
            target_classes: Liste des classes cibles (None = toutes)
            energy_threshold: Seuil d'énergie en dB pour pré-filtrage
            
        Returns:
            Liste des détections
        """
        # Charger l'audio
        try:
            audio, sr = torchaudio.load(str(audio_path))
            
            # Resample si nécessaire
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Convertir en mono si nécessaire
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
        except Exception as e:
            logger.error(f"Erreur chargement {audio_path}: {e}")
            return []
        
        # Calculer les paramètres de fenêtrage
        window_samples = int(window_size * self.sample_rate)
        hop_samples = int(hop_size * self.sample_rate)
        
        detections = []
        
        # Parcourir avec fenêtre glissante
        for start_idx in range(0, audio.shape[1] - window_samples + 1, hop_samples):
            end_idx = start_idx + window_samples
            window_audio = audio[:, start_idx:end_idx]
            
            # Pré-filtrage par énergie (optionnel)
            if energy_threshold is not None:
                rms = torch.sqrt(torch.mean(window_audio**2))
                rms_db = 20 * torch.log10(rms + 1e-8)
                if rms_db < energy_threshold:
                    continue
            
            # Détection
            result = self.process_audio_window(window_audio, min_confidence)
            
            # Filtrer par classes cibles si spécifié
            if target_classes and result['class'] not in target_classes:
                result['detected'] = False
            
            if result['detected']:
                result['start_time'] = start_idx / self.sample_rate
                result['end_time'] = end_idx / self.sample_rate
                result['file_path'] = str(audio_path)
                detections.append(result)
        
        return detections


def extract_and_save_segment(audio_path: Path, 
                           detection: Dict,
                           output_dir: Path,
                           sample_rate: int = 48000,
                           preserve_structure: bool = True) -> Path:
    """
    Extrait et sauvegarde un segment audio basé sur une détection.
    
    Args:
        audio_path: Chemin du fichier audio source
        detection: Dict contenant les infos de détection
        output_dir: Répertoire de sortie
        sample_rate: Taux d'échantillonnage
        preserve_structure: Préserver la structure des dossiers
        
    Returns:
        Chemin du fichier sauvegardé
    """
    # Charger l'audio
    audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    
    # Extraire le segment
    start_sample = int(detection['start_time'] * sample_rate)
    end_sample = int(detection['end_time'] * sample_rate)
    segment = audio[start_sample:end_sample]
    
    # Déterminer le chemin de sortie
    if preserve_structure:
        # Garder la structure originale des classes
        relative_path = audio_path.parent.name  # Nom de la classe
        class_output_dir = output_dir / relative_path
    else:
        # Organiser par classe détectée
        class_output_dir = output_dir / detection['class']
    
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Nom du fichier avec métadonnées
    timestamp = f"{int(detection['start_time']*1000):06d}"
    confidence = int(detection['confidence'] * 100)
    output_name = f"{audio_path.stem}_seg{timestamp}_conf{confidence}_{detection['class']}.wav"
    output_path = class_output_dir / output_name
    
    # Sauvegarder
    sf.write(str(output_path), segment, sample_rate)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Sélection des segments audio basée sur la détection par modèle ML"
    )
    
    # Entrées/Sorties
    parser.add_argument('--audio-root', type=Path, required=True,
                       help="Répertoire racine contenant les fichiers audio")
    parser.add_argument('--output-dir', type=Path, required=True,
                       help="Répertoire de sortie pour les segments filtrés")
    
    # Modèle
    parser.add_argument('--model', type=str, required=True,
                       help="Chemin vers le checkpoint du modèle (.pth)")
    parser.add_argument('--device', type=str, default=None,
                       help="Device à utiliser (cuda/cpu, auto si non spécifié)")
    parser.add_argument('--index-db', type=str, default=None,
                       help="Base SQLite pour charger les noms de classes (balanced_audio_index.db)")
    parser.add_argument('--training-db', type=str, default=None,
                       help="Base SQLite utilisée pendant l'entraînement (IMPORTANT pour l'ordre des classes)")
    parser.add_argument('--class-list', type=str, default=None,
                       help="Fichier texte contenant les noms de classes (une par ligne)")
    
    # Paramètres de détection
    parser.add_argument('--min-conf', type=float, default=0.3,
                       help="Confiance minimale pour garder un segment (défaut: 0.3)")
    parser.add_argument('--target-list', type=Path, default=None,
                       help="Fichier texte contenant les classes cibles (une par ligne)")
    
    # Paramètres de fenêtrage
    parser.add_argument('--win', type=float, default=3.0,
                       help="Taille de la fenêtre en secondes (défaut: 3.0 comme l'entraînement)")
    parser.add_argument('--hop', type=float, default=1.5,
                       help="Décalage entre fenêtres en secondes (défaut: 1.5)")
    
    # Paramètres audio
    parser.add_argument('--sr', type=str, default='22050',
                       help="Taux d'échantillonnage (défaut: 22050 comme l'entraînement)")
    parser.add_argument('--energy-threshold', type=float, default=-50.0,
                       help="Seuil d'énergie en dB pour pré-filtrage (défaut: -50)")
    
    # Options
    parser.add_argument('--preserve-structure', action='store_true',
                       help="Préserver la structure originale des dossiers")
    parser.add_argument('--max-segments-per-file', type=int, default=None,
                       help="Nombre maximum de segments par fichier source")
    parser.add_argument('--dry-run', action='store_true',
                       help="Mode simulation - ne pas extraire les fichiers")
    parser.add_argument('--report', type=Path, default=None,
                       help="Générer un rapport de détection (JSON)")
    parser.add_argument('--verbose', action='store_true',
                       help="Afficher des informations détaillées pendant le traitement")
    parser.add_argument('--validation-mode', action='store_true',
                       help="Mode validation : compare les détections avec les dossiers sources et affiche les statistiques")
    
    args = parser.parse_args()
    
    # Parser le taux d'échantillonnage
    if args.sr.endswith('k'):
        sample_rate = int(float(args.sr[:-1]) * 1000)
    else:
        sample_rate = int(args.sr)
    
    # Charger les classes cibles si spécifié
    target_classes = None
    if args.target_list:
        with open(args.target_list, 'r') as f:
            target_classes = [line.strip() for line in f if line.strip()]
        logger.info(f"Classes cibles: {', '.join(target_classes)}")
    
    # Avertissement si pas de base d'entraînement
    if not args.training_db:
        logger.warning("⚠️  ATTENTION: --training-db non spécifié")
        logger.warning("Les classes pourraient être dans le mauvais ordre, causant des détections incorrectes")
        logger.warning("Utilisez la même base SQLite que celle utilisée pour l'entraînement")
        
    # Créer le détecteur
    detector = ModelBasedDetector(
        model_path=args.model,
        device=args.device,
        sample_rate=sample_rate,
        index_db=args.index_db,
        training_db=args.training_db,
        class_list_file=args.class_list
    )
    
    # Scanner les fichiers audio
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = []
    
    for ext in audio_extensions:
        files = args.audio_root.rglob(f'*{ext}')
        # Filtrer les fichiers cachés macOS (commençant par ._)
        audio_files.extend([f for f in files if not f.name.startswith('._')])
    
    logger.info(f"Fichiers audio trouvés: {len(audio_files)}")
    
    # Créer le répertoire de sortie
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistiques globales
    global_stats = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(args.model),
        'min_confidence': args.min_conf,
        'window_size': args.win,
        'hop_size': args.hop,
        'total_files': len(audio_files),
        'total_detections': 0,
        'detections_by_class': {},
        'files_with_detections': 0,
        'all_detections': [],
        'validation_stats': {
            'correct_detections': 0,
            'incorrect_detections': 0,
            'accuracy_by_folder': {},
            'confusion_matrix': {}
        } if args.validation_mode else None
    }
    
    # Traiter chaque fichier
    pbar = tqdm(audio_files, desc="Traitement des fichiers")
    for file_idx, audio_file in enumerate(pbar):
        # Détections pour ce fichier
        detections = detector.process_file(
            audio_file,
            window_size=args.win,
            hop_size=args.hop,
            min_confidence=args.min_conf,
            target_classes=target_classes,
            energy_threshold=args.energy_threshold
        )
        
        # Classe originale du fichier (nom du dossier)
        original_class = audio_file.parent.name
        
        # Statistiques de validation si mode validation
        if args.validation_mode and detections:
            if original_class not in global_stats['validation_stats']['accuracy_by_folder']:
                global_stats['validation_stats']['accuracy_by_folder'][original_class] = {
                    'correct': 0, 'incorrect': 0, 'total': 0
                }
            
            # Initialiser la matrice de confusion pour cette classe
            if original_class not in global_stats['validation_stats']['confusion_matrix']:
                global_stats['validation_stats']['confusion_matrix'][original_class] = {}
        
        # Logs détaillés si verbose
        if args.verbose and detections:
            logger.info(f"\n{'='*60}")
            logger.info(f"Fichier: {audio_file.name}")
            logger.info(f"Classe originale: {original_class}")
            logger.info(f"Détections trouvées: {len(detections)}")
            
            # Afficher les détections avec validation
            for i, det in enumerate(detections[:5]):  # Limiter à 5 pour ne pas spam
                is_correct = det['class'] == original_class
                status = "✓" if is_correct else "✗"
                logger.info(f"  [{i+1}] {det['start_time']:.1f}-{det['end_time']:.1f}s : "
                          f"{det['class']} (conf: {det['confidence']:.2%}) {status}")
            if len(detections) > 5:
                logger.info(f"  ... et {len(detections)-5} autres détections")
        
        if detections:
            global_stats['files_with_detections'] += 1
            
            # Limiter le nombre de segments par fichier si demandé
            if args.max_segments_per_file and len(detections) > args.max_segments_per_file:
                # Garder les détections avec la plus haute confiance
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                detections = detections[:args.max_segments_per_file]
                if args.verbose:
                    logger.info(f"  → Limité à {args.max_segments_per_file} segments (plus haute confiance)")
            
            # Extraire et sauvegarder les segments
            for detection in detections:
                # Validation : vérifier si la détection correspond au dossier
                is_correct_detection = detection['class'] == original_class
                
                # En mode validation, compter et éventuellement filtrer
                if args.validation_mode:
                    global_stats['validation_stats']['accuracy_by_folder'][original_class]['total'] += 1
                    
                    if is_correct_detection:
                        global_stats['validation_stats']['correct_detections'] += 1
                        global_stats['validation_stats']['accuracy_by_folder'][original_class]['correct'] += 1
                    else:
                        global_stats['validation_stats']['incorrect_detections'] += 1
                        global_stats['validation_stats']['accuracy_by_folder'][original_class]['incorrect'] += 1
                        
                        # Matrice de confusion
                        detected_class = detection['class']
                        if detected_class not in global_stats['validation_stats']['confusion_matrix'][original_class]:
                            global_stats['validation_stats']['confusion_matrix'][original_class][detected_class] = 0
                        global_stats['validation_stats']['confusion_matrix'][original_class][detected_class] += 1
                    
                    # En mode validation strict, ne garder que les détections correctes
                    if not is_correct_detection and not args.dry_run:
                        continue  # Skip cette détection
                
                global_stats['total_detections'] += 1
                
                # Statistiques par classe
                class_name = detection['class']
                if class_name not in global_stats['detections_by_class']:
                    global_stats['detections_by_class'][class_name] = 0
                global_stats['detections_by_class'][class_name] += 1
                
                # Sauvegarder si pas en mode dry-run
                if not args.dry_run:
                    output_path = extract_and_save_segment(
                        audio_file,
                        detection,
                        args.output_dir,
                        sample_rate,
                        args.preserve_structure
                    )
                    detection['output_path'] = str(output_path)
                elif args.verbose:
                    # En dry-run verbose, montrer ce qui serait sauvé
                    status = "✓" if is_correct_detection else "✗"
                    logger.info(f"  [DRY-RUN] Sauverait: {detection['start_time']:.1f}-{detection['end_time']:.1f}s "
                              f"comme {detection['class']}_conf{int(detection['confidence']*100)}.wav {status}")
                
                # Ajouter aux détections globales pour le rapport
                if args.report:
                    detection['is_correct'] = is_correct_detection
                    detection['original_class'] = original_class
                    global_stats['all_detections'].append(detection)
        
        # Mettre à jour la barre de progression avec les stats
        if args.validation_mode and global_stats['validation_stats']['correct_detections'] + global_stats['validation_stats']['incorrect_detections'] > 0:
            accuracy = global_stats['validation_stats']['correct_detections'] / (global_stats['validation_stats']['correct_detections'] + global_stats['validation_stats']['incorrect_detections']) * 100
            pbar.set_postfix({
                'Détections': global_stats['total_detections'],
                'Corrects': global_stats['validation_stats']['correct_detections'],
                'Précision': f"{accuracy:.1f}%",
                'Classe': original_class[:15]
            })
        else:
            pbar.set_postfix({
                'Détections': global_stats['total_detections'],
                'Fichiers OK': global_stats['files_with_detections'],
                'Classe': original_class[:15]
            })
        
        # Résumé périodique
        if (file_idx + 1) % 100 == 0:
            logger.info(f"\n--- Résumé après {file_idx + 1} fichiers ---")
            logger.info(f"Fichiers avec détections: {global_stats['files_with_detections']}")
            logger.info(f"Total détections: {global_stats['total_detections']}")
            logger.info(f"Détections par classe: {dict(sorted(global_stats['detections_by_class'].items()))}")
    
    # Afficher les résultats
    print(f"\n{'='*60}")
    print("RÉSULTATS DE LA SÉLECTION PAR DÉTECTION")
    print(f"{'='*60}")
    print(f"Fichiers traités: {global_stats['total_files']}")
    print(f"Fichiers avec détections: {global_stats['files_with_detections']}")
    print(f"Total détections: {global_stats['total_detections']}")
    
    # Résultats de validation si mode validation
    if args.validation_mode:
        val_stats = global_stats['validation_stats']
        total_val = val_stats['correct_detections'] + val_stats['incorrect_detections']
        if total_val > 0:
            accuracy = val_stats['correct_detections'] / total_val * 100
            print(f"\n🎯 STATISTIQUES DE VALIDATION:")
            print(f"  Détections correctes: {val_stats['correct_detections']}")
            print(f"  Détections incorrectes: {val_stats['incorrect_detections']}")
            print(f"  Précision globale: {accuracy:.2f}%")
            
            print(f"\n📊 Précision par dossier:")
            for folder, stats in sorted(val_stats['accuracy_by_folder'].items()):
                if stats['total'] > 0:
                    folder_accuracy = stats['correct'] / stats['total'] * 100
                    print(f"  {folder}: {stats['correct']}/{stats['total']} ({folder_accuracy:.1f}%)")
            
            # Top confusions
            print(f"\n❌ Top 10 confusions:")
            confusions = []
            for true_class, predictions in val_stats['confusion_matrix'].items():
                for pred_class, count in predictions.items():
                    confusions.append((true_class, pred_class, count))
            confusions.sort(key=lambda x: x[2], reverse=True)
            for true_class, pred_class, count in confusions[:10]:
                print(f"  {true_class} → {pred_class}: {count}")
    
    print(f"\nDétections par classe:")
    for class_name, count in sorted(global_stats['detections_by_class'].items()):
        print(f"  {class_name}: {count}")
    
    # Sauvegarder le rapport si demandé
    if args.report:
        # Limiter la taille du rapport
        if len(global_stats['all_detections']) > 1000:
            logger.warning("Rapport limité aux 1000 premières détections")
            global_stats['all_detections'] = global_stats['all_detections'][:1000]
        
        with open(args.report, 'w') as f:
            json.dump(global_stats, f, indent=2)
        print(f"\nRapport sauvegardé: {args.report}")
    
    if args.dry_run:
        print("\n⚠️  Mode DRY-RUN - Aucun fichier n'a été créé")
    else:
        print(f"\n✅ Segments sauvegardés dans: {args.output_dir}")


if __name__ == "__main__":
    main()