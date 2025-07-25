"""
Dataset PyTorch scalable pour grandes bases de données audio (1500+ classes)
Optimisé pour gérer des millions d'échantillons avec une empreinte mémoire minimale
"""

import os
import sys
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Iterator
import logging
import random
import json
import gc
from contextlib import contextmanager

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter les chemins pour les imports
sys.path.append(str(Path(__file__).parent.parent))
from spectrogram_config import SpectrogramConfig, get_config_for_animal

# Limites de mémoire
MAX_MEMORY_MB = 2048  # Maximum 2GB pour le dataset
MAX_SAMPLES_IN_MEMORY = 10000  # Maximum d'échantillons gardés en mémoire


class AudioDatasetScalable(Dataset):
    """
    Dataset scalable pour l'entraînement sur de très grandes bases de données audio.
    Utilise SQLite pour l'indexation et charge les données à la demande.
    """
    
    def __init__(self, 
                 index_db: str,
                 audio_root: Path,
                 config: Optional[SpectrogramConfig] = None,
                 animal_type: str = 'general',
                 transform=None,
                 augment: bool = False,
                 max_samples_per_class: int = 500,
                 balance_classes: bool = True,
                 chunk_size: int = 10000):
        """
        Args:
            index_db: Chemin vers la base SQLite contenant l'index
            audio_root: Répertoire racine des fichiers audio
            config: Configuration de spectrogramme
            animal_type: Type d'animal pour configuration automatique
            transform: Transformations PyTorch à appliquer
            augment: Si True, applique l'augmentation à la volée
            max_samples_per_class: Limite max d'échantillons par classe
            balance_classes: Si True, équilibre les classes lors du sampling
            chunk_size: Taille des chunks pour le chargement par batch
        """
        self.index_db = index_db
        self.audio_root = Path(audio_root)
        self.transform = transform
        self.augment = augment
        self.max_samples_per_class = max_samples_per_class
        self.balance_classes = balance_classes
        self.chunk_size = chunk_size
        
        # Configuration du spectrogramme
        if config is None:
            self.config = get_config_for_animal(animal_type)
        else:
            self.config = config
        
        # Paramètres extraits de la config
        self.sample_rate = self.config.sample_rate
        self.n_mels = self.config.n_mels
        self.duration = self.config.duration
        
        # Connexion à la base de données
        self.conn = sqlite3.connect(index_db, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Initialiser les métadonnées
        self._init_metadata()
        
        # Cache léger pour les échantillons récents
        self.cache = {}
        self.cache_size = 0
        self.max_cache_size = 1000  # Maximum 1000 spectrogrammes en cache
        
        logger.info(f"Dataset initialisé: {self.total_samples} échantillons, {self.num_classes} classes")
        logger.info(f"Mémoire maximale: {MAX_MEMORY_MB}MB, Cache max: {self.max_cache_size} spectrogrammes")
    
    def _init_metadata(self):
        """Initialise les métadonnées depuis la base de données."""
        cursor = self.conn.cursor()
        
        # Obtenir le nombre total d'échantillons
        cursor.execute("SELECT COUNT(*) FROM audio_samples")
        self.total_samples = cursor.fetchone()[0]
        
        # Obtenir les classes et leurs comptes
        cursor.execute("""
            SELECT class_name, COUNT(*) as count 
            FROM audio_samples 
            GROUP BY class_name
            ORDER BY class_name
        """)
        
        self.class_info = {}
        self.class_names = []
        self.class_to_idx = {}
        
        for idx, row in enumerate(cursor.fetchall()):
            class_name = row['class_name']
            count = min(row['count'], self.max_samples_per_class)
            
            self.class_names.append(class_name)
            self.class_to_idx[class_name] = idx
            self.class_info[class_name] = {
                'count': count,
                'original_count': row['count'],
                'idx': idx
            }
        
        self.num_classes = len(self.class_names)
        
        # Créer l'index d'échantillonnage équilibré si nécessaire
        if self.balance_classes:
            self._create_balanced_index()
        
        cursor.close()
    
    def _create_balanced_index(self):
        """Crée un index équilibré pour l'échantillonnage uniforme des classes."""
        logger.info("Création de l'index équilibré...")
        
        cursor = self.conn.cursor()
        
        # Créer une table temporaire pour l'index équilibré
        cursor.execute("DROP TABLE IF EXISTS balanced_index")
        cursor.execute("""
            CREATE TEMPORARY TABLE balanced_index (
                idx INTEGER PRIMARY KEY,
                sample_id INTEGER,
                class_name TEXT
            )
        """)
        
        # Remplir l'index équilibré
        idx = 0
        for class_name, info in self.class_info.items():
            # Obtenir tous les échantillons de cette classe
            cursor.execute(
                "SELECT id FROM audio_samples WHERE class_name = ? LIMIT ?",
                (class_name, self.max_samples_per_class)
            )
            sample_ids = [row[0] for row in cursor.fetchall()]
            
            # Si moins d'échantillons que max_samples_per_class, dupliquer
            while len(sample_ids) < self.max_samples_per_class:
                sample_ids.extend(sample_ids[:min(len(sample_ids), 
                                                 self.max_samples_per_class - len(sample_ids))])
            
            # Insérer dans l'index équilibré
            for sample_id in sample_ids[:self.max_samples_per_class]:
                cursor.execute(
                    "INSERT INTO balanced_index (idx, sample_id, class_name) VALUES (?, ?, ?)",
                    (idx, sample_id, class_name)
                )
                idx += 1
        
        self.conn.commit()
        self.balanced_samples = idx
        logger.info(f"Index équilibré créé: {self.balanced_samples} échantillons")
        cursor.close()
    
    def __len__(self) -> int:
        """Retourne le nombre d'échantillons."""
        if self.balance_classes:
            return self.balanced_samples
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Charge un échantillon avec gestion mémoire stricte.
        """
        # Vérifier le cache
        if idx in self.cache:
            spectrogram, label_idx = self.cache[idx]
            return spectrogram.clone(), label_idx
        
        # Obtenir les infos de l'échantillon
        cursor = self.conn.cursor()
        
        if self.balance_classes:
            cursor.execute("""
                SELECT s.* FROM audio_samples s
                JOIN balanced_index b ON s.id = b.sample_id
                WHERE b.idx = ?
            """, (idx,))
        else:
            cursor.execute("SELECT * FROM audio_samples WHERE id = ?", (idx + 1,))
        
        row = cursor.fetchone()
        cursor.close()
        
        if row is None:
            raise IndexError(f"Index {idx} hors limites")
        
        # Construire le chemin audio
        audio_path = self.audio_root / row['file_path']
        label = row['class_name']
        label_idx = self.class_to_idx[label]
        
        # Générer le spectrogramme
        spectrogram = self._generate_spectrogram_memory_efficient(audio_path)
        
        # Appliquer l'augmentation si demandé
        if self.augment:
            spectrogram = self._augment_spectrogram(spectrogram)
        
        # Appliquer les transformations
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # Gérer le cache avec limite de taille
        self._manage_cache(idx, spectrogram, label_idx)
        
        return spectrogram, label_idx
    
    def _generate_spectrogram_memory_efficient(self, audio_path: Path) -> torch.Tensor:
        """
        Génère un spectrogramme avec gestion mémoire stricte.
        """
        try:
            # Charger seulement la durée nécessaire
            metadata = torchaudio.info(str(audio_path))
            duration_samples = int(self.duration * metadata.sample_rate)
            
            # Charger l'audio avec limite de durée
            waveform, sr = torchaudio.load(
                str(audio_path),
                num_frames=duration_samples
            )
            
            # Convertir en mono
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Rééchantillonner si nécessaire
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                del resampler  # Libérer la mémoire
            
            # Créer le spectrogramme mel avec float16 pour économiser la mémoire
            mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.n_mels,
                f_min=self.config.fmin,
                f_max=self.config.fmax
            )
            
            mel_spec = mel_transform(waveform)
            
            # Convertir en dB
            db_transform = T.AmplitudeToDB(top_db=self.config.top_db)
            mel_spec_db = db_transform(mel_spec)
            
            # Normaliser
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
            
            # Convertir en 3 canaux pour le modèle
            if mel_spec_db.dim() == 3:
                mel_spec_db = mel_spec_db.squeeze(0)
            mel_spec_db = mel_spec_db.unsqueeze(0).repeat(3, 1, 1)
            
            # Libérer la mémoire
            del waveform, mel_spec, mel_transform, db_transform
            gc.collect()
            
            return mel_spec_db.to(torch.float16)  # Utiliser float16
            
        except Exception as e:
            logger.error(f"Erreur génération spectrogramme {audio_path}: {e}")
            # Retourner un spectrogramme vide en cas d'erreur
            return torch.zeros((3, self.n_mels, 345), dtype=torch.float16)
    
    def _augment_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Applique des augmentations légères au spectrogramme."""
        # SpecAugment simple
        if random.random() < 0.5:
            # Masquage temporel
            time_mask = T.TimeMasking(time_mask_param=30)
            spectrogram = time_mask(spectrogram)
        
        if random.random() < 0.5:
            # Masquage fréquentiel
            freq_mask = T.FrequencyMasking(freq_mask_param=20)
            spectrogram = freq_mask(spectrogram)
        
        return spectrogram
    
    def _manage_cache(self, idx: int, spectrogram: torch.Tensor, label_idx: int):
        """Gère le cache avec limite de taille."""
        if len(self.cache) >= self.max_cache_size:
            # Supprimer les éléments les plus anciens
            remove_count = len(self.cache) // 4
            remove_keys = list(self.cache.keys())[:remove_count]
            for key in remove_keys:
                del self.cache[key]
        
        self.cache[idx] = (spectrogram.clone(), label_idx)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calcule les poids de classe pour la loss pondérée."""
        counts = [info['count'] for info in self.class_info.values()]
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        weights = weights / weights.sum() * len(weights)
        return weights
    
    def close(self):
        """Ferme la connexion à la base de données."""
        self.conn.close()
    
    def __del__(self):
        """Destructeur pour s'assurer que la connexion est fermée."""
        if hasattr(self, 'conn'):
            self.conn.close()


def create_index_database(audio_dir: Path, output_db: str, extensions: List[str] = ['.wav', '.mp3']):
    """
    Crée une base de données SQLite d'index pour les fichiers audio.
    
    Args:
        audio_dir: Répertoire contenant les fichiers audio organisés par classe
        output_db: Chemin de sortie pour la base SQLite
        extensions: Extensions de fichiers à indexer
    """
    logger.info(f"Création de l'index pour {audio_dir}...")
    
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()
    
    # Créer la table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audio_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            class_name TEXT NOT NULL,
            duration REAL,
            file_size INTEGER
        )
    """)
    
    # Créer les index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_class ON audio_samples(class_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_path ON audio_samples(file_path)")
    
    # Scanner les fichiers
    sample_count = 0
    class_count = 0
    
    for class_dir in sorted(audio_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        class_samples = 0
        
        for audio_file in class_dir.iterdir():
            # Ignorer les fichiers cachés macOS et autres fichiers système
            if audio_file.name.startswith('._') or audio_file.name == '.DS_Store':
                continue
                
            if audio_file.suffix.lower() in extensions:
                relative_path = audio_file.relative_to(audio_dir)
                
                # Vérifier que le fichier est valide
                try:
                    file_size = audio_file.stat().st_size
                    if file_size == 0:
                        logger.warning(f"Fichier vide ignoré: {relative_path}")
                        continue
                except Exception as e:
                    logger.warning(f"Impossible d'accéder au fichier {relative_path}: {e}")
                    continue
                
                # Essayer d'obtenir la durée
                duration = None
                try:
                    info = torchaudio.info(str(audio_file))
                    duration = info.num_frames / info.sample_rate
                except Exception as e:
                    logger.debug(f"Impossible d'obtenir la durée de {relative_path}: {e}")
                
                cursor.execute("""
                    INSERT INTO audio_samples (file_path, class_name, duration, file_size)
                    VALUES (?, ?, ?, ?)
                """, (str(relative_path), class_name, duration, file_size))
                
                class_samples += 1
                sample_count += 1
                
                if sample_count % 10000 == 0:
                    logger.info(f"Indexé {sample_count} fichiers...")
                    conn.commit()
        
        if class_samples > 0:
            class_count += 1
            logger.info(f"Classe '{class_name}': {class_samples} échantillons")
    
    conn.commit()
    
    # Statistiques finales
    cursor.execute("SELECT COUNT(DISTINCT class_name) as classes, COUNT(*) as samples FROM audio_samples")
    stats = cursor.fetchone()
    
    logger.info(f"\nIndex créé avec succès!")
    logger.info(f"Total: {stats[1]} échantillons, {stats[0]} classes")
    logger.info(f"Base de données sauvegardée: {output_db}")
    
    conn.close()


def create_scalable_data_loaders(
    index_db: str,
    audio_root: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.1,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    Crée les DataLoaders scalables pour train/val/test.
    """
    # TODO: Implémenter le split train/val/test directement dans SQLite
    # Pour l'instant, utiliser un seul dataset
    
    dataset = AudioDatasetScalable(
        index_db=index_db,
        audio_root=audio_root,
        augment=True,
        **dataset_kwargs
    )
    
    # Créer un sampler équilibré
    if dataset.balance_classes:
        # Utiliser l'échantillonnage uniforme par défaut
        sampler = None
        shuffle = True
    else:
        # Pour un dataset non équilibré, utiliser WeightedRandomSampler
        weights = dataset.get_class_weights()
        sample_weights = torch.zeros(len(dataset))
        
        # Cette partie nécessiterait une optimisation pour éviter de charger tout en mémoire
        # Pour l'instant, on utilise le shuffle simple
        sampler = None
        shuffle = True
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,  # Désactivé pour économiser la mémoire
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    
    return {'train': loader}


if __name__ == "__main__":
    # Test de création d'index
    import argparse
    
    parser = argparse.ArgumentParser(description="Créer un index SQLite pour un dataset audio")
    parser.add_argument('audio_dir', type=str, help="Répertoire contenant les fichiers audio")
    parser.add_argument('--output-db', type=str, default='audio_index.db', 
                       help="Fichier de sortie SQLite")
    
    args = parser.parse_args()
    
    create_index_database(
        audio_dir=Path(args.audio_dir),
        output_db=args.output_db
    )