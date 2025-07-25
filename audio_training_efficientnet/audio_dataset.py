"""
Dataset PyTorch pour charger les données audio NightScan
Supporte le chargement de fichiers WAV et de spectrogrammes .npy
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import json
from typing import Optional, Tuple, Dict, List
import logging
import random

# Ajouter les chemins pour les imports
sys.path.append(str(Path(__file__).parent.parent))
from nightscan_pi.Program.spectrogram_gen import wav_to_spec

# Importer la configuration et l'augmentation
from spectrogram_config import SpectrogramConfig, get_config_for_animal
from audio_augmentation import AudioAugmentation, AdaptiveAudioAugmentation, PreprocessingPipeline

logger = logging.getLogger(__name__)


class AudioSpectrogramDataset(Dataset):
    """
    Dataset pour charger les spectrogrammes audio ou générer à partir de WAV.
    """
    
    def __init__(self, 
                 csv_file: Path,
                 audio_dir: Path,
                 spectrogram_dir: Optional[Path] = None,
                 classes_json: Optional[Path] = None,
                 config: Optional[SpectrogramConfig] = None,
                 animal_type: str = 'general',
                 transform=None,
                 cache_spectrograms: bool = True,
                 augment: bool = False,
                 augmentation_params: Optional[dict] = None,
                 adaptive_augment: bool = True,
                 enable_oversampling: bool = False):
        """
        Args:
            csv_file: Fichier CSV avec colonnes 'filename' et 'label'
            audio_dir: Répertoire racine contenant les fichiers audio
            spectrogram_dir: Répertoire pour sauvegarder/charger les spectrogrammes
            classes_json: Fichier JSON contenant les informations sur les classes
            config: Configuration de spectrogramme (optionnelle)
            animal_type: Type d'animal pour configuration automatique
            transform: Transformations à appliquer
            cache_spectrograms: Si True, sauvegarde les spectrogrammes générés
            augment: Si True, applique l'augmentation des données
            augmentation_params: Paramètres d'augmentation personnalisés
            adaptive_augment: Si True, utilise l'augmentation adaptative selon la taille de classe
            enable_oversampling: Si True, active l'oversampling pour les classes < 500 échantillons
        """
        self.csv_file = Path(csv_file)
        self.audio_dir = Path(audio_dir)
        self.spectrogram_dir = Path(spectrogram_dir) if spectrogram_dir else None
        self.transform = transform
        self.cache_spectrograms = cache_spectrograms
        self.augment = augment
        self.adaptive_augment = adaptive_augment
        self.enable_oversampling = enable_oversampling
        
        # Configuration du spectrogramme
        if config is None:
            self.config = get_config_for_animal(animal_type)
        else:
            self.config = config
        
        # Récupérer les paramètres depuis la config
        self.sample_rate = self.config.sample_rate
        self.duration = self.config.duration
        self.n_mels = self.config.n_mels
        
        # Charger le CSV
        self.data_df = pd.read_csv(self.csv_file)
        
        # Compter les échantillons par classe
        self.class_counts = self.data_df['label'].value_counts().to_dict()
        
        # Charger les informations sur les classes
        if classes_json and classes_json.exists():
            with open(classes_json, 'r', encoding='utf-8') as f:
                class_info = json.load(f)
            self.class_names = class_info['class_names']
            self.class_to_idx = class_info['class_to_idx']
        else:
            # Déduire les classes depuis le CSV
            unique_labels = sorted(self.data_df['label'].unique())
            self.class_names = unique_labels
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        self.num_classes = len(self.class_names)
        
        # Initialiser l'augmentation
        if self.augment:
            if self.adaptive_augment:
                self.augmenter = AdaptiveAudioAugmentation(self.sample_rate, augmentation_params)
                logger.info("Utilisation de l'augmentation adaptative")
                # Logger les statistiques par classe
                for class_name in self.class_names:
                    count = self.class_counts.get(class_name, 0)
                    category = self.augmenter.get_class_category(count)
                    multiplier = self.augmenter.get_augmentation_multiplier(count)
                    logger.info(f"Classe '{class_name}': {count} échantillons - Catégorie: {category}, Multiplicateur: {multiplier}x")
            else:
                self.augmenter = AudioAugmentation(self.sample_rate, augmentation_params)
            self.preprocessor = PreprocessingPipeline(self.sample_rate, self.config.preprocessing_params)
        else:
            self.augmenter = None
            self.preprocessor = PreprocessingPipeline(self.sample_rate, self.config.preprocessing_params)
        
        # Appliquer l'oversampling si demandé
        if self.enable_oversampling:
            self._apply_oversampling()
        
        # Créer le répertoire de spectrogrammes si nécessaire
        if self.spectrogram_dir and self.cache_spectrograms:
            self.spectrogram_dir.mkdir(parents=True, exist_ok=True)
            # Créer les sous-répertoires pour les spectrogrammes
            (self.spectrogram_dir / "original").mkdir(parents=True, exist_ok=True)
            (self.spectrogram_dir / "augmented").mkdir(parents=True, exist_ok=True)
            logger.info(f"Répertoires de cache créés: {self.spectrogram_dir}/[original|augmented]")
        
        logger.info(f"Dataset initialisé: {len(self.data_df)} échantillons, {self.num_classes} classes")
    
    def _apply_oversampling(self):
        """
        Applique l'oversampling pour les classes avec moins de 500 échantillons.
        Limite stricte à 500 échantillons maximum par classe.
        """
        MAX_SAMPLES_PER_CLASS = 500
        new_rows = []
        
        for class_name in self.class_names:
            class_df = self.data_df[self.data_df['label'] == class_name]
            count = len(class_df)
            
            if count < MAX_SAMPLES_PER_CLASS and count > 0:
                # Calculer combien de fois dupliquer selon le multiplicateur
                multiplier = self.augmenter.get_augmentation_multiplier(count)
                target_count = min(count * multiplier, MAX_SAMPLES_PER_CLASS)
                n_duplicates = (target_count // count) - 1
                
                if n_duplicates > 0:
                    # Dupliquer les échantillons
                    for _ in range(n_duplicates):
                        new_rows.append(class_df)
                    
                    # Ajouter des échantillons aléatoires pour atteindre exactement target_count
                    remaining = target_count - count * (n_duplicates + 1)
                    if remaining > 0:
                        sampled_rows = class_df.sample(n=min(remaining, len(class_df)), replace=True)
                        new_rows.append(sampled_rows)
                    
                    logger.info(f"Oversampling classe '{class_name}': {count} -> {target_count} échantillons (max: {MAX_SAMPLES_PER_CLASS})")
        
        if new_rows:
            # Combiner avec le dataframe original
            self.data_df = pd.concat([self.data_df] + new_rows, ignore_index=True)
            # Mélanger les données
            self.data_df = self.data_df.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info(f"Dataset après oversampling: {len(self.data_df)} échantillons")
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Charge un échantillon.
        
        Returns:
            Tuple (spectrogramme, label_idx)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Obtenir les informations de l'échantillon
        row = self.data_df.iloc[idx]
        audio_path = self.audio_dir / row['filename']
        label = row['label']
        label_idx = self.class_to_idx[label]
        
        # Déterminer si on veut appliquer l'augmentation
        should_augment = False
        aug_index = 0
        
        if self.augment and self.augmenter:
            class_count = self.class_counts.get(label, 1)
            if self.adaptive_augment:
                should_augment = self.augmenter.should_augment(class_count)
            else:
                should_augment = random.random() < 0.8
            
            if should_augment:
                # Générer un index d'augmentation pseudo-aléatoire basé sur l'époque
                # Cela permet de varier les augmentations entre époques
                aug_index = random.randint(0, 999)
        
        # Essayer de charger un spectrogramme existant
        spectrogram = None
        if self.spectrogram_dir:
            spec_path = self._get_spectrogram_path(row['filename'], augmented=should_augment, aug_index=aug_index)
            if spec_path.exists():
                try:
                    spectrogram = np.load(spec_path)
                except Exception as e:
                    logger.warning(f"Erreur chargement spectrogramme {spec_path}: {e}")
        
        # Si pas de spectrogramme en cache, le générer
        if spectrogram is None:
            if should_augment:
                spectrogram = self._generate_augmented_spectrogram(audio_path, label)
            else:
                spectrogram = self._generate_spectrogram(audio_path)
            
            # Sauvegarder dans le cache
            if self.spectrogram_dir and self.cache_spectrograms:
                spec_path = self._get_spectrogram_path(row['filename'], augmented=should_augment, aug_index=aug_index)
                spec_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    np.save(spec_path, spectrogram)
                    logger.debug(f"Spectrogramme sauvegardé: {spec_path}")
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde du spectrogramme {spec_path}: {e}")
        
        # Convertir en tensor PyTorch
        spectrogram = self._prepare_spectrogram(spectrogram)
        
        # Appliquer les transformations
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label_idx
    
    def _get_spectrogram_path(self, audio_filename: str, augmented: bool = False, aug_index: int = 0) -> Path:
        """Génère le chemin du spectrogramme correspondant."""
        audio_path = Path(audio_filename)
        
        # Extraire le nom de la classe (dossier parent immédiat du fichier)
        # Gérer les chemins absolus en prenant seulement le nom du dossier parent
        class_name = audio_path.parent.name
        
        if augmented:
            # Créer un nom unique pour les spectrogrammes augmentés
            spec_filename = f"{audio_path.stem}_aug{aug_index:03d}.npy"
        else:
            spec_filename = audio_path.with_suffix('.npy').name
        
        # Organiser par sous-dossiers: original/augmented -> classe -> fichier
        subdir = "augmented" if augmented else "original"
        spec_dir = self.spectrogram_dir / subdir / class_name
        
        return spec_dir / spec_filename
    
    def _generate_spectrogram(self, audio_path: Path) -> np.ndarray:
        """Génère un spectrogramme mel depuis un fichier audio."""
        # Charger l'audio
        waveform, orig_sr = torchaudio.load(str(audio_path))
        
        # Convertir en mono si nécessaire
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Rééchantillonner si nécessaire
        if orig_sr != self.sample_rate:
            resampler = T.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Prétraitement
        if self.preprocessor:
            waveform_np = waveform.numpy().squeeze()
            waveform_np = self.preprocessor.process(
                waveform_np, 
                self.config.fmin, 
                self.config.fmax
            ).numpy()
            # Faire une copie pour éviter les problèmes de strides négatifs
            waveform_np = waveform_np.copy()
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        
        # Ajuster la durée
        target_length = int(self.sample_rate * self.duration)
        if waveform.shape[1] < target_length:
            # Padding
            pad = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.shape[1] > target_length:
            # Truncate
            waveform = waveform[:, :target_length]
        
        # Créer le spectrogramme mel avec la configuration unifiée
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            window_fn=torch.hann_window if self.config.window == 'hann' else torch.hamming_window,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            power=self.config.power
        )
        
        mel_spec = mel_spectrogram(waveform)
        
        # Convertir en dB
        mel_spec_db = T.AmplitudeToDB(top_db=self.config.top_db)(mel_spec)
        
        return mel_spec_db.squeeze(0).numpy()
    
    def _generate_augmented_spectrogram(self, audio_path: Path, label: Optional[str] = None) -> np.ndarray:
        """Génère un spectrogramme avec augmentation."""
        # Charger l'audio
        waveform, orig_sr = torchaudio.load(str(audio_path))
        
        # Convertir en mono si nécessaire
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Rééchantillonner si nécessaire
        if orig_sr != self.sample_rate:
            resampler = T.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Appliquer l'augmentation audio
        if self.augmenter:
            if self.adaptive_augment and label is not None:
                # Utiliser l'augmentation adaptative
                class_count = self.class_counts.get(label, 1)
                waveform = self.augmenter.random_augment_waveform_adaptive(waveform, class_count)
            else:
                # Utiliser l'augmentation standard
                waveform = self.augmenter.random_augment_waveform(waveform)
        
        # Prétraitement
        if self.preprocessor:
            waveform_np = waveform.numpy().squeeze()
            waveform_np = self.preprocessor.process(
                waveform_np, 
                self.config.fmin, 
                self.config.fmax
            ).numpy()
            # Faire une copie pour éviter les problèmes de strides négatifs
            waveform_np = waveform_np.copy()
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        
        # Ajuster la durée
        target_length = int(self.sample_rate * self.duration)
        if waveform.shape[1] < target_length:
            pad = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        
        # Créer le spectrogramme mel
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            f_min=self.config.fmin,
            f_max=self.config.fmax,
            window_fn=torch.hann_window if self.config.window == 'hann' else torch.hamming_window,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
            power=self.config.power
        )
        
        mel_spec = mel_spectrogram(waveform)
        
        # Convertir en dB
        mel_spec_db = T.AmplitudeToDB(top_db=self.config.top_db)(mel_spec)
        
        # Appliquer SpecAugment si disponible
        if self.augmenter and random.random() < 0.5:
            mel_spec_db = self.augmenter.spec_augment(mel_spec_db)
        
        return mel_spec_db.squeeze(0).numpy()
    
    def _prepare_spectrogram(self, spectrogram: np.ndarray) -> torch.Tensor:
        """
        Prépare le spectrogramme pour le modèle.
        Normalise et convertit en format RGB.
        """
        # Faire une copie pour éviter les problèmes de strides
        spectrogram = spectrogram.copy()
        
        # Normaliser
        spec_norm = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)
        
        # Convertir en RGB (3 canaux)
        if len(spec_norm.shape) == 2:
            spec_rgb = np.stack([spec_norm, spec_norm, spec_norm], axis=0)
        else:
            spec_rgb = spec_norm
        
        # Faire une copie finale pour s'assurer que les strides sont corrects
        spec_rgb = spec_rgb.copy()
        
        # Appliquer une augmentation supplémentaire sur le spectrogramme si demandé
        if self.augment and self.transform is None and random.random() < 0.3:
            spec_tensor = torch.FloatTensor(spec_rgb)
            if self.augmenter:
                spec_tensor = self.augmenter.spec_augment(spec_tensor)
            return spec_tensor
        
        return torch.FloatTensor(spec_rgb)
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calcule les poids de classe pour gérer le déséquilibre.
        """
        class_counts = self.data_df['label'].value_counts()
        total_samples = len(self.data_df)
        
        weights = []
        for class_name in self.class_names:
            count = class_counts.get(class_name, 1)  # Éviter division par zéro
            weight = total_samples / (len(self.class_names) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)


def create_data_loaders(csv_dir: Path,
                       audio_dir: Path,
                       batch_size: int = 32,
                       num_workers: int = 8,
                       spectrogram_dir: Optional[Path] = None,
                       config: Optional[SpectrogramConfig] = None,
                       animal_type: str = 'general',
                       augment_train: bool = True,
                       adaptive_augment: bool = True,
                       enable_oversampling: bool = False,
                       persistent_workers: bool = True,
                       prefetch_factor: int = 2,
                       **dataset_kwargs) -> Dict[str, DataLoader]:
    """
    Crée les DataLoaders pour train, val et test.
    
    Args:
        csv_dir: Répertoire contenant les CSV
        audio_dir: Répertoire racine des fichiers audio
        batch_size: Taille du batch
        num_workers: Nombre de workers pour le chargement (défaut: 8)
        spectrogram_dir: Répertoire pour les spectrogrammes
        config: Configuration de spectrogramme
        animal_type: Type d'animal pour configuration automatique
        augment_train: Si True, applique l'augmentation sur le dataset d'entraînement
        adaptive_augment: Si True, utilise l'augmentation adaptative selon la taille de classe
        enable_oversampling: Si True, active l'oversampling pour les classes < 500 échantillons
        persistent_workers: Si True, garde les workers en vie entre les epochs (défaut: True)
        prefetch_factor: Nombre de batches à précharger par worker (défaut: 2)
        **dataset_kwargs: Arguments additionnels pour AudioSpectrogramDataset
        
    Returns:
        Dict avec les DataLoaders 'train', 'val', 'test'
    """
    loaders = {}
    
    # Charger les informations sur les classes
    classes_json = csv_dir / 'classes.json'
    
    for split in ['train', 'val', 'test']:
        csv_file = csv_dir / f'{split}.csv'
        
        if not csv_file.exists():
            logger.warning(f"Fichier {csv_file} non trouvé, split {split} ignoré")
            continue
        
        # Créer le dataset avec augmentation pour train uniquement
        augment = augment_train and (split == 'train')
        
        # L'oversampling n'est appliqué que sur le dataset d'entraînement
        apply_oversampling = enable_oversampling and (split == 'train')
        
        dataset = AudioSpectrogramDataset(
            csv_file=csv_file,
            audio_dir=audio_dir,
            spectrogram_dir=spectrogram_dir,
            classes_json=classes_json,
            config=config,
            animal_type=animal_type,
            augment=augment,
            adaptive_augment=adaptive_augment,
            enable_oversampling=apply_oversampling,
            **dataset_kwargs
        )
        
        # Créer le DataLoader avec optimisations
        shuffle = (split == 'train')
        
        # Paramètres optimisés pour le DataLoader
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available() or torch.backends.mps.is_available(),
            'persistent_workers': persistent_workers and num_workers > 0,
            'prefetch_factor': prefetch_factor if num_workers > 0 else None,
            'drop_last': split == 'train',  # Évite les problèmes avec les petits batches finaux
        }
        
        loader = DataLoader(dataset, **loader_kwargs)
        
        loaders[split] = loader
        logger.info(f"DataLoader {split}: {len(dataset)} échantillons, {len(loader)} batches")
        if num_workers > 0:
            logger.info(f"  - Workers: {num_workers} (persistants: {persistent_workers}, prefetch: {prefetch_factor})")
        if spectrogram_dir:
            logger.info(f"  - Cache spectrogrammes: {spectrogram_dir}")
    
    return loaders


def test_dataset():
    """Fonction de test du dataset."""
    print("=== Test du Dataset Audio avec Configuration Unifiée ===\n")
    
    # Chemins de test
    csv_file = Path("data/processed/csv/train.csv")
    audio_dir = Path("audio_data")
    
    if not csv_file.exists():
        print("Fichier CSV de test non trouvé. Créez d'abord les données avec prepare_audio_data.py")
        return
    
    # Créer un dataset
    dataset = AudioSpectrogramDataset(
        csv_file=csv_file,
        audio_dir=audio_dir,
        spectrogram_dir=Path("data/spectrograms")
    )
    
    print(f"Dataset créé: {len(dataset)} échantillons")
    print(f"Classes: {dataset.class_names}")
    
    # Tester le chargement d'un échantillon
    if len(dataset) > 0:
        spec, label_idx = dataset[0]
        print(f"\nPremier échantillon:")
        print(f"- Forme du spectrogramme: {spec.shape}")
        print(f"- Label index: {label_idx}")
        print(f"- Label name: {dataset.class_names[label_idx]}")
        
    # Tester le DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_idx, (specs, labels) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"- Spectrogrammes: {specs.shape}")
        print(f"- Labels: {labels}")
        break


if __name__ == "__main__":
    test_dataset()