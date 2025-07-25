"""
Module d'augmentation des données audio pour NightScan
Implémente diverses techniques d'augmentation pour améliorer la robustesse du modèle
"""

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Optional, Union
import random
from scipy import signal
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class AudioAugmentation:
    """
    Classe pour l'augmentation des données audio et spectrogrammes.
    """
    
    def __init__(self, sample_rate: int = 22050, augmentation_params: dict = None):
        """
        Initialise l'augmenteur audio.
        
        Args:
            sample_rate: Taux d'échantillonnage
            augmentation_params: Paramètres d'augmentation personnalisés
        """
        self.sample_rate = sample_rate
        
        # Paramètres par défaut
        self.params = {
            'time_mask_max': 30,
            'freq_mask_max': 20,
            'n_time_masks': 2,
            'n_freq_masks': 2,
            'noise_level': 0.005,
            'time_shift_max': 0.2,
            'speed_change': 0.1,
            'pitch_shift': 2,
            'mixup_alpha': 0.2,
            'mixup_prob': 0.5,
            'phase_shift_range': np.pi/4,  # ±π/4 radians pour le décalage de phase
        }
        
        if augmentation_params:
            self.params.update(augmentation_params)
    
    def add_noise(self, waveform: torch.Tensor, noise_level: Optional[float] = None) -> torch.Tensor:
        """
        Ajoute du bruit gaussien au signal audio.
        
        Args:
            waveform: Signal audio (C, T)
            noise_level: Niveau de bruit (utilise le paramètre par défaut si None)
            
        Returns:
            Signal avec bruit ajouté
        """
        if noise_level is None:
            noise_level = self.params['noise_level']
        
        if noise_level > 0:
            noise = torch.randn_like(waveform) * noise_level
            return waveform + noise
        return waveform
    
    def time_shift(self, waveform: torch.Tensor, shift_max: Optional[float] = None) -> torch.Tensor:
        """
        Décale le signal dans le temps (circular shift).
        
        Args:
            waveform: Signal audio (C, T)
            shift_max: Décalage maximum en proportion de la longueur
            
        Returns:
            Signal décalé
        """
        if shift_max is None:
            shift_max = self.params['time_shift_max']
        
        if shift_max > 0:
            shift = int(random.uniform(-shift_max, shift_max) * waveform.shape[-1])
            return torch.roll(waveform, shifts=shift, dims=-1)
        return waveform
    
    def change_speed(self, waveform: torch.Tensor, speed_factor: Optional[float] = None) -> torch.Tensor:
        """
        Change la vitesse du signal (time stretching).
        
        Args:
            waveform: Signal audio (C, T)
            speed_factor: Facteur de changement de vitesse
            
        Returns:
            Signal avec vitesse modifiée
        """
        if speed_factor is None:
            max_change = self.params['speed_change']
            speed_factor = random.uniform(1 - max_change, 1 + max_change)
        
        if speed_factor != 1.0:
            # Utiliser la transformation de vitesse de torchaudio
            transform = T.TimeStretch(fixed_rate=speed_factor)
            # Créer un spectrogramme complexe pour TimeStretch
            n_fft = 2048
            hop_length = 512
            spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
            complex_spec = spec_transform(waveform)
            
            # Appliquer time stretch
            stretched_spec = transform(complex_spec)
            
            # Inverser pour obtenir le signal audio
            inverse_transform = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
            return inverse_transform(stretched_spec)
        return waveform
    
    def pitch_shift(self, waveform: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """
        Change la hauteur du signal.
        
        Args:
            waveform: Signal audio (C, T)
            n_steps: Nombre de demi-tons de décalage
            
        Returns:
            Signal avec hauteur modifiée
        """
        if n_steps is None:
            max_shift = self.params['pitch_shift']
            n_steps = random.randint(-max_shift, max_shift)
        
        if n_steps != 0:
            # Calculer le facteur de changement de fréquence
            factor = 2.0 ** (n_steps / 12.0)
            
            # Utiliser resample pour simuler un pitch shift
            # Note: Ce n'est pas un vrai pitch shift mais une approximation
            new_sample_rate = int(self.sample_rate * factor)
            
            # Resample
            if new_sample_rate != self.sample_rate:
                resampler = T.Resample(self.sample_rate, new_sample_rate)
                shifted = resampler(waveform)
                
                # Ramener à la longueur originale
                if shifted.shape[-1] > waveform.shape[-1]:
                    shifted = shifted[..., :waveform.shape[-1]]
                else:
                    pad = waveform.shape[-1] - shifted.shape[-1]
                    shifted = torch.nn.functional.pad(shifted, (0, pad))
                
                return shifted
        return waveform
    
    def phase_shift(self, waveform: torch.Tensor, phase_shift: Optional[float] = None) -> torch.Tensor:
        """
        Applique un décalage de phase au signal audio en utilisant STFT/iSTFT.
        
        Args:
            waveform: Signal audio (C, T)
            phase_shift: Décalage de phase en radians (utilise le paramètre par défaut si None)
            
        Returns:
            Signal avec phase décalée
        """
        if phase_shift is None:
            max_shift = self.params['phase_shift_range']
            phase_shift = random.uniform(-max_shift, max_shift)
        
        if phase_shift != 0:
            # Paramètres STFT
            n_fft = 2048
            hop_length = 512
            window = torch.hann_window(n_fft)
            
            # Calculer la STFT avec torch.stft
            stft = torch.stft(
                waveform, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                window=window,
                return_complex=True
            )
            
            # Appliquer le décalage de phase
            # Multiplier par e^(j*phase_shift) pour décaler la phase
            phase_factor = np.exp(1j * phase_shift)
            shifted_stft = stft * phase_factor
            
            # Reconstruire le signal avec iSTFT
            reconstructed = torch.istft(
                shifted_stft, 
                n_fft=n_fft, 
                hop_length=hop_length,
                window=window,
                length=waveform.shape[-1]
            )
            
            # Si le signal est multi-canal, s'assurer que la forme est correcte
            if waveform.dim() > 1 and reconstructed.dim() < waveform.dim():
                reconstructed = reconstructed.unsqueeze(0)
            
            return reconstructed
        
        return waveform
    
    def spec_augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Applique SpecAugment au spectrogramme.
        
        Args:
            spectrogram: Spectrogramme (C, F, T) ou (F, T)
            
        Returns:
            Spectrogramme augmenté
        """
        spec = spectrogram.clone()
        
        # S'assurer que le spectrogramme a 3 dimensions
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        
        # Masquage temporel
        for _ in range(self.params['n_time_masks']):
            t = spec.shape[-1]
            mask_len = random.randint(1, min(self.params['time_mask_max'], t))
            mask_start = random.randint(0, t - mask_len)
            spec[..., mask_start:mask_start + mask_len] = spec.mean()
        
        # Masquage fréquentiel
        for _ in range(self.params['n_freq_masks']):
            f = spec.shape[-2]
            mask_len = random.randint(1, min(self.params['freq_mask_max'], f))
            mask_start = random.randint(0, f - mask_len)
            spec[..., mask_start:mask_start + mask_len, :] = spec.mean()
        
        # Retourner dans la forme originale
        if spectrogram.dim() == 2:
            spec = spec.squeeze(0)
        
        return spec
    
    def mixup(self, waveform1: torch.Tensor, waveform2: torch.Tensor, 
              label1: int, label2: int, alpha: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applique mixup entre deux échantillons.
        
        Args:
            waveform1, waveform2: Signaux audio
            label1, label2: Labels correspondants
            alpha: Paramètre de mixup
            
        Returns:
            Signal mixé et label mixé
        """
        if alpha is None:
            alpha = self.params['mixup_alpha']
        
        if random.random() < self.params['mixup_prob']:
            # Échantillonner lambda depuis la distribution Beta
            lam = np.random.beta(alpha, alpha)
            
            # Mixer les signaux
            mixed_waveform = lam * waveform1 + (1 - lam) * waveform2
            
            # Créer le label mixé (one-hot encoding supposé)
            mixed_label = torch.tensor([lam, 1 - lam])
            
            return mixed_waveform, mixed_label
        
        return waveform1, torch.tensor([1.0, 0.0])
    
    def random_augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applique une augmentation aléatoire au signal audio.
        
        Args:
            waveform: Signal audio
            
        Returns:
            Signal augmenté
        """
        # Liste des augmentations possibles
        augmentations = [
            lambda x: self.add_noise(x),
            lambda x: self.time_shift(x),
            # lambda x: self.change_speed(x),  # Désactivé temporairement - problème de dimensions
            lambda x: self.pitch_shift(x),
            lambda x: self.phase_shift(x),
            lambda x: x  # Pas d'augmentation
        ]
        
        # Choisir et appliquer une augmentation aléatoire
        augmentation = random.choice(augmentations)
        return augmentation(waveform)
    
    def random_augment_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Applique une augmentation aléatoire au spectrogramme.
        
        Args:
            spectrogram: Spectrogramme
            
        Returns:
            Spectrogramme augmenté
        """
        if random.random() < 0.5:
            return self.spec_augment(spectrogram)
        return spectrogram


class AdaptiveAudioAugmentation(AudioAugmentation):
    """
    Classe d'augmentation adaptative qui ajuste l'intensité selon la taille de la classe.
    """
    
    def __init__(self, sample_rate: int = 22050, augmentation_params: dict = None):
        """
        Initialise l'augmenteur adaptatif.
        
        Args:
            sample_rate: Taux d'échantillonnage
            augmentation_params: Paramètres d'augmentation personnalisés
        """
        super().__init__(sample_rate, augmentation_params)
        
        # Paramètres spécifiques pour les classes minoritaires
        self.minority_params = {
            'very_small': {  # < 50 échantillons
                'noise_level': 0.02,  # Plus de bruit (0.5-2%)
                'time_shift_max': 0.3,  # ±30% de décalage temporel
                'pitch_shift': 4,  # ±4 semitones
                'phase_shift_range': np.pi/3,  # ±π/3 radians
                'time_mask_max': 50,
                'freq_mask_max': 30,
                'augmentation_prob': 0.9,  # 90% de probabilité d'augmentation
                'augmentation_multiplier': 5  # 5x augmentation
            },
            'small': {  # 50-250 échantillons
                'noise_level': 0.01,
                'time_shift_max': 0.2,
                'pitch_shift': 2,
                'phase_shift_range': np.pi/5,
                'time_mask_max': 30,
                'freq_mask_max': 20,
                'augmentation_prob': 0.8,
                'augmentation_multiplier': 2  # 2x augmentation
            },
            'large': {  # > 250 échantillons
                'noise_level': 0.005,
                'time_shift_max': 0.2,
                'pitch_shift': 2,
                'phase_shift_range': np.pi/6,
                'time_mask_max': 30,
                'freq_mask_max': 20,
                'augmentation_prob': 0.8,
                'augmentation_multiplier': 1  # Pas de multiplication
            }
        }
        
        # Seuils pour déterminer la catégorie de classe
        self.class_size_thresholds = {
            'very_small': 50,
            'small': 250
        }
    
    def get_class_category(self, n_samples: int) -> str:
        """
        Détermine la catégorie de la classe selon le nombre d'échantillons.
        
        Args:
            n_samples: Nombre d'échantillons dans la classe
            
        Returns:
            Catégorie de la classe ('very_small', 'small', 'large')
        """
        if n_samples < self.class_size_thresholds['very_small']:
            return 'very_small'
        elif n_samples < self.class_size_thresholds['small']:
            return 'small'
        else:
            return 'large'
    
    def set_params_for_class_size(self, n_samples: int):
        """
        Ajuste les paramètres d'augmentation selon la taille de la classe.
        
        Args:
            n_samples: Nombre d'échantillons dans la classe
        """
        category = self.get_class_category(n_samples)
        category_params = self.minority_params[category].copy()
        
        # Retirer les paramètres non-augmentation
        self.augmentation_prob = category_params.pop('augmentation_prob', 0.8)
        self.augmentation_multiplier = category_params.pop('augmentation_multiplier', 1)
        
        # Mettre à jour les paramètres d'augmentation
        self.params.update(category_params)
    
    def get_augmentation_multiplier(self, n_samples: int) -> int:
        """
        Retourne le multiplicateur d'augmentation pour une classe donnée.
        
        Args:
            n_samples: Nombre d'échantillons dans la classe
            
        Returns:
            Multiplicateur d'augmentation
        """
        category = self.get_class_category(n_samples)
        return self.minority_params[category]['augmentation_multiplier']
    
    def should_augment(self, n_samples: int) -> bool:
        """
        Détermine si l'augmentation doit être appliquée selon la probabilité de la classe.
        
        Args:
            n_samples: Nombre d'échantillons dans la classe
            
        Returns:
            True si l'augmentation doit être appliquée
        """
        category = self.get_class_category(n_samples)
        prob = self.minority_params[category]['augmentation_prob']
        return random.random() < prob
    
    def add_noise_adaptive(self, waveform: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Ajoute du bruit avec intensité adaptée à la taille de la classe.
        
        Args:
            waveform: Signal audio
            n_samples: Nombre d'échantillons dans la classe
            
        Returns:
            Signal avec bruit ajouté
        """
        category = self.get_class_category(n_samples)
        noise_level = self.minority_params[category]['noise_level']
        
        # Variation aléatoire pour les classes très petites
        if category == 'very_small':
            noise_level = random.uniform(0.005, noise_level)
        
        return self.add_noise(waveform, noise_level)
    
    def random_augment_waveform_adaptive(self, waveform: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Applique une augmentation adaptative selon la taille de la classe.
        
        Args:
            waveform: Signal audio
            n_samples: Nombre d'échantillons dans la classe
            
        Returns:
            Signal augmenté
        """
        # Configurer les paramètres pour cette taille de classe
        self.set_params_for_class_size(n_samples)
        
        # Vérifier si on doit augmenter
        if not self.should_augment(n_samples):
            return waveform
        
        # Pour les classes très petites, appliquer plusieurs augmentations
        if n_samples < self.class_size_thresholds['very_small']:
            # Appliquer 2-3 augmentations en cascade
            n_augmentations = random.randint(2, 3)
            augmented = waveform
            
            # Liste des augmentations possibles
            augmentation_funcs = [
                lambda x: self.add_noise_adaptive(x, n_samples),
                lambda x: self.time_shift(x),
                lambda x: self.pitch_shift(x),
                lambda x: self.phase_shift(x),
            ]
            
            # Sélectionner et appliquer n augmentations
            selected_augs = random.sample(augmentation_funcs, n_augmentations)
            for aug_func in selected_augs:
                augmented = aug_func(augmented)
            
            return augmented
        else:
            # Pour les autres classes, appliquer une seule augmentation
            return self.random_augment_waveform(waveform)


class PreprocessingPipeline:
    """
    Pipeline de prétraitement audio pour améliorer la qualité des spectrogrammes.
    """
    
    def __init__(self, sample_rate: int = 22050, preprocessing_params: dict = None):
        """
        Initialise le pipeline de prétraitement.
        
        Args:
            sample_rate: Taux d'échantillonnage
            preprocessing_params: Paramètres de prétraitement personnalisés
        """
        self.sample_rate = sample_rate
        
        # Paramètres par défaut
        self.params = {
            'remove_silence': True,
            'silence_threshold': -40,
            'min_silence_duration': 0.1,
            'normalize': True,
            'pre_emphasis': 0.97,
            'use_pcen': False,
        }
        
        if preprocessing_params:
            self.params.update(preprocessing_params)
    
    def pre_emphasis(self, waveform: np.ndarray, coef: float = 0.97) -> np.ndarray:
        """
        Applique un filtre de pré-emphasis pour accentuer les hautes fréquences.
        
        Args:
            waveform: Signal audio
            coef: Coefficient de pré-emphasis
            
        Returns:
            Signal filtré
        """
        emphasized = np.append(waveform[0], waveform[1:] - coef * waveform[:-1])
        return emphasized.copy()
    
    def remove_silence(self, waveform: torch.Tensor, threshold_db: float = -40) -> torch.Tensor:
        """
        Supprime les parties silencieuses du signal.
        
        Args:
            waveform: Signal audio
            threshold_db: Seuil de silence en dB
            
        Returns:
            Signal sans silence
        """
        # Calculer l'énergie en dB
        energy = 20 * torch.log10(torch.abs(waveform) + 1e-10)
        
        # Trouver les parties non-silencieuses
        mask = energy > threshold_db
        
        if mask.any():
            # Trouver les indices de début et fin
            indices = torch.where(mask)[0]
            start = indices[0]
            end = indices[-1] + 1
            
            return waveform[start:end]
        
        return waveform
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalise le signal audio.
        
        Args:
            waveform: Signal audio
            
        Returns:
            Signal normalisé
        """
        # Normalisation par la valeur maximale
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            return waveform / max_val
        return waveform
    
    def apply_bandpass_filter(self, waveform: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
        """
        Applique un filtre passe-bande pour isoler les fréquences d'intérêt.
        
        Args:
            waveform: Signal audio
            fmin: Fréquence minimale
            fmax: Fréquence maximale
            
        Returns:
            Signal filtré
        """
        nyquist = self.sample_rate / 2
        low = fmin / nyquist
        high = fmax / nyquist
        
        # Éviter les erreurs si les fréquences sont hors limites
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        # Créer le filtre Butterworth
        b, a = signal.butter(5, [low, high], btype='band')
        
        # Appliquer le filtre et faire une copie pour éviter les strides négatifs
        filtered = signal.filtfilt(b, a, waveform)
        return filtered.copy()
    
    def process(self, waveform: Union[torch.Tensor, np.ndarray], 
                fmin: Optional[float] = None, fmax: Optional[float] = None) -> torch.Tensor:
        """
        Applique le pipeline complet de prétraitement.
        
        Args:
            waveform: Signal audio
            fmin: Fréquence minimale pour le filtre passe-bande
            fmax: Fréquence maximale pour le filtre passe-bande
            
        Returns:
            Signal prétraité
        """
        # Convertir en numpy si nécessaire
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.numpy()
            is_torch = True
        else:
            waveform_np = waveform
            is_torch = False
        
        # Appliquer pré-emphasis
        if self.params['pre_emphasis'] > 0:
            waveform_np = self.pre_emphasis(waveform_np, self.params['pre_emphasis'])
        
        # Appliquer filtre passe-bande si spécifié
        if fmin is not None and fmax is not None:
            waveform_np = self.apply_bandpass_filter(waveform_np, fmin, fmax)
        
        # Convertir en torch
        waveform = torch.from_numpy(waveform_np).float()
        
        # Supprimer le silence
        if self.params['remove_silence']:
            waveform = self.remove_silence(waveform, self.params['silence_threshold'])
        
        # Normaliser
        if self.params['normalize']:
            waveform = self.normalize_audio(waveform)
        
        return waveform


def create_augmented_dataset(waveforms: list, labels: list, n_augmentations: int = 5) -> Tuple[list, list]:
    """
    Crée un dataset augmenté à partir d'un ensemble de signaux audio.
    
    Args:
        waveforms: Liste de signaux audio
        labels: Liste de labels correspondants
        n_augmentations: Nombre d'augmentations par échantillon
        
    Returns:
        Signaux augmentés et labels correspondants
    """
    augmenter = AudioAugmentation()
    augmented_waveforms = []
    augmented_labels = []
    
    for waveform, label in zip(waveforms, labels):
        # Ajouter l'original
        augmented_waveforms.append(waveform)
        augmented_labels.append(label)
        
        # Ajouter les augmentations
        for _ in range(n_augmentations):
            aug_waveform = augmenter.random_augment_waveform(waveform)
            augmented_waveforms.append(aug_waveform)
            augmented_labels.append(label)
    
    return augmented_waveforms, augmented_labels


if __name__ == "__main__":
    # Test des augmentations
    print("Test du module d'augmentation audio...")
    
    # Créer un signal de test
    sample_rate = 22050
    duration = 3
    t = torch.linspace(0, duration, sample_rate * duration)
    
    # Signal avec plusieurs fréquences (simule un chant d'oiseau)
    waveform = 0.5 * torch.sin(2 * np.pi * 1000 * t)  # 1kHz
    waveform += 0.3 * torch.sin(2 * np.pi * 2000 * t)  # 2kHz
    waveform += 0.2 * torch.sin(2 * np.pi * 3000 * t)  # 3kHz
    
    # Tester les augmentations
    augmenter = AudioAugmentation(sample_rate)
    
    print(f"Signal original: shape={waveform.shape}")
    
    # Tester chaque augmentation
    noisy = augmenter.add_noise(waveform)
    print(f"Avec bruit: shape={noisy.shape}")
    
    shifted = augmenter.time_shift(waveform)
    print(f"Décalé temporellement: shape={shifted.shape}")
    
    # Créer un spectrogramme de test
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
    spectrogram = mel_transform(waveform)
    print(f"\nSpectrogramme original: shape={spectrogram.shape}")
    
    # Tester SpecAugment
    aug_spec = augmenter.spec_augment(spectrogram)
    print(f"Spectrogramme augmenté: shape={aug_spec.shape}")
    
    # Tester le prétraitement
    preprocessor = PreprocessingPipeline(sample_rate)
    processed = preprocessor.process(waveform, fmin=500, fmax=5000)
    print(f"\nSignal prétraité: shape={processed.shape}")
    
    print("\nTests terminés avec succès!")