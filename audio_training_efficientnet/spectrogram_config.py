"""
Configuration unifiée pour la génération des spectrogrammes NightScan
Basé sur les meilleures pratiques pour la classification audio d'animaux sauvages
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Paramètres optimisés pour la classification audio d'animaux
SPECTROGRAM_PARAMS = {
    # Paramètres de base recommandés par la recherche
    'sample_rate': 22050,      # Taux d'échantillonnage standard
    'n_fft': 2048,            # Taille FFT (résolution fréquentielle)
    'hop_length': 512,        # Pas entre fenêtres (n_fft/4)
    'n_mels': 128,            # Nombre de bandes mel
    'fmin': 50,               # Fréquence minimale (filtre le bruit <50Hz)
    'fmax': 11000,            # Fréquence maximale (Nyquist/2)
    'top_db': 80,             # Plage dynamique en dB
    'duration': 8,            # Durée en secondes
    'window': 'hann',         # Fenêtre d'analyse
    'center': True,           # Centrer les fenêtres
    'pad_mode': 'constant',   # Mode de padding
    'power': 2.0,             # Puissance pour le spectrogramme (2.0 = puissance)
    'ref': 1.0,               # Référence pour la conversion en dB
    'amin': 1e-10,            # Valeur minimale pour éviter log(0)
}

# Paramètres spécifiques selon le type d'animal
ANIMAL_FREQ_RANGES = {
    'bat': {
        'fmin': 15000,        # Chauves-souris: ultrasons
        'fmax': 100000,       
        'sample_rate': 192000, # Taux élevé pour capturer les ultrasons
        'n_mels': 256,        # Plus de résolution pour les hautes fréquences
    },
    'owl': {
        'fmin': 200,          # Chouettes: basses fréquences
        'fmax': 4000,
        'sample_rate': 16000,  # Taux suffisant pour les basses fréquences
    },
    'bird_song': {
        'fmin': 1000,         # Chants d'oiseaux: moyennes fréquences
        'fmax': 8000,
        'sample_rate': 22050,
    },
    'mammal': {
        'fmin': 100,          # Mammifères: large spectre
        'fmax': 5000,
        'sample_rate': 22050,
    },
    'amphibian': {
        'fmin': 200,          # Amphibiens: souvent basses fréquences
        'fmax': 5000,
        'sample_rate': 16000,
    },
    'insect': {
        'fmin': 2000,         # Insectes: hautes fréquences
        'fmax': 15000,
        'sample_rate': 44100,
    },
    'general': {              # Paramètres par défaut
        'fmin': 50,
        'fmax': 11000,
        'sample_rate': 22050,
    }
}

# Paramètres pour l'augmentation des données
AUGMENTATION_PARAMS = {
    # SpecAugment
    'time_mask_max': 30,      # Masquage temporel max (frames)
    'freq_mask_max': 20,      # Masquage fréquentiel max (bandes mel)
    'n_time_masks': 2,        # Nombre de masques temporels
    'n_freq_masks': 2,        # Nombre de masques fréquentiels
    
    # Variations audio
    'noise_level': 0.005,     # Niveau de bruit gaussien
    'time_shift_max': 0.2,    # Décalage temporel max (proportion)
    'speed_change': 0.1,      # Variation de vitesse (+/- 10%)
    'pitch_shift': 2,         # Variation de pitch (+/- 2 demi-tons)
    
    # Mixup
    'mixup_alpha': 0.2,       # Paramètre alpha pour mixup
    'mixup_prob': 0.5,        # Probabilité d'appliquer mixup
}

# Paramètres de prétraitement
PREPROCESSING_PARAMS = {
    'remove_silence': True,           # Supprimer les silences
    'silence_threshold': -40,         # Seuil de silence en dB
    'min_silence_duration': 0.1,      # Durée minimale de silence (secondes)
    'normalize': True,                # Normaliser l'audio
    'pre_emphasis': 0.97,             # Coefficient de pré-emphasis
    'use_pcen': False,                # Utiliser PCEN au lieu de log-mel
    'pcen_params': {
        'gain': 0.98,
        'bias': 2,
        'power': 0.5,
        'time_constant': 0.4,
        'eps': 1e-6
    }
}

# Configuration pour l'augmentation adaptative des classes minoritaires
MAX_SAMPLES_PER_CLASS = 500  # Limite maximale d'échantillons par classe après augmentation

MINORITY_CLASS_CONFIG = {
    'thresholds': {
        'very_small': 50,      # Classes avec < 50 échantillons
        'small': 250,          # Classes avec 50-250 échantillons
    },
    'augmentation_multipliers': {
        'very_small': 5,       # 5x augmentation pour < 50 échantillons (mais limité à 500 total)
        'small': 2,            # 2x augmentation pour 50-250 échantillons (mais limité à 500 total)
        'large': 1,            # Pas de multiplication pour > 250 échantillons
    },
    'augmentation_params': {
        'very_small': {
            'noise_level': 0.02,         # Plus de bruit (0.5-2%)
            'time_shift_max': 0.3,       # ±30% de décalage temporel
            'pitch_shift': 4,            # ±4 semitones
            'phase_shift_range': np.pi/3, # ±π/3 radians
            'time_mask_max': 50,
            'freq_mask_max': 30,
            'n_time_masks': 3,
            'n_freq_masks': 3,
        },
        'small': {
            'noise_level': 0.01,
            'time_shift_max': 0.2,
            'pitch_shift': 2,
            'phase_shift_range': np.pi/5,
            'time_mask_max': 30,
            'freq_mask_max': 20,
            'n_time_masks': 2,
            'n_freq_masks': 2,
        },
        'large': AUGMENTATION_PARAMS,  # Utiliser les paramètres standard
    }
}

@dataclass
class SpectrogramConfig:
    """Configuration complète pour la génération de spectrogrammes."""
    
    # Paramètres de base
    sample_rate: int = SPECTROGRAM_PARAMS['sample_rate']
    n_fft: int = SPECTROGRAM_PARAMS['n_fft']
    hop_length: int = SPECTROGRAM_PARAMS['hop_length']
    n_mels: int = SPECTROGRAM_PARAMS['n_mels']
    fmin: float = SPECTROGRAM_PARAMS['fmin']
    fmax: float = SPECTROGRAM_PARAMS['fmax']
    top_db: float = SPECTROGRAM_PARAMS['top_db']
    duration: float = SPECTROGRAM_PARAMS['duration']
    window: str = SPECTROGRAM_PARAMS['window']
    center: bool = SPECTROGRAM_PARAMS['center']
    pad_mode: str = SPECTROGRAM_PARAMS['pad_mode']
    power: float = SPECTROGRAM_PARAMS['power']
    
    # Type d'animal (pour adaptation automatique)
    animal_type: Optional[str] = None
    
    # Augmentation
    use_augmentation: bool = False
    augmentation_params: Dict = None
    
    # Prétraitement
    preprocessing_params: Dict = None
    
    # Support des classes minoritaires
    enable_minority_support: bool = True
    
    def __post_init__(self):
        """Ajuste les paramètres selon le type d'animal."""
        if self.animal_type and self.animal_type in ANIMAL_FREQ_RANGES:
            animal_params = ANIMAL_FREQ_RANGES[self.animal_type]
            for key, value in animal_params.items():
                setattr(self, key, value)
        
        if self.augmentation_params is None:
            self.augmentation_params = AUGMENTATION_PARAMS.copy()
        
        if self.preprocessing_params is None:
            self.preprocessing_params = PREPROCESSING_PARAMS.copy()
    
    def get_spectrogram_shape(self) -> Tuple[int, int]:
        """Calcule la forme du spectrogramme résultant."""
        n_frames = int(np.ceil(self.duration * self.sample_rate / self.hop_length))
        return (self.n_mels, n_frames)
    
    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire."""
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'top_db': self.top_db,
            'duration': self.duration,
            'window': self.window,
            'center': self.center,
            'pad_mode': self.pad_mode,
            'power': self.power,
            'animal_type': self.animal_type,
            'enable_minority_support': self.enable_minority_support,
        }
    
    def get_augmentation_for_class_size(self, n_samples: int) -> Dict:
        """
        Retourne les paramètres d'augmentation adaptés à la taille de classe.
        
        Args:
            n_samples: Nombre d'échantillons dans la classe
            
        Returns:
            Paramètres d'augmentation adaptés
        """
        if self.enable_minority_support:
            return get_augmentation_params_for_class(n_samples)
        else:
            return self.augmentation_params


def get_config_for_animal(animal_type: str = 'general') -> SpectrogramConfig:
    """
    Retourne une configuration optimisée pour un type d'animal.
    
    Args:
        animal_type: Type d'animal ('bat', 'owl', 'bird_song', etc.)
        
    Returns:
        SpectrogramConfig optimisée
    """
    return SpectrogramConfig(animal_type=animal_type)


def get_minority_class_category(n_samples: int) -> str:
    """
    Détermine la catégorie d'une classe selon son nombre d'échantillons.
    
    Args:
        n_samples: Nombre d'échantillons dans la classe
        
    Returns:
        Catégorie ('very_small', 'small', 'large')
    """
    thresholds = MINORITY_CLASS_CONFIG['thresholds']
    if n_samples < thresholds['very_small']:
        return 'very_small'
    elif n_samples < thresholds['small']:
        return 'small'
    else:
        return 'large'


def get_augmentation_params_for_class(n_samples: int) -> Dict:
    """
    Retourne les paramètres d'augmentation adaptés à la taille de la classe.
    
    Args:
        n_samples: Nombre d'échantillons dans la classe
        
    Returns:
        Dictionnaire des paramètres d'augmentation
    """
    category = get_minority_class_category(n_samples)
    return MINORITY_CLASS_CONFIG['augmentation_params'][category].copy()


def get_augmentation_multiplier(n_samples: int) -> int:
    """
    Retourne le multiplicateur d'augmentation pour une classe.
    Limité pour ne pas dépasser MAX_SAMPLES_PER_CLASS.
    
    Args:
        n_samples: Nombre d'échantillons dans la classe
        
    Returns:
        Multiplicateur d'augmentation
    """
    if n_samples >= MAX_SAMPLES_PER_CLASS:
        return 1  # Pas d'augmentation si déjà >= 500 échantillons
    
    category = get_minority_class_category(n_samples)
    base_multiplier = MINORITY_CLASS_CONFIG['augmentation_multipliers'][category]
    
    # Calculer le multiplicateur maximal pour ne pas dépasser 500
    max_multiplier = MAX_SAMPLES_PER_CLASS // n_samples
    
    # Retourner le minimum entre le multiplicateur de base et le max calculé
    return min(base_multiplier, max_multiplier)


def validate_audio_params(sample_rate: int, duration: float, n_samples: int) -> bool:
    """
    Valide la cohérence des paramètres audio.
    
    Args:
        sample_rate: Taux d'échantillonnage
        duration: Durée en secondes
        n_samples: Nombre d'échantillons
        
    Returns:
        True si les paramètres sont cohérents
    """
    expected_samples = int(sample_rate * duration)
    tolerance = sample_rate * 0.1  # 10% de tolérance
    return abs(n_samples - expected_samples) < tolerance


# Configuration par défaut exportée
DEFAULT_CONFIG = SpectrogramConfig()

if __name__ == "__main__":
    # Test des configurations
    print("Configuration par défaut:")
    print(DEFAULT_CONFIG.to_dict())
    print(f"Forme du spectrogramme: {DEFAULT_CONFIG.get_spectrogram_shape()}")
    
    print("\nConfiguration pour chauves-souris:")
    bat_config = get_config_for_animal('bat')
    print(bat_config.to_dict())
    
    print("\nConfiguration pour chouettes:")
    owl_config = get_config_for_animal('owl')
    print(owl_config.to_dict())