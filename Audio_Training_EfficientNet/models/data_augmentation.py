"""
Advanced Data Augmentation for Wildlife Audio

Comprehensive data augmentation techniques specifically designed for
wildlife audio spectrograms, including SpecAugment, time stretching,
pitch shifting, and mixup.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path
import librosa
import soundfile as sf
from pydub import AudioSegment
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SpecAugment(nn.Module):
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    
    Applied to mel-spectrograms with frequency and time masking.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 30,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        freq_mask_prob: float = 0.5,
        time_mask_prob: float = 0.5
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        
        self.freq_mask = T.FrequencyMasking(freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param)
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to a spectrogram.
        
        Args:
            spectrogram: Input spectrogram tensor (C, F, T)
            
        Returns:
            Augmented spectrogram tensor
        """
        if not self.training:
            return spectrogram
        
        # Apply frequency masking
        if random.random() < self.freq_mask_prob:
            for _ in range(self.num_freq_masks):
                spectrogram = self.freq_mask(spectrogram)
        
        # Apply time masking
        if random.random() < self.time_mask_prob:
            for _ in range(self.num_time_masks):
                spectrogram = self.time_mask(spectrogram)
        
        return spectrogram


class TimeStretchAugment:
    """
    Time stretching augmentation for audio signals.
    
    Changes the speed of audio without affecting pitch.
    """
    
    def __init__(
        self,
        stretch_factors: Tuple[float, float] = (0.8, 1.2),
        prob: float = 0.5
    ):
        self.stretch_factors = stretch_factors
        self.prob = prob
    
    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply time stretching to audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio
            
        Returns:
            Time-stretched audio array
        """
        if random.random() > self.prob:
            return audio
        
        # Random stretch factor
        stretch_factor = random.uniform(*self.stretch_factors)
        
        # Apply time stretching using librosa
        stretched_audio = librosa.effects.time_stretch(
            audio, 
            rate=stretch_factor
        )
        
        return stretched_audio


class PitchShiftAugment:
    """
    Pitch shifting augmentation for audio signals.
    
    Changes the pitch of audio without affecting duration.
    """
    
    def __init__(
        self,
        pitch_shift_range: Tuple[float, float] = (-2.0, 2.0),
        prob: float = 0.5
    ):
        self.pitch_shift_range = pitch_shift_range
        self.prob = prob
    
    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply pitch shifting to audio.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate of the audio
            
        Returns:
            Pitch-shifted audio array
        """
        if random.random() > self.prob:
            return audio
        
        # Random pitch shift in semitones
        pitch_shift = random.uniform(*self.pitch_shift_range)
        
        # Apply pitch shifting using librosa
        shifted_audio = librosa.effects.pitch_shift(
            audio,
            sr=sample_rate,
            n_steps=pitch_shift
        )
        
        return shifted_audio


class AddGaussianNoise:
    """
    Add Gaussian noise to audio signals.
    """
    
    def __init__(
        self,
        noise_factor_range: Tuple[float, float] = (0.005, 0.02),
        prob: float = 0.3
    ):
        self.noise_factor_range = noise_factor_range
        self.prob = prob
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to audio.
        
        Args:
            audio: Input audio array
            
        Returns:
            Noisy audio array
        """
        if random.random() > self.prob:
            return audio
        
        # Random noise factor
        noise_factor = random.uniform(*self.noise_factor_range)
        
        # Generate and add noise
        noise = np.random.normal(0, noise_factor, audio.shape)
        noisy_audio = audio + noise
        
        # Clip to prevent overflow
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        
        return noisy_audio


class MixupAugment:
    """
    Mixup augmentation for spectrograms and labels.
    
    Combines two samples linearly with a random mixing coefficient.
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self, 
        spec1: torch.Tensor, 
        spec2: torch.Tensor, 
        label1: int, 
        label2: int,
        num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup to two spectrograms and their labels.
        
        Args:
            spec1: First spectrogram
            spec2: Second spectrogram
            label1: First label (integer)
            label2: Second label (integer)
            num_classes: Total number of classes
            
        Returns:
            Mixed spectrogram and mixed label (one-hot encoded)
        """
        if random.random() > self.prob:
            # Convert to one-hot
            label1_onehot = F.one_hot(torch.tensor(label1), num_classes).float()
            return spec1, label1_onehot
        
        # Sample mixing coefficient
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Mix spectrograms
        mixed_spec = lam * spec1 + (1 - lam) * spec2
        
        # Mix labels (one-hot encoded)
        label1_onehot = F.one_hot(torch.tensor(label1), num_classes).float()
        label2_onehot = F.one_hot(torch.tensor(label2), num_classes).float()
        mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot
        
        return mixed_spec, mixed_label


class VolumeAugment:
    """
    Volume augmentation for audio signals.
    """
    
    def __init__(
        self,
        volume_range: Tuple[float, float] = (0.5, 2.0),
        prob: float = 0.4
    ):
        self.volume_range = volume_range
        self.prob = prob
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply volume augmentation to audio.
        
        Args:
            audio: Input audio array
            
        Returns:
            Volume-adjusted audio array
        """
        if random.random() > self.prob:
            return audio
        
        # Random volume factor
        volume_factor = random.uniform(*self.volume_range)
        
        # Apply volume change
        augmented_audio = audio * volume_factor
        
        # Clip to prevent overflow
        augmented_audio = np.clip(augmented_audio, -1.0, 1.0)
        
        return augmented_audio


class AudioAugmentationPipeline:
    """
    Complete audio augmentation pipeline.
    
    Combines multiple augmentation techniques in a configurable pipeline.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        use_time_stretch: bool = True,
        use_pitch_shift: bool = True,
        use_noise: bool = True,
        use_volume: bool = True,
        use_spec_augment: bool = True,
        use_mixup: bool = True,
        spec_augment_config: Optional[Dict[str, Any]] = None,
        mixup_alpha: float = 0.2
    ):
        self.sample_rate = sample_rate
        self.use_time_stretch = use_time_stretch
        self.use_pitch_shift = use_pitch_shift
        self.use_noise = use_noise
        self.use_volume = use_volume
        self.use_spec_augment = use_spec_augment
        self.use_mixup = use_mixup
        
        # Initialize audio augmentations
        if use_time_stretch:
            self.time_stretch = TimeStretchAugment()
        
        if use_pitch_shift:
            self.pitch_shift = PitchShiftAugment()
        
        if use_noise:
            self.noise_augment = AddGaussianNoise()
        
        if use_volume:
            self.volume_augment = VolumeAugment()
        
        # Initialize spectrogram augmentations
        if use_spec_augment:
            if spec_augment_config is None:
                spec_augment_config = {}
            self.spec_augment = SpecAugment(**spec_augment_config)
        
        if use_mixup:
            self.mixup = MixupAugment(alpha=mixup_alpha)
    
    def augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio-level augmentations.
        
        Args:
            audio: Input audio array
            
        Returns:
            Augmented audio array
        """
        # Apply augmentations in sequence
        if self.use_time_stretch:
            audio = self.time_stretch(audio, self.sample_rate)
        
        if self.use_pitch_shift:
            audio = self.pitch_shift(audio, self.sample_rate)
        
        if self.use_noise:
            audio = self.noise_augment(audio)
        
        if self.use_volume:
            audio = self.volume_augment(audio)
        
        return audio
    
    def augment_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply spectrogram-level augmentations.
        
        Args:
            spectrogram: Input spectrogram tensor
            
        Returns:
            Augmented spectrogram tensor
        """
        if self.use_spec_augment:
            spectrogram = self.spec_augment(spectrogram)
        
        return spectrogram
    
    def apply_mixup(
        self,
        spec1: torch.Tensor,
        spec2: torch.Tensor,
        label1: int,
        label2: int,
        num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation.
        
        Args:
            spec1: First spectrogram
            spec2: Second spectrogram
            label1: First label
            label2: Second label
            num_classes: Total number of classes
            
        Returns:
            Mixed spectrogram and mixed label
        """
        if self.use_mixup:
            return self.mixup(spec1, spec2, label1, label2, num_classes)
        else:
            # Return original with one-hot encoded label
            label1_onehot = F.one_hot(torch.tensor(label1), num_classes).float()
            return spec1, label1_onehot


def create_augmentation_pipeline(
    config: Dict[str, Any],
    sample_rate: int = 22050
) -> AudioAugmentationPipeline:
    """
    Create an augmentation pipeline from configuration.
    
    Args:
        config: Configuration dictionary
        sample_rate: Audio sample rate
        
    Returns:
        Configured augmentation pipeline
    """
    return AudioAugmentationPipeline(
        sample_rate=sample_rate,
        use_time_stretch=config.get("use_time_stretch", True),
        use_pitch_shift=config.get("use_pitch_shift", True),
        use_noise=config.get("use_noise", True),
        use_volume=config.get("use_volume", True),
        use_spec_augment=config.get("use_spec_augment", True),
        use_mixup=config.get("use_mixup", True),
        spec_augment_config=config.get("spec_augment_config", {}),
        mixup_alpha=config.get("mixup_alpha", 0.2)
    )


if __name__ == "__main__":
    # Test the augmentation pipeline
    print("Testing Audio Augmentation Pipeline...")
    
    # Create a sample audio signal
    sample_rate = 22050
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Create augmentation pipeline
    pipeline = AudioAugmentationPipeline(sample_rate=sample_rate)
    
    # Test audio augmentation
    augmented_audio = pipeline.augment_audio(audio)
    print(f"Original audio shape: {audio.shape}")
    print(f"Augmented audio shape: {augmented_audio.shape}")
    
    # Test spectrogram augmentation
    spectrogram = torch.randn(1, 128, 128)  # Dummy spectrogram
    augmented_spec = pipeline.augment_spectrogram(spectrogram)
    print(f"Original spectrogram shape: {spectrogram.shape}")
    print(f"Augmented spectrogram shape: {augmented_spec.shape}")
    
    # Test mixup
    spec1 = torch.randn(1, 128, 128)
    spec2 = torch.randn(1, 128, 128)
    mixed_spec, mixed_label = pipeline.apply_mixup(spec1, spec2, 0, 1, 6)
    print(f"Mixed spectrogram shape: {mixed_spec.shape}")
    print(f"Mixed label shape: {mixed_label.shape}")
    
    print("All tests passed!")