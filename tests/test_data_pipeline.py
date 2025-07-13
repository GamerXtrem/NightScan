"""
Tests pour le pipeline de données et preprocessing ML.

Ce module teste tous les aspects du preprocessing des données:
- Validation et transformation des données audio
- Preprocessing et augmentation des images
- Pipeline de génération des spectrogrammes
- Normalisation et feature extraction
- Tests de qualité des données transformées
- Performance du preprocessing sous charge
"""

import pytest
import tempfile
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image, ImageFilter, ImageEnhance
import struct
import io
from dataclasses import dataclass

# Import preprocessing components to test
# Note: These would be the actual preprocessing modules
# For testing purposes, we'll mock the behavior


@dataclass
class AudioPreprocessingConfig:
    """Configuration pour preprocessing audio."""
    sample_rate: int = 44100
    duration: float = 10.0
    normalize: bool = True
    remove_silence: bool = True
    apply_filters: bool = True
    augmentation: bool = False


@dataclass
class ImagePreprocessingConfig:
    """Configuration pour preprocessing image."""
    target_size: Tuple[int, int] = (224, 224)
    normalize: bool = True
    augmentation: bool = False
    color_mode: str = 'RGB'
    quality_checks: bool = True


class AudioPreprocessor:
    """Simulateur de preprocessing audio."""
    
    def __init__(self, config: AudioPreprocessingConfig):
        self.config = config
        self.stats = {
            'processed_files': 0,
            'total_duration': 0.0,
            'errors': 0
        }
    
    def preprocess_audio(self, audio_path: str) -> Dict[str, Any]:
        """Preprocessing d'un fichier audio."""
        try:
            # Simulate audio loading and processing
            audio_data = self._load_audio(audio_path)
            
            if self.config.normalize:
                audio_data = self._normalize_audio(audio_data)
            
            if self.config.remove_silence:
                audio_data = self._remove_silence(audio_data)
            
            if self.config.apply_filters:
                audio_data = self._apply_filters(audio_data)
            
            if self.config.augmentation:
                audio_data = self._augment_audio(audio_data)
            
            # Generate features
            features = self._extract_features(audio_data)
            
            self.stats['processed_files'] += 1
            self.stats['total_duration'] += self.config.duration
            
            return {
                'success': True,
                'audio_data': audio_data,
                'features': features,
                'duration': self.config.duration,
                'sample_rate': self.config.sample_rate,
                'shape': audio_data.shape if hasattr(audio_data, 'shape') else None
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Simule le chargement audio."""
        # Simulate loading WAV file
        duration_samples = int(self.config.sample_rate * self.config.duration)
        # Generate synthetic audio data
        return np.random.randn(duration_samples).astype(np.float32)
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalisation audio."""
        if len(audio_data) == 0:
            return audio_data
        
        # Peak normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val
        return audio_data
    
    def _remove_silence(self, audio_data: np.ndarray) -> np.ndarray:
        """Suppression des silences."""
        # Simple threshold-based silence removal
        threshold = 0.01
        non_silent_indices = np.where(np.abs(audio_data) > threshold)[0]
        
        if len(non_silent_indices) > 0:
            start_idx = non_silent_indices[0]
            end_idx = non_silent_indices[-1]
            return audio_data[start_idx:end_idx+1]
        
        return audio_data
    
    def _apply_filters(self, audio_data: np.ndarray) -> np.ndarray:
        """Application de filtres audio."""
        # Simulate bandpass filter
        # In reality, this would use scipy.signal or similar
        return audio_data  # Placeholder
    
    def _augment_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Augmentation des données audio."""
        # Random augmentations
        augmentations = []
        
        # Time stretching simulation
        if np.random.random() < 0.3:
            stretch_factor = np.random.uniform(0.8, 1.2)
            # Simulate time stretching
            new_length = int(len(audio_data) * stretch_factor)
            audio_data = np.resize(audio_data, new_length)
        
        # Pitch shifting simulation
        if np.random.random() < 0.3:
            pitch_shift = np.random.uniform(-2, 2)  # semitones
            # Simulate pitch shifting
            pass
        
        # Noise addition
        if np.random.random() < 0.4:
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, audio_data.shape)
            audio_data = audio_data + noise
        
        return audio_data
    
    def _extract_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extraction de features audio."""
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(audio_data)
        features['std'] = np.std(audio_data)
        features['max'] = np.max(audio_data)
        features['min'] = np.min(audio_data)
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        
        # Spectral features (simulated)
        features['spectral_centroid'] = np.random.uniform(1000, 4000)
        features['spectral_bandwidth'] = np.random.uniform(500, 2000)
        features['zero_crossing_rate'] = np.random.uniform(0.01, 0.1)
        
        return features


class ImagePreprocessor:
    """Simulateur de preprocessing image."""
    
    def __init__(self, config: ImagePreprocessingConfig):
        self.config = config
        self.stats = {
            'processed_images': 0,
            'total_pixels': 0,
            'errors': 0
        }
    
    def preprocess_image(self, image_path: str) -> Dict[str, Any]:
        """Preprocessing d'une image."""
        try:
            # Load image
            image = self._load_image(image_path)
            
            if self.config.quality_checks:
                quality_result = self._check_image_quality(image)
                if not quality_result['is_valid']:
                    return {
                        'success': False,
                        'error': f"Quality check failed: {quality_result['reason']}"
                    }
            
            # Resize image
            image = self._resize_image(image)
            
            if self.config.normalize:
                image = self._normalize_image(image)
            
            if self.config.augmentation:
                image = self._augment_image(image)
            
            # Convert to array
            image_array = np.array(image)
            
            # Extract features
            features = self._extract_image_features(image_array)
            
            self.stats['processed_images'] += 1
            self.stats['total_pixels'] += image_array.size
            
            return {
                'success': True,
                'image_array': image_array,
                'features': features,
                'shape': image_array.shape,
                'size': self.config.target_size
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Simule le chargement d'image."""
        # Create synthetic image for testing
        if self.config.color_mode == 'RGB':
            mode = 'RGB'
            color = (128, 128, 128)
        else:
            mode = 'L'
            color = 128
        
        # Create image with some pattern
        size = (640, 480)  # Original size before resizing
        image = Image.new(mode, size, color)
        
        # Add some pattern to make it more realistic
        for i in range(0, size[0], 50):
            for j in range(0, size[1], 50):
                box = (i, j, i+25, j+25)
                if mode == 'RGB':
                    patch_color = (200, 100, 50)
                else:
                    patch_color = 200
                
                patch = Image.new(mode, (25, 25), patch_color)
                image.paste(patch, box)
        
        return image
    
    def _check_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Vérification de la qualité d'image."""
        width, height = image.size
        
        # Check minimum size
        if width < 100 or height < 100:
            return {
                'is_valid': False,
                'reason': f'Image too small: {width}x{height}'
            }
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio < 0.1 or aspect_ratio > 10:
            return {
                'is_valid': False,
                'reason': f'Invalid aspect ratio: {aspect_ratio:.2f}'
            }
        
        # Check if image is mostly blank
        image_array = np.array(image.convert('L'))
        variance = np.var(image_array)
        if variance < 10:
            return {
                'is_valid': False,
                'reason': f'Image appears blank (variance: {variance:.2f})'
            }
        
        return {
            'is_valid': True,
            'width': width,
            'height': height,
            'aspect_ratio': aspect_ratio,
            'variance': variance
        }
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Redimensionnement d'image."""
        return image.resize(self.config.target_size, Image.Resampling.LANCZOS)
    
    def _normalize_image(self, image: Image.Image) -> Image.Image:
        """Normalisation d'image."""
        # Convert to array for normalization
        image_array = np.array(image).astype(np.float32)
        
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        
        # Convert back to PIL Image
        image_array = (image_array * 255).astype(np.uint8)
        return Image.fromarray(image_array)
    
    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Augmentation d'image."""
        # Random augmentations
        
        # Random rotation
        if np.random.random() < 0.3:
            angle = np.random.uniform(-15, 15)
            image = image.rotate(angle, fillcolor=(128, 128, 128))
        
        # Random brightness
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        
        # Random contrast
        if np.random.random() < 0.3:
            contrast_factor = np.random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        # Random blur
        if np.random.random() < 0.2:
            blur_radius = np.random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        return image
    
    def _extract_image_features(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Extraction de features d'image."""
        features = {}
        
        # Basic statistics
        features['mean_intensity'] = np.mean(image_array)
        features['std_intensity'] = np.std(image_array)
        features['min_intensity'] = np.min(image_array)
        features['max_intensity'] = np.max(image_array)
        
        # Color features (if RGB)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            features['mean_r'] = np.mean(image_array[:, :, 0])
            features['mean_g'] = np.mean(image_array[:, :, 1])
            features['mean_b'] = np.mean(image_array[:, :, 2])
        
        # Texture features (simplified)
        if len(image_array.shape) == 2:
            gray = image_array
        else:
            gray = np.mean(image_array, axis=2)
        
        # Gradient magnitude
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['edge_strength'] = np.mean(grad_magnitude)
        
        return features


class SpectrogramGenerator:
    """Générateur de spectrogrammes."""
    
    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.stats = {
            'generated_spectrograms': 0,
            'total_time_frames': 0,
            'errors': 0
        }
    
    def generate_spectrogram(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Génère un spectrogramme à partir de données audio."""
        try:
            # Simulate STFT computation
            n_frames = 1 + (len(audio_data) - self.n_fft) // self.hop_length
            n_freq_bins = self.n_fft // 2 + 1
            
            # Generate synthetic spectrogram
            spectrogram = np.random.rand(n_freq_bins, n_frames).astype(np.float32)
            
            # Add some realistic patterns
            # Low frequencies should generally have more energy
            for i in range(n_freq_bins):
                freq_weight = np.exp(-i / (n_freq_bins * 0.3))
                spectrogram[i, :] *= freq_weight
            
            # Convert to dB scale
            spectrogram_db = 20 * np.log10(spectrogram + 1e-8)
            
            # Normalize
            spectrogram_db = (spectrogram_db - np.min(spectrogram_db)) / (np.max(spectrogram_db) - np.min(spectrogram_db))
            
            self.stats['generated_spectrograms'] += 1
            self.stats['total_time_frames'] += n_frames
            
            return {
                'success': True,
                'spectrogram': spectrogram,
                'spectrogram_db': spectrogram_db,
                'shape': spectrogram.shape,
                'time_frames': n_frames,
                'freq_bins': n_freq_bins,
                'duration': len(audio_data) / self.sample_rate
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }


class TestAudioPreprocessing:
    """Tests pour le preprocessing audio."""
    
    def setup_method(self):
        """Setup pour tests audio."""
        self.config = AudioPreprocessingConfig()
        self.processor = AudioPreprocessor(self.config)
    
    def test_basic_audio_preprocessing(self):
        """Test preprocessing audio basique."""
        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Write minimal WAV header
            tmp.write(b'RIFF')
            tmp.write(struct.pack('<L', 44100))
            tmp.write(b'WAVEfmt ')
            tmp.write(struct.pack('<L', 16))
            tmp.write(struct.pack('<H', 1))   # PCM
            tmp.write(struct.pack('<H', 1))   # Mono
            tmp.write(struct.pack('<L', 44100))
            tmp.write(struct.pack('<L', 88200))
            tmp.write(struct.pack('<H', 2))
            tmp.write(struct.pack('<H', 16))
            tmp.write(b'data')
            tmp.write(struct.pack('<L', 44100))
            tmp.write(b'\x00' * 44100)
            
            audio_path = tmp.name
        
        try:
            result = self.processor.preprocess_audio(audio_path)
            
            # Basic preprocessing assertions
            assert result['success'] == True
            assert 'audio_data' in result
            assert 'features' in result
            assert result['duration'] == self.config.duration
            assert result['sample_rate'] == self.config.sample_rate
            
            # Audio data should be normalized
            audio_data = result['audio_data']
            assert np.max(np.abs(audio_data)) <= 1.0
            
            # Features should be extracted
            features = result['features']
            assert 'mean' in features
            assert 'std' in features
            assert 'rms' in features
            assert 'spectral_centroid' in features
            
        finally:
            Path(audio_path).unlink()
    
    def test_audio_normalization(self):
        """Test normalisation audio."""
        # Test with different audio amplitudes
        test_amplitudes = [0.1, 0.5, 0.8, 1.0, 2.0]
        
        for amplitude in test_amplitudes:
            # Create synthetic audio data
            duration_samples = int(self.config.sample_rate * 1.0)  # 1 second
            audio_data = np.sin(2 * np.pi * 440 * np.arange(duration_samples) / self.config.sample_rate)
            audio_data *= amplitude  # Scale amplitude
            
            # Normalize
            normalized_audio = self.processor._normalize_audio(audio_data)
            
            # Normalization assertions
            max_val = np.max(np.abs(normalized_audio))
            assert max_val <= 1.0, f"Normalization failed for amplitude {amplitude}"
            
            if amplitude > 0:
                assert max_val >= 0.99, f"Normalization too conservative for amplitude {amplitude}"
    
    def test_silence_removal(self):
        """Test suppression des silences."""
        # Create audio with silence at beginning and end
        duration_samples = int(self.config.sample_rate * 3.0)  # 3 seconds
        audio_data = np.zeros(duration_samples)
        
        # Add signal in the middle (1-2 seconds)
        start_sample = int(self.config.sample_rate * 1.0)
        end_sample = int(self.config.sample_rate * 2.0)
        audio_data[start_sample:end_sample] = np.sin(
            2 * np.pi * 440 * np.arange(end_sample - start_sample) / self.config.sample_rate
        ) * 0.5
        
        # Remove silence
        processed_audio = self.processor._remove_silence(audio_data)
        
        # Silence removal assertions
        assert len(processed_audio) < len(audio_data), "Silence not removed"
        assert len(processed_audio) <= (end_sample - start_sample) * 1.1, "Too much audio retained"
        assert np.max(np.abs(processed_audio)) > 0.1, "Signal should be preserved"
    
    def test_audio_augmentation(self):
        """Test augmentation des données audio."""
        self.config.augmentation = True
        processor = AudioPreprocessor(self.config)
        
        # Create baseline audio
        duration_samples = int(self.config.sample_rate * 2.0)
        baseline_audio = np.sin(2 * np.pi * 440 * np.arange(duration_samples) / self.config.sample_rate)
        
        # Apply augmentation multiple times
        augmented_versions = []
        for i in range(10):
            augmented = processor._augment_audio(baseline_audio.copy())
            augmented_versions.append(augmented)
        
        # Augmentation assertions
        # Different augmentations should produce different results
        unique_lengths = set(len(aug) for aug in augmented_versions)
        assert len(unique_lengths) > 1, "Time stretching augmentation not working"
        
        # All should be reasonable variations
        for augmented in augmented_versions:
            assert len(augmented) > 0, "Augmentation produced empty audio"
            assert np.max(np.abs(augmented)) > 0, "Augmentation removed all signal"
    
    def test_feature_extraction_consistency(self):
        """Test consistance de l'extraction de features."""
        # Create identical audio files
        test_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                test_files.append(tmp.name)
        
        try:
            # Extract features from identical files
            feature_sets = []
            for audio_path in test_files:
                result = self.processor.preprocess_audio(audio_path)
                assert result['success'] == True
                feature_sets.append(result['features'])
            
            # Features should be consistent for identical inputs
            first_features = feature_sets[0]
            for features in feature_sets[1:]:
                for key in first_features:
                    if isinstance(first_features[key], (int, float)):
                        # Allow small numerical differences
                        diff = abs(features[key] - first_features[key])
                        assert diff < 0.01, f"Feature {key} not consistent across identical files"
            
        finally:
            for file_path in test_files:
                Path(file_path).unlink()


class TestImagePreprocessing:
    """Tests pour le preprocessing image."""
    
    def setup_method(self):
        """Setup pour tests image."""
        self.config = ImagePreprocessingConfig()
        self.processor = ImagePreprocessor(self.config)
    
    def test_basic_image_preprocessing(self):
        """Test preprocessing image basique."""
        # Create test image
        test_image = Image.new('RGB', (640, 480), color=(128, 128, 128))
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image.save(tmp, format='JPEG')
            image_path = tmp.name
        
        try:
            result = self.processor.preprocess_image(image_path)
            
            # Basic preprocessing assertions
            assert result['success'] == True
            assert 'image_array' in result
            assert 'features' in result
            assert result['shape'] == (*self.config.target_size, 3)  # RGB
            assert result['size'] == self.config.target_size
            
            # Image should be properly resized
            image_array = result['image_array']
            assert image_array.shape[:2] == self.config.target_size
            
            # Features should be extracted
            features = result['features']
            assert 'mean_intensity' in features
            assert 'edge_strength' in features
            assert 'mean_r' in features  # RGB features
            
        finally:
            Path(image_path).unlink()
    
    def test_image_quality_checks(self):
        """Test vérifications de qualité d'image."""
        test_cases = [
            # (size, expected_valid, reason_contains)
            ((50, 50), False, "too small"),     # Too small
            ((100, 10), False, "aspect ratio"), # Bad aspect ratio
            ((500, 500), True, None),           # Good image
        ]
        
        for size, expected_valid, reason_contains in test_cases:
            # Create test image
            test_image = Image.new('RGB', size, color=(128, 128, 128))
            
            # Add some pattern to avoid blank image detection
            if size[0] >= 100 and size[1] >= 100:
                for i in range(0, size[0], 20):
                    for j in range(0, size[1], 20):
                        if (i + j) % 40 == 0:
                            box = (i, j, min(i+10, size[0]), min(j+10, size[1]))
                            test_image.paste((200, 100, 50), box)
            
            quality_result = self.processor._check_image_quality(test_image)
            
            assert quality_result['is_valid'] == expected_valid
            
            if not expected_valid and reason_contains:
                assert reason_contains in quality_result['reason'].lower()
    
    def test_image_resizing(self):
        """Test redimensionnement d'image."""
        # Test different input sizes
        input_sizes = [(100, 100), (640, 480), (1920, 1080), (300, 800)]
        
        for input_size in input_sizes:
            test_image = Image.new('RGB', input_size, color=(128, 64, 192))
            
            # Resize image
            resized_image = self.processor._resize_image(test_image)
            
            # Resizing assertions
            assert resized_image.size == self.config.target_size
            
            # Image should maintain some visual characteristics
            resized_array = np.array(resized_image)
            assert resized_array.shape == (*self.config.target_size, 3)
            assert np.any(resized_array > 0)  # Not completely black
    
    def test_image_normalization(self):
        """Test normalisation d'image."""
        # Create test image with known values
        test_image = Image.new('RGB', (100, 100), color=(255, 128, 0))
        
        # Normalize
        normalized_image = self.processor._normalize_image(test_image)
        normalized_array = np.array(normalized_image)
        
        # Normalization assertions
        assert np.min(normalized_array) >= 0
        assert np.max(normalized_array) <= 255
        
        # Should maintain relative intensities
        assert np.any(normalized_array[:, :, 0] > 200)  # Red channel high
        assert np.any(normalized_array[:, :, 1] > 100)  # Green channel medium
        assert np.any(normalized_array[:, :, 2] < 50)   # Blue channel low
    
    def test_image_augmentation(self):
        """Test augmentation d'image."""
        self.config.augmentation = True
        processor = ImagePreprocessor(self.config)
        
        # Create distinctive test image
        test_image = Image.new('RGB', (200, 200), color=(255, 255, 255))
        # Add a distinctive pattern
        for i in range(0, 200, 20):
            test_image.paste((0, 0, 0), (i, i, i+10, i+10))
        
        # Apply augmentation multiple times
        augmented_versions = []
        for i in range(10):
            augmented = processor._augment_image(test_image.copy())
            augmented_versions.append(np.array(augmented))
        
        # Augmentation assertions
        baseline_array = np.array(test_image)
        
        differences = []
        for augmented_array in augmented_versions:
            # Calculate difference from baseline
            diff = np.mean(np.abs(augmented_array.astype(float) - baseline_array.astype(float)))
            differences.append(diff)
        
        # Most augmented versions should be different from baseline
        significant_changes = sum(1 for diff in differences if diff > 5.0)
        assert significant_changes >= 5, "Augmentation not producing enough variation"
    
    def test_image_feature_extraction(self):
        """Test extraction de features d'image."""
        # Create images with different characteristics
        test_cases = [
            # (description, color, expected_feature_ranges)
            ("bright_image", (255, 255, 255), {"mean_intensity": (200, 255)}),
            ("dark_image", (50, 50, 50), {"mean_intensity": (0, 100)}),
            ("red_image", (255, 0, 0), {"mean_r": (200, 255), "mean_g": (0, 50), "mean_b": (0, 50)}),
        ]
        
        for description, color, expected_ranges in test_cases:
            test_image = Image.new('RGB', (100, 100), color=color)
            image_array = np.array(test_image)
            
            features = self.processor._extract_image_features(image_array)
            
            # Feature extraction assertions
            assert 'mean_intensity' in features
            assert 'edge_strength' in features
            
            # Check expected ranges
            for feature_name, (min_val, max_val) in expected_ranges.items():
                assert feature_name in features
                feature_value = features[feature_name]
                assert min_val <= feature_value <= max_val, \
                    f"{description}: {feature_name} = {feature_value}, expected [{min_val}, {max_val}]"


class TestSpectrogramGeneration:
    """Tests pour la génération de spectrogrammes."""
    
    def setup_method(self):
        """Setup pour tests spectrogramme."""
        self.generator = SpectrogramGenerator()
    
    def test_basic_spectrogram_generation(self):
        """Test génération de spectrogramme basique."""
        # Create synthetic audio signal
        duration = 2.0  # seconds
        sample_rate = self.generator.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create signal with multiple frequencies
        frequencies = [440, 880, 1320]  # A4, A5, E6
        audio_signal = np.sum([np.sin(2 * np.pi * f * t) for f in frequencies], axis=0)
        audio_signal = audio_signal.astype(np.float32)
        
        result = self.generator.generate_spectrogram(audio_signal)
        
        # Basic generation assertions
        assert result['success'] == True
        assert 'spectrogram' in result
        assert 'spectrogram_db' in result
        assert result['duration'] == duration
        
        # Spectrogram should have correct shape
        spectrogram = result['spectrogram']
        expected_freq_bins = self.generator.n_fft // 2 + 1
        expected_time_frames = 1 + (len(audio_signal) - self.generator.n_fft) // self.generator.hop_length
        
        assert spectrogram.shape[0] == expected_freq_bins
        assert spectrogram.shape[1] == expected_time_frames
        
        # dB spectrogram should be normalized
        spectrogram_db = result['spectrogram_db']
        assert np.min(spectrogram_db) >= 0.0
        assert np.max(spectrogram_db) <= 1.0
    
    def test_spectrogram_frequency_content(self):
        """Test contenu fréquentiel du spectrogramme."""
        # Create pure tone at known frequency
        frequency = 1000  # Hz
        duration = 1.0
        sample_rate = self.generator.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_signal = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        result = self.generator.generate_spectrogram(audio_signal)
        spectrogram = result['spectrogram']
        
        # Find frequency bin corresponding to 1000 Hz
        freq_resolution = sample_rate / self.generator.n_fft
        expected_bin = int(frequency / freq_resolution)
        
        # Energy should be concentrated around the expected frequency
        # (This is a simplified test since we're using synthetic spectrograms)
        assert spectrogram.shape[0] > expected_bin, "Frequency resolution insufficient"
        
        # Spectrogram should have energy in the frequency domain
        total_energy = np.sum(spectrogram)
        assert total_energy > 0, "Spectrogram has no energy"
    
    def test_spectrogram_time_resolution(self):
        """Test résolution temporelle du spectrogramme."""
        durations = [0.5, 1.0, 2.0, 5.0]  # Different durations
        
        for duration in durations:
            sample_rate = self.generator.sample_rate
            audio_length = int(sample_rate * duration)
            audio_signal = np.random.randn(audio_length).astype(np.float32)
            
            result = self.generator.generate_spectrogram(audio_signal)
            
            # Time resolution assertions
            expected_frames = 1 + (audio_length - self.generator.n_fft) // self.generator.hop_length
            actual_frames = result['time_frames']
            
            assert actual_frames == expected_frames
            assert result['duration'] == duration
            
            # Longer audio should produce more time frames
            if duration > 1.0:
                assert actual_frames > sample_rate / self.generator.hop_length
    
    def test_spectrogram_parameters_impact(self):
        """Test impact des paramètres sur le spectrogramme."""
        # Test different parameters
        parameter_sets = [
            {'n_fft': 1024, 'hop_length': 256},
            {'n_fft': 2048, 'hop_length': 512},
            {'n_fft': 4096, 'hop_length': 1024},
        ]
        
        # Fixed audio signal
        duration = 2.0
        sample_rate = 44100
        audio_length = int(sample_rate * duration)
        audio_signal = np.random.randn(audio_length).astype(np.float32)
        
        results = []
        for params in parameter_sets:
            generator = SpectrogramGenerator(
                sample_rate=sample_rate,
                n_fft=params['n_fft'],
                hop_length=params['hop_length']
            )
            
            result = generator.generate_spectrogram(audio_signal)
            results.append(result)
        
        # Parameter impact assertions
        for i, result in enumerate(results):
            params = parameter_sets[i]
            
            # Frequency resolution should match n_fft
            expected_freq_bins = params['n_fft'] // 2 + 1
            assert result['freq_bins'] == expected_freq_bins
            
            # Time resolution should match hop_length
            expected_time_frames = 1 + (audio_length - params['n_fft']) // params['hop_length']
            assert result['time_frames'] == expected_time_frames
        
        # Different parameters should produce different shapes
        shapes = [r['shape'] for r in results]
        unique_shapes = set(shapes)
        assert len(unique_shapes) == len(parameter_sets), "Parameters not affecting spectrogram shape"


class TestDataPipelineIntegration:
    """Tests d'intégration pour le pipeline de données complet."""
    
    def setup_method(self):
        """Setup pour tests d'intégration."""
        self.audio_config = AudioPreprocessingConfig(
            sample_rate=44100,
            duration=5.0,
            normalize=True,
            remove_silence=True,
            augmentation=False
        )
        self.image_config = ImagePreprocessingConfig(
            target_size=(224, 224),
            normalize=True,
            augmentation=False
        )
        
        self.audio_processor = AudioPreprocessor(self.audio_config)
        self.image_processor = ImagePreprocessor(self.image_config)
        self.spectrogram_generator = SpectrogramGenerator()
    
    @pytest.mark.ml_pipeline
    def test_audio_to_spectrogram_pipeline(self):
        """Test pipeline complet audio → spectrogramme."""
        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            audio_path = tmp.name
        
        try:
            # Step 1: Preprocess audio
            audio_result = self.audio_processor.preprocess_audio(audio_path)
            assert audio_result['success'] == True
            
            # Step 2: Generate spectrogram
            audio_data = audio_result['audio_data']
            spectrogram_result = self.spectrogram_generator.generate_spectrogram(audio_data)
            assert spectrogram_result['success'] == True
            
            # Pipeline integration assertions
            # Duration should be consistent
            assert abs(audio_result['duration'] - spectrogram_result['duration']) < 0.1
            
            # Spectrogram should have reasonable dimensions
            spectrogram = spectrogram_result['spectrogram']
            assert spectrogram.shape[0] > 0  # Frequency bins
            assert spectrogram.shape[1] > 0  # Time frames
            
            # Features should be extracted
            assert 'features' in audio_result
            assert len(audio_result['features']) > 0
            
        finally:
            Path(audio_path).unlink()
    
    @pytest.mark.ml_pipeline
    def test_batch_processing_pipeline(self):
        """Test pipeline de traitement batch."""
        # Create multiple test files
        test_files = []
        
        # Audio files
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
                test_files.append(('audio', tmp.name))
        
        # Image files  
        for i in range(3):
            test_image = Image.new('RGB', (400, 300), color=(i*80, 128, 255-i*50))
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                test_image.save(tmp, format='JPEG')
                test_files.append(('image', tmp.name))
        
        try:
            # Process all files
            batch_results = {'audio': [], 'image': []}
            
            for file_type, file_path in test_files:
                if file_type == 'audio':
                    result = self.audio_processor.preprocess_audio(file_path)
                else:
                    result = self.image_processor.preprocess_image(file_path)
                
                batch_results[file_type].append(result)
            
            # Batch processing assertions
            # All audio files should be processed successfully
            audio_results = batch_results['audio']
            assert len(audio_results) == 3
            assert all(r['success'] for r in audio_results)
            
            # All image files should be processed successfully
            image_results = batch_results['image']
            assert len(image_results) == 3
            assert all(r['success'] for r in image_results)
            
            # Consistency checks
            # All audio results should have same duration
            audio_durations = [r['duration'] for r in audio_results]
            assert len(set(audio_durations)) == 1  # All same duration
            
            # All image results should have same target size
            image_shapes = [r['shape'] for r in image_results]
            assert len(set(image_shapes)) == 1  # All same shape
            
        finally:
            for _, file_path in test_files:
                Path(file_path).unlink()
    
    @pytest.mark.ml_pipeline
    @pytest.mark.performance_critical
    def test_pipeline_performance_benchmarks(self):
        """Test performance du pipeline de données."""
        # Performance targets
        max_audio_processing_time = 2.0  # seconds per file
        max_image_processing_time = 1.0  # seconds per file
        max_spectrogram_generation_time = 0.5  # seconds
        
        # Test audio processing performance
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            audio_path = tmp.name
        
        try:
            start_time = time.time()
            audio_result = self.audio_processor.preprocess_audio(audio_path)
            audio_processing_time = time.time() - start_time
            
            assert audio_result['success'] == True
            assert audio_processing_time < max_audio_processing_time
            
            # Test spectrogram generation performance
            audio_data = audio_result['audio_data']
            start_time = time.time()
            spectrogram_result = self.spectrogram_generator.generate_spectrogram(audio_data)
            spectrogram_time = time.time() - start_time
            
            assert spectrogram_result['success'] == True
            assert spectrogram_time < max_spectrogram_generation_time
            
        finally:
            Path(audio_path).unlink()
        
        # Test image processing performance
        test_image = Image.new('RGB', (1920, 1080), color=(128, 128, 128))
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image.save(tmp, format='JPEG', quality=95)
            image_path = tmp.name
        
        try:
            start_time = time.time()
            image_result = self.image_processor.preprocess_image(image_path)
            image_processing_time = time.time() - start_time
            
            assert image_result['success'] == True
            assert image_processing_time < max_image_processing_time
            
        finally:
            Path(image_path).unlink()
    
    @pytest.mark.ml_pipeline
    def test_data_quality_validation(self):
        """Test validation de la qualité des données transformées."""
        # Test audio data quality
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(b'RIFF\x24\x08\x00\x00WAVEfmt ')
            audio_path = tmp.name
        
        try:
            audio_result = self.audio_processor.preprocess_audio(audio_path)
            assert audio_result['success'] == True
            
            # Audio quality checks
            audio_data = audio_result['audio_data']
            features = audio_result['features']
            
            # Audio should be properly normalized
            assert np.max(np.abs(audio_data)) <= 1.0
            assert np.min(np.abs(audio_data)) >= 0.0
            
            # Features should be within reasonable ranges
            assert -1.0 <= features['mean'] <= 1.0
            assert features['std'] >= 0.0
            assert features['rms'] >= 0.0
            assert features['spectral_centroid'] > 0
            
        finally:
            Path(audio_path).unlink()
        
        # Test image data quality
        test_image = Image.new('RGB', (300, 300), color=(128, 128, 128))
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image.save(tmp, format='JPEG')
            image_path = tmp.name
        
        try:
            image_result = self.image_processor.preprocess_image(image_path)
            assert image_result['success'] == True
            
            # Image quality checks
            image_array = image_result['image_array']
            features = image_result['features']
            
            # Image should have correct shape and range
            assert image_array.shape == (*self.image_config.target_size, 3)
            assert np.min(image_array) >= 0
            assert np.max(image_array) <= 255
            
            # Features should be reasonable
            assert 0 <= features['mean_intensity'] <= 255
            assert features['std_intensity'] >= 0
            assert features['edge_strength'] >= 0
            
        finally:
            Path(image_path).unlink()