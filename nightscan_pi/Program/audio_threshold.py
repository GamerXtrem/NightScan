"""Audio threshold detection system for NightScanPi with configurable sensitivity."""
from __future__ import annotations

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import pyaudio

logger = logging.getLogger(__name__)


@dataclass
class AudioThresholdConfig:
    """Configuration for audio threshold detection."""
    threshold_db: float = -30.0          # dB threshold for detection
    min_duration_ms: int = 100           # Minimum duration for valid detection
    max_silence_ms: int = 2000           # Maximum silence before reset
    sample_window_ms: int = 50           # Analysis window size
    adaptive_enabled: bool = True        # Enable adaptive threshold
    adaptive_factor: float = 0.1         # Adaptation speed (0.0-1.0)
    noise_floor_db: float = -60.0       # Estimated noise floor
    auto_calibrate: bool = True          # Auto-calibrate on start
    environment_preset: str = "balanced" # quiet, balanced, noisy


class AudioThresholdDetector:
    """Real-time audio threshold detection with configurable sensitivity."""
    
    # Environment presets
    PRESETS = {
        "quiet": {
            "threshold_db": -40.0,
            "min_duration_ms": 50,
            "adaptive_factor": 0.05,
            "description": "Environnement silencieux (intérieur, nuit)"
        },
        "balanced": {
            "threshold_db": -30.0,
            "min_duration_ms": 100,
            "adaptive_factor": 0.1,
            "description": "Environnement modéré (jardin, campagne)"
        },
        "noisy": {
            "threshold_db": -20.0,
            "min_duration_ms": 200,
            "adaptive_factor": 0.2,
            "description": "Environnement bruyant (ville, route)"
        }
    }
    
    def __init__(self, config_file: Path = None):
        self.config_file = config_file or Path.home() / ".nightscan" / "audio_threshold.json"
        self.config = self._load_config()
        
        # Detection state
        self.is_detecting = False
        self.detection_start_time = 0
        self.last_detection_time = 0
        self.adaptive_noise_floor = self.config.noise_floor_db
        self.recent_levels = []
        
        # Audio setup
        self.audio_config = None
        self.pyaudio_instance = None
        self.stream = None
        
        # Initialize audio system
        self._setup_audio()
    
    def _load_config(self) -> AudioThresholdConfig:
        """Load configuration from file or create default."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    config = AudioThresholdConfig(**data)
                    logger.info(f"Loaded threshold config from {self.config_file}")
                    return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        
        # Return default config
        config = AudioThresholdConfig()
        self._save_config(config)
        return config
    
    def _save_config(self, config: AudioThresholdConfig = None) -> None:
        """Save configuration to file."""
        if config is None:
            config = self.config
        
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2)
            logger.info(f"Saved threshold config to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _setup_audio(self) -> None:
        """Setup audio system for threshold detection."""
        try:
            from .audio_capture import get_audio_config
            self.audio_config = get_audio_config()
            
            # Use shorter chunks for real-time detection
            self.chunk_size = max(256, self.audio_config['chunk_size'] // 4)
            self.sample_rate = self.audio_config['sample_rate']
            
            logger.info(f"Audio threshold setup: {self.sample_rate}Hz, {self.chunk_size} samples")
            
        except Exception as e:
            logger.error(f"Failed to setup audio for threshold detection: {e}")
            # Fallback configuration
            self.chunk_size = 256
            self.sample_rate = 16000
    
    def update_config(self, **kwargs) -> bool:
        """Update configuration parameters."""
        try:
            updated = False
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    updated = True
                    logger.info(f"Updated threshold config: {key} = {value}")
            
            if updated:
                self._save_config()
                
                # Apply environment preset if specified
                if 'environment_preset' in kwargs:
                    self.apply_preset(kwargs['environment_preset'])
            
            return updated
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False
    
    def apply_preset(self, preset_name: str) -> bool:
        """Apply environment preset configuration."""
        if preset_name not in self.PRESETS:
            logger.error(f"Unknown preset: {preset_name}")
            return False
        
        try:
            preset = self.PRESETS[preset_name]
            self.config.threshold_db = preset["threshold_db"]
            self.config.min_duration_ms = preset["min_duration_ms"]
            self.config.adaptive_factor = preset["adaptive_factor"]
            self.config.environment_preset = preset_name
            
            self._save_config()
            logger.info(f"Applied preset '{preset_name}': {preset['description']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply preset {preset_name}: {e}")
            return False
    
    def calculate_db_level(self, audio_data: bytes) -> float:
        """Calculate dB level from audio data."""
        try:
            # Convert bytes to numpy array
            if self.audio_config and self.audio_config.get('is_respeaker') and len(audio_data) > 0:
                # Handle stereo data (2 channels, 16-bit)
                samples = np.frombuffer(audio_data, dtype=np.int16)
                if len(samples) % 2 == 0:
                    # Convert stereo to mono by averaging
                    samples = samples.reshape(-1, 2).mean(axis=1)
            else:
                # Mono data
                samples = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(samples) == 0:
                return -80.0  # Very quiet
            
            # Calculate RMS (Root Mean Square)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            
            # Convert to dB (relative to max 16-bit value)
            if rms > 0:
                db_level = 20 * np.log10(rms / 32768.0)
            else:
                db_level = -80.0  # Silence
            
            return float(db_level)
            
        except Exception as e:
            logger.debug(f"Error calculating dB level: {e}")
            return -80.0
    
    def update_adaptive_threshold(self, current_db: float) -> None:
        """Update adaptive threshold based on recent audio levels."""
        if not self.config.adaptive_enabled:
            return
        
        # Keep recent levels for adaptation
        self.recent_levels.append(current_db)
        if len(self.recent_levels) > 100:  # Keep last 100 samples
            self.recent_levels.pop(0)
        
        # Update noise floor estimate
        if len(self.recent_levels) >= 10:
            # Use 10th percentile as noise floor estimate
            noise_estimate = np.percentile(self.recent_levels, 10)
            
            # Slowly adapt noise floor
            alpha = self.config.adaptive_factor
            self.adaptive_noise_floor = (
                alpha * noise_estimate + 
                (1 - alpha) * self.adaptive_noise_floor
            )
    
    def is_audio_detected(self, audio_data: bytes) -> Tuple[bool, Dict]:
        """Check if audio exceeds threshold and return detection info."""
        current_time = time.time() * 1000  # ms
        current_db = self.calculate_db_level(audio_data)
        
        # Update adaptive threshold
        self.update_adaptive_threshold(current_db)
        
        # Determine effective threshold
        effective_threshold = self.config.threshold_db
        if self.config.adaptive_enabled:
            # Adjust threshold based on noise floor
            margin = 10.0  # dB above noise floor
            adaptive_threshold = self.adaptive_noise_floor + margin
            effective_threshold = max(self.config.threshold_db, adaptive_threshold)
        
        # Check if current level exceeds threshold
        above_threshold = current_db > effective_threshold
        
        detection_info = {
            "current_db": current_db,
            "threshold_db": effective_threshold,
            "noise_floor_db": self.adaptive_noise_floor,
            "above_threshold": above_threshold,
            "is_detecting": False,
            "detection_duration_ms": 0
        }
        
        if above_threshold:
            if not self.is_detecting:
                # Start new detection
                self.is_detecting = True
                self.detection_start_time = current_time
                logger.debug(f"Audio detection started: {current_db:.1f}dB > {effective_threshold:.1f}dB")
            
            self.last_detection_time = current_time
            detection_duration = current_time - self.detection_start_time
            
            detection_info.update({
                "is_detecting": True,
                "detection_duration_ms": detection_duration
            })
            
            # Check if detection meets minimum duration
            if detection_duration >= self.config.min_duration_ms:
                detection_info["valid_detection"] = True
                return True, detection_info
            else:
                detection_info["valid_detection"] = False
                
        else:
            # Below threshold
            if self.is_detecting:
                silence_duration = current_time - self.last_detection_time
                
                # Check if silence exceeds maximum allowed
                if silence_duration >= self.config.max_silence_ms:
                    # End detection
                    self.is_detecting = False
                    total_duration = current_time - self.detection_start_time
                    logger.debug(f"Audio detection ended after {total_duration:.0f}ms")
                else:
                    # Still in detection period (brief silence)
                    detection_info.update({
                        "is_detecting": True,
                        "detection_duration_ms": current_time - self.detection_start_time
                    })
        
        return False, detection_info
    
    def calibrate_noise_floor(self, duration_seconds: int = 5) -> bool:
        """Calibrate noise floor by sampling quiet environment."""
        if not self.config.auto_calibrate:
            return False
        
        logger.info(f"Calibrating noise floor for {duration_seconds} seconds...")
        
        try:
            from .audio_capture import get_audio_config
            audio_config = get_audio_config()
            
            if not audio_config:
                logger.error("Failed to get audio configuration for calibration")
                return False
            
            # Setup audio stream for calibration
            p = pyaudio.PyAudio()
            
            stream_kwargs = {
                'format': pyaudio.paInt16,
                'channels': audio_config['channels'],
                'rate': audio_config['sample_rate'],
                'input': True,
                'frames_per_buffer': self.chunk_size
            }
            
            if audio_config['device_id'] is not None:
                stream_kwargs['input_device_index'] = audio_config['device_id']
            
            stream = p.open(**stream_kwargs)
            
            # Collect samples
            samples_per_second = audio_config['sample_rate'] // self.chunk_size
            total_samples = duration_seconds * samples_per_second
            db_levels = []
            
            for i in range(total_samples):
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    db_level = self.calculate_db_level(data)
                    db_levels.append(db_level)
                    
                    if i % samples_per_second == 0:  # Progress every second
                        logger.info(f"Calibration progress: {i // samples_per_second + 1}/{duration_seconds}s")
                        
                except Exception as e:
                    logger.warning(f"Calibration sample error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if not db_levels:
                logger.error("No valid samples collected during calibration")
                return False
            
            # Calculate noise floor (use median to avoid outliers)
            noise_floor = float(np.median(db_levels))
            self.adaptive_noise_floor = noise_floor
            self.config.noise_floor_db = noise_floor
            
            # Adjust threshold based on noise floor
            if self.config.adaptive_enabled:
                suggested_threshold = noise_floor + 12.0  # 12dB above noise
                if suggested_threshold != self.config.threshold_db:
                    self.config.threshold_db = suggested_threshold
                    logger.info(f"Adjusted threshold to {suggested_threshold:.1f}dB based on noise floor")
            
            self._save_config()
            
            logger.info(f"Calibration completed: noise floor = {noise_floor:.1f}dB")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get current threshold detection status."""
        return {
            "config": asdict(self.config),
            "adaptive_noise_floor": self.adaptive_noise_floor,
            "is_detecting": self.is_detecting,
            "presets": self.PRESETS,
            "recent_levels_count": len(self.recent_levels)
        }
    
    def test_threshold(self, duration_seconds: int = 3) -> Dict:
        """Test current threshold settings and return analysis."""
        logger.info(f"Testing threshold for {duration_seconds} seconds...")
        
        try:
            from .audio_capture import get_audio_config
            audio_config = get_audio_config()
            
            p = pyaudio.PyAudio()
            
            stream_kwargs = {
                'format': pyaudio.paInt16,
                'channels': audio_config['channels'],
                'rate': audio_config['sample_rate'],
                'input': True,
                'frames_per_buffer': self.chunk_size
            }
            
            if audio_config['device_id'] is not None:
                stream_kwargs['input_device_index'] = audio_config['device_id']
            
            stream = p.open(**stream_kwargs)
            
            # Test data collection
            samples_per_second = audio_config['sample_rate'] // self.chunk_size
            total_samples = duration_seconds * samples_per_second
            
            detections = 0
            db_levels = []
            detection_events = []
            
            for i in range(total_samples):
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    is_detected, info = self.is_audio_detected(data)
                    
                    db_levels.append(info['current_db'])
                    
                    if is_detected:
                        detections += 1
                        detection_events.append({
                            'time_ms': i * (1000 // samples_per_second),
                            'db_level': info['current_db'],
                            'duration_ms': info['detection_duration_ms']
                        })
                        
                except Exception as e:
                    logger.warning(f"Test sample error: {e}")
                    continue
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Analysis
            if db_levels:
                analysis = {
                    "duration_seconds": duration_seconds,
                    "total_detections": detections,
                    "detection_rate": detections / total_samples,
                    "min_db": float(np.min(db_levels)),
                    "max_db": float(np.max(db_levels)),
                    "mean_db": float(np.mean(db_levels)),
                    "median_db": float(np.median(db_levels)),
                    "current_threshold": self.config.threshold_db,
                    "adaptive_noise_floor": self.adaptive_noise_floor,
                    "detection_events": detection_events,
                    "recommendation": self._get_threshold_recommendation(db_levels, detections, total_samples)
                }
            else:
                analysis = {"error": "No audio samples collected"}
            
            logger.info(f"Threshold test completed: {detections} detections in {duration_seconds}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Threshold test failed: {e}")
            return {"error": str(e)}
    
    def _get_threshold_recommendation(self, db_levels: list, detections: int, total_samples: int) -> str:
        """Get threshold adjustment recommendation based on test results."""
        detection_rate = detections / total_samples if total_samples > 0 else 0
        mean_db = np.mean(db_levels)
        
        if detection_rate > 0.5:  # Too many detections
            return "Threshold trop sensible - augmentez le seuil de 3-5dB"
        elif detection_rate < 0.01:  # Too few detections
            return "Threshold pas assez sensible - diminuez le seuil de 3-5dB"
        elif mean_db > self.config.threshold_db + 10:
            return "Environnement bruyant - considérez le preset 'noisy'"
        elif mean_db < self.config.threshold_db - 15:
            return "Environnement silencieux - considérez le preset 'quiet'"
        else:
            return "Configuration threshold optimale"


# Global threshold detector instance
_threshold_detector: Optional[AudioThresholdDetector] = None


def get_threshold_detector() -> AudioThresholdDetector:
    """Get global threshold detector instance."""
    global _threshold_detector
    if _threshold_detector is None:
        _threshold_detector = AudioThresholdDetector()
    return _threshold_detector


def audio_triggered() -> bool:
    """Check if audio threshold is currently triggered."""
    try:
        detector = get_threshold_detector()
        # This would be called from the main detection loop
        # For now, return the current detection state
        return detector.is_detecting
    except Exception as e:
        logger.error(f"Audio threshold check failed: {e}")
        return False


if __name__ == "__main__":
    # Test threshold detection
    logging.basicConfig(level=logging.INFO)
    
    detector = AudioThresholdDetector()
    
    # Test calibration
    print("Testing noise floor calibration...")
    detector.calibrate_noise_floor(duration_seconds=3)
    
    # Test threshold
    print("Testing threshold detection...")
    result = detector.test_threshold(duration_seconds=5)
    print(f"Test result: {result}")