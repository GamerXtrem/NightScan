"""
Enhanced Audio Preprocessing for EfficientNet Training

Advanced preprocessing pipeline with integrated audio augmentations,
quality validation, and class balancing for wildlife audio classification.
"""

import argparse
import logging
import multiprocessing
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import csv

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as T
from pydub import AudioSegment, silence
from pydub.exceptions import CouldntDecodeError
from pydub.utils import which
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import librosa
import soundfile as sf

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.data_augmentation import AudioAugmentationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TARGET_DURATION_MS = 8000  # 8 seconds
TARGET_SAMPLE_RATE = 22050
TARGET_CHANNELS = 1


class AudioQualityValidator:
    """
    Validates audio quality and filters out problematic samples.
    """
    
    def __init__(
        self,
        min_duration_ms: int = 1000,
        max_duration_ms: int = 30000,
        min_sample_rate: int = 16000,
        max_silence_ratio: float = 0.9,
        min_dynamic_range_db: float = 10.0
    ):
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.min_sample_rate = min_sample_rate
        self.max_silence_ratio = max_silence_ratio
        self.min_dynamic_range_db = min_dynamic_range_db
    
    def validate_audio(self, audio_path: Path) -> Dict[str, Any]:
        """
        Validate audio file quality.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Load audio with pydub for basic validation
            audio = AudioSegment.from_file(audio_path)
            
            # Basic checks
            duration_ms = len(audio)
            sample_rate = audio.frame_rate
            channels = audio.channels
            
            if duration_ms < self.min_duration_ms:
                return {"valid": False, "reason": "too_short"}
            
            if duration_ms > self.max_duration_ms:
                return {"valid": False, "reason": "too_long"}
            
            if sample_rate < self.min_sample_rate:
                return {"valid": False, "reason": "low_sample_rate"}
            
            # Check for excessive silence
            silence_ratio = self._calculate_silence_ratio(audio)
            if silence_ratio > self.max_silence_ratio:
                return {"valid": False, "reason": "too_much_silence"}
            
            # Check dynamic range
            dynamic_range = self._calculate_dynamic_range(audio)
            if dynamic_range < self.min_dynamic_range_db:
                return {"valid": False, "reason": "low_dynamic_range"}
            
            return {
                "valid": True,
                "duration_ms": duration_ms,
                "sample_rate": sample_rate,
                "channels": channels,
                "silence_ratio": silence_ratio,
                "dynamic_range": dynamic_range
            }
            
        except Exception as e:
            return {"valid": False, "reason": f"error: {str(e)}"}
    
    def _calculate_silence_ratio(self, audio: AudioSegment) -> float:
        """Calculate ratio of silence in audio."""
        try:
            # Detect silence with a reasonable threshold
            silence_threshold = audio.dBFS - 20
            silent_chunks = silence.detect_silence(
                audio,
                min_silence_len=100,
                silence_thresh=silence_threshold
            )
            
            total_silence_ms = sum(end - start for start, end in silent_chunks)
            return total_silence_ms / len(audio)
        except:
            return 0.0
    
    def _calculate_dynamic_range(self, audio: AudioSegment) -> float:
        """Calculate dynamic range of audio."""
        try:
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)
            
            # Calculate percentiles
            p95 = np.percentile(np.abs(samples), 95)
            p5 = np.percentile(np.abs(samples), 5)
            
            # Dynamic range in dB
            if p5 > 0:
                return 20 * np.log10(p95 / p5)
            else:
                return 0.0
        except:
            return 0.0


class EnhancedAudioProcessor:
    """
    Enhanced audio processor with augmentation and quality validation.
    """
    
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        target_duration_ms: int = TARGET_DURATION_MS,
        augmentation_config: Optional[Dict[str, Any]] = None,
        quality_validator: Optional[AudioQualityValidator] = None
    ):
        self.sample_rate = sample_rate
        self.target_duration_ms = target_duration_ms
        self.quality_validator = quality_validator or AudioQualityValidator()
        
        # Initialize augmentation pipeline
        if augmentation_config:
            self.augmentation_pipeline = AudioAugmentationPipeline(
                sample_rate=sample_rate,
                **augmentation_config
            )
        else:
            self.augmentation_pipeline = None
    
    def process_audio_file(
        self,
        input_path: Path,
        output_dir: Path,
        class_label: str,
        apply_augmentation: bool = True,
        augmentation_factor: int = 2
    ) -> List[Path]:
        """
        Process a single audio file with optional augmentation.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save processed files
            class_label: Class label for the audio
            apply_augmentation: Whether to apply augmentation
            augmentation_factor: Number of augmented versions to create
            
        Returns:
            List of paths to processed files
        """
        try:
            # Validate audio quality
            validation_result = self.quality_validator.validate_audio(input_path)
            if not validation_result["valid"]:
                logger.warning(f"Skipping {input_path}: {validation_result['reason']}")
                return []
            
            # Load and preprocess audio
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Resample if needed
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)
            
            # Normalize
            if audio.max_dBFS != float("-inf"):
                audio = audio.apply_gain(-audio.max_dBFS)
            
            # Split on silence and create segments
            segments = self._create_segments(audio)
            
            if not segments:
                logger.warning(f"No valid segments found in {input_path}")
                return []
            
            # Process segments
            processed_files = []
            
            for idx, segment in enumerate(segments):
                # Save original segment
                base_name = f"{input_path.stem}_{idx}"
                original_path = output_dir / f"{base_name}.wav"
                segment.export(original_path, format="wav")
                processed_files.append(original_path)
                
                # Apply augmentation if enabled
                if apply_augmentation and self.augmentation_pipeline:
                    augmented_files = self._apply_audio_augmentation(
                        segment, output_dir, base_name, augmentation_factor
                    )
                    processed_files.extend(augmented_files)
            
            return processed_files
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            return []
    
    def _create_segments(self, audio: AudioSegment) -> List[AudioSegment]:
        """Create fixed-length segments from audio."""
        # Split on silence
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=300,
            silence_thresh=audio.dBFS - 20,
            keep_silence=150
        )
        
        segments = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            
            # Pad or truncate to target duration
            if len(chunk) > self.target_duration_ms:
                chunk = chunk[:self.target_duration_ms]
            elif len(chunk) < self.target_duration_ms:
                pad_len = self.target_duration_ms - len(chunk)
                chunk = (
                    AudioSegment.silent(pad_len // 2) +
                    chunk +
                    AudioSegment.silent(pad_len - pad_len // 2)
                )
            
            # Skip if too quiet
            if chunk.dBFS < -50:
                continue
            
            segments.append(chunk)
        
        return segments
    
    def _apply_audio_augmentation(
        self,
        segment: AudioSegment,
        output_dir: Path,
        base_name: str,
        augmentation_factor: int
    ) -> List[Path]:
        """Apply audio augmentation to a segment."""
        if not self.augmentation_pipeline:
            return []
        
        augmented_files = []
        
        # Convert to numpy array
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.iinfo(samples.dtype).max
        
        # Apply different augmentations
        for i in range(augmentation_factor):
            try:
                # Apply augmentation
                augmented_samples = self.augmentation_pipeline.augment_audio(samples)
                
                # Convert back to AudioSegment
                augmented_samples = np.clip(augmented_samples, -1.0, 1.0)
                augmented_samples = (augmented_samples * 32767).astype(np.int16)
                
                augmented_audio = AudioSegment(
                    augmented_samples.tobytes(),
                    frame_rate=self.sample_rate,
                    sample_width=2,
                    channels=1
                )
                
                # Save augmented version
                aug_path = output_dir / f"{base_name}_aug_{i}.wav"
                augmented_audio.export(aug_path, format="wav")
                augmented_files.append(aug_path)
                
            except Exception as e:
                logger.warning(f"Augmentation failed for {base_name}: {str(e)}")
        
        return augmented_files
    
    def generate_spectrogram(self, audio_path: Path, output_path: Path) -> bool:
        """
        Generate mel-spectrogram from audio file.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to save spectrogram
            
        Returns:
            True if successful
        """
        try:
            # Load audio with torchaudio
            waveform, orig_sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if orig_sr != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, orig_sr, self.sample_rate
                )
            
            # Generate mel-spectrogram
            mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=128,
                f_min=0.0,
                f_max=self.sample_rate // 2
            )
            
            db_transform = T.AmplitudeToDB(top_db=80)
            
            mel_spec = mel_transform(waveform)
            mel_spec_db = db_transform(mel_spec)
            
            # Save spectrogram
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, mel_spec_db.squeeze(0).numpy())
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating spectrogram for {audio_path}: {str(e)}")
            return False


def process_class_directory(
    class_dir: Path,
    output_dir: Path,
    processor: EnhancedAudioProcessor,
    apply_augmentation: bool = True,
    augmentation_factor: int = 2
) -> List[Tuple[Path, str]]:
    """
    Process all audio files in a class directory.
    
    Args:
        class_dir: Directory containing audio files for one class
        output_dir: Output directory
        processor: Audio processor
        apply_augmentation: Whether to apply augmentation
        augmentation_factor: Number of augmented versions per file
        
    Returns:
        List of (processed_file_path, class_label) tuples
    """
    class_label = class_dir.name
    class_output_dir = output_dir / "segments" / class_label
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for ext in [".wav", ".mp3", ".flac", ".m4a", ".aac"]:
        audio_files.extend(class_dir.glob(f"**/*{ext}"))
    
    logger.info(f"Processing {len(audio_files)} files for class: {class_label}")
    
    processed_files = []
    
    for audio_file in tqdm(audio_files, desc=f"Processing {class_label}"):
        try:
            file_processed = processor.process_audio_file(
                audio_file,
                class_output_dir,
                class_label,
                apply_augmentation=apply_augmentation,
                augmentation_factor=augmentation_factor
            )
            
            for processed_file in file_processed:
                processed_files.append((processed_file, class_label))
                
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {str(e)}")
    
    logger.info(f"Processed {len(processed_files)} segments for class: {class_label}")
    return processed_files


def generate_spectrograms_batch(
    audio_files: List[Tuple[Path, str]],
    output_dir: Path,
    processor: EnhancedAudioProcessor,
    num_workers: int = 4
) -> List[Tuple[Path, str]]:
    """
    Generate spectrograms for a batch of audio files.
    
    Args:
        audio_files: List of (audio_path, class_label) tuples
        output_dir: Output directory
        processor: Audio processor
        num_workers: Number of worker processes
        
    Returns:
        List of (spectrogram_path, class_label) tuples
    """
    spec_dir = output_dir / "spectrograms"
    spec_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_file(args):
        audio_path, class_label = args
        spec_path = spec_dir / class_label / f"{audio_path.stem}.npy"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        
        if processor.generate_spectrogram(audio_path, spec_path):
            return (spec_path, class_label)
        return None
    
    # Process spectrograms in parallel
    spectrograms = []
    
    if num_workers == 1:
        for audio_file in tqdm(audio_files, desc="Generating spectrograms"):
            result = process_single_file(audio_file)
            if result:
                spectrograms.append(result)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_file, audio_files),
                total=len(audio_files),
                desc="Generating spectrograms"
            ))
            
            spectrograms = [r for r in results if r is not None]
    
    return spectrograms


def balance_classes(
    spectrograms: List[Tuple[Path, str]],
    target_samples_per_class: Optional[int] = None,
    method: str = "oversample"
) -> List[Tuple[Path, str]]:
    """
    Balance classes by oversampling or undersampling.
    
    Args:
        spectrograms: List of (spectrogram_path, class_label) tuples
        target_samples_per_class: Target number of samples per class
        method: Balancing method ("oversample" or "undersample")
        
    Returns:
        Balanced list of spectrograms
    """
    # Group by class
    class_samples = {}
    for spec_path, class_label in spectrograms:
        if class_label not in class_samples:
            class_samples[class_label] = []
        class_samples[class_label].append(spec_path)
    
    # Determine target size
    if target_samples_per_class is None:
        if method == "oversample":
            target_samples_per_class = max(len(samples) for samples in class_samples.values())
        else:  # undersample
            target_samples_per_class = min(len(samples) for samples in class_samples.values())
    
    logger.info(f"Balancing classes to {target_samples_per_class} samples per class")
    
    # Balance classes
    balanced_spectrograms = []
    
    for class_label, samples in class_samples.items():
        current_count = len(samples)
        
        if current_count >= target_samples_per_class:
            # Undersample
            selected_samples = random.sample(samples, target_samples_per_class)
        else:
            # Oversample
            selected_samples = samples.copy()
            while len(selected_samples) < target_samples_per_class:
                additional_needed = target_samples_per_class - len(selected_samples)
                to_add = min(additional_needed, current_count)
                selected_samples.extend(random.sample(samples, to_add))
        
        # Add to balanced list
        for sample in selected_samples:
            balanced_spectrograms.append((sample, class_label))
        
        logger.info(f"Class {class_label}: {current_count} -> {len(selected_samples)} samples")
    
    return balanced_spectrograms


def create_splits(
    spectrograms: List[Tuple[Path, str]],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42
) -> None:
    """
    Create train/validation/test splits and save to CSV files.
    
    Args:
        spectrograms: List of (spectrogram_path, class_label) tuples
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify: Whether to stratify splits
        random_state: Random state for reproducibility
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Extract data
    paths = [str(path) for path, _ in spectrograms]
    labels = [label for _, label in spectrograms]
    
    # Create label mapping
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    label_indices = [label_to_idx[label] for label in labels]
    
    # Split data
    if stratify:
        # First split: train vs (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, label_indices,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=label_indices
        )
        
        # Second split: val vs test
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=random_state,
            stratify=temp_labels
        )
    else:
        # Simple random split
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, label_indices,
            test_size=(val_ratio + test_ratio),
            random_state=random_state
        )
        
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=random_state
        )
    
    # Save splits to CSV
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels)
    }
    
    for split_name, (split_paths, split_labels) in splits.items():
        csv_path = csv_dir / f"{split_name}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            
            for path, label in zip(split_paths, split_labels):
                writer.writerow([path, label])
        
        logger.info(f"Saved {len(split_paths)} samples to {csv_path}")
    
    # Save label mapping
    label_mapping = {
        "classes": unique_labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": {idx: label for label, idx in label_to_idx.items()}
    }
    
    with open(output_dir / "label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    logger.info(f"Label mapping saved to {output_dir / 'label_mapping.json'}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Enhanced audio preprocessing for EfficientNet")
    
    # Required arguments
    parser.add_argument("--input_dir", type=Path, required=True,
                       help="Directory with audio files organized by class")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Directory to store processed data")
    
    # Processing options
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of worker processes")
    parser.add_argument("--sample_rate", type=int, default=TARGET_SAMPLE_RATE,
                       help="Target sample rate")
    parser.add_argument("--target_duration_ms", type=int, default=TARGET_DURATION_MS,
                       help="Target duration in milliseconds")
    
    # Augmentation options
    parser.add_argument("--apply_augmentation", action="store_true",
                       help="Apply audio augmentation")
    parser.add_argument("--augmentation_factor", type=int, default=2,
                       help="Number of augmented versions per file")
    
    # Class balancing
    parser.add_argument("--balance_classes", action="store_true",
                       help="Balance classes by oversampling")
    parser.add_argument("--target_samples_per_class", type=int, default=None,
                       help="Target number of samples per class")
    
    # Split options
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="Test set ratio")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    quality_validator = AudioQualityValidator()
    
    augmentation_config = None
    if args.apply_augmentation:
        augmentation_config = {
            "use_time_stretch": True,
            "use_pitch_shift": True,
            "use_noise": True,
            "use_volume": True,
            "use_spec_augment": False,  # Applied later in training
            "use_mixup": False  # Applied later in training
        }
    
    processor = EnhancedAudioProcessor(
        sample_rate=args.sample_rate,
        target_duration_ms=args.target_duration_ms,
        augmentation_config=augmentation_config,
        quality_validator=quality_validator
    )
    
    # Process audio files by class
    logger.info("Starting enhanced audio preprocessing...")
    
    all_processed_files = []
    class_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        processed_files = process_class_directory(
            class_dir,
            args.output_dir,
            processor,
            apply_augmentation=args.apply_augmentation,
            augmentation_factor=args.augmentation_factor
        )
        all_processed_files.extend(processed_files)
    
    logger.info(f"Processed {len(all_processed_files)} audio segments")
    
    # Generate spectrograms
    logger.info("Generating spectrograms...")
    spectrograms = generate_spectrograms_batch(
        all_processed_files,
        args.output_dir,
        processor,
        args.workers
    )
    
    logger.info(f"Generated {len(spectrograms)} spectrograms")
    
    # Balance classes if requested
    if args.balance_classes:
        logger.info("Balancing classes...")
        spectrograms = balance_classes(
            spectrograms,
            target_samples_per_class=args.target_samples_per_class,
            method="oversample"
        )
        logger.info(f"Balanced dataset: {len(spectrograms)} spectrograms")
    
    # Create splits
    logger.info("Creating train/validation/test splits...")
    create_splits(
        spectrograms,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=True,
        random_state=args.random_state
    )
    
    # Save processing summary
    summary = {
        "total_processed_files": len(all_processed_files),
        "total_spectrograms": len(spectrograms),
        "augmentation_applied": args.apply_augmentation,
        "augmentation_factor": args.augmentation_factor,
        "classes_balanced": args.balance_classes,
        "sample_rate": args.sample_rate,
        "target_duration_ms": args.target_duration_ms,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio
        }
    }
    
    with open(args.output_dir / "processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Enhanced preprocessing completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()