"""
EfficientNet Prediction Script for Wildlife Audio Classification

High-performance prediction script with batch processing, confidence scoring,
and compatibility with existing NightScan API infrastructure.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms as T
from pydub import AudioSegment
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_config import EfficientNetConfig, create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TARGET_SAMPLE_RATE = 22050
TARGET_DURATION_MS = 8000


class EfficientNetPredictor:
    """
    High-performance EfficientNet predictor for wildlife audio classification.
    """
    
    def __init__(
        self,
        model_path: Path,
        config_path: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        self.model_path = model_path
        self.device = device or self._get_best_device()
        
        # Load configuration
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = EfficientNetConfig(**config_dict)
        else:
            # Use default configuration
            self.config = EfficientNetConfig()
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load class information
        self.class_names = [
            "bird_song", "mammal_call", "insect_sound",
            "amphibian_call", "environmental_sound", "unknown_species"
        ]
        
        # Load confidence thresholds
        self.confidence_thresholds = {
            "bird_song": 0.7,
            "mammal_call": 0.75,
            "insect_sound": 0.65,
            "amphibian_call": 0.7,
            "environmental_sound": 0.8,
            "unknown_species": 0.5
        }
        
        # Initialize transforms
        self._init_transforms()
        
        logger.info(f"EfficientNet predictor initialized on {self.device}")
        logger.info(f"Model: {self.config.model_name}")
    
    def _get_best_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _load_model(self) -> nn.Module:
        """Load the trained EfficientNet model."""
        try:
            # Create model
            model = create_model(self.config)
            
            # Load weights
            if self.model_path.suffix == '.pth':
                # Load state dict
                state_dict = torch.load(self.model_path, map_location=self.device)
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
            else:
                # Load complete model
                model = torch.load(self.model_path, map_location=self.device)
            
            model.to(self.device)
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _init_transforms(self) -> None:
        """Initialize audio transforms."""
        self.mel_transform = T.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=0.0,
            f_max=TARGET_SAMPLE_RATE // 2
        )
        
        self.db_transform = T.AmplitudeToDB(top_db=80)
        
        # Normalization for spectrograms
        self.normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    
    def preprocess_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Preprocess audio file to spectrogram tensor.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed spectrogram tensor
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != TARGET_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, TARGET_SAMPLE_RATE
                )
            
            # Ensure target duration
            target_length = int(TARGET_SAMPLE_RATE * TARGET_DURATION_MS / 1000)
            if waveform.size(1) > target_length:
                waveform = waveform[:, :target_length]
            elif waveform.size(1) < target_length:
                pad_length = target_length - waveform.size(1)
                waveform = F.pad(waveform, (0, pad_length))
            
            # Generate mel-spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec_db = self.db_transform(mel_spec)
            
            # Normalize
            mel_spec_db = self.normalize(mel_spec_db)
            
            # Convert to 3 channels for EfficientNet
            if mel_spec_db.size(0) == 1:
                mel_spec_db = mel_spec_db.repeat(3, 1, 1)
            
            # Resize to model input size
            mel_spec_db = F.interpolate(
                mel_spec_db.unsqueeze(0),
                size=self.config.input_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            return mel_spec_db
            
        except Exception as e:
            logger.error(f"Error preprocessing {audio_path}: {str(e)}")
            raise
    
    def predict_single(
        self,
        audio_path: Path,
        return_all_scores: bool = False,
        apply_threshold: bool = True
    ) -> Dict[str, Any]:
        """
        Predict class for a single audio file.
        
        Args:
            audio_path: Path to audio file
            return_all_scores: Whether to return all class scores
            apply_threshold: Whether to apply confidence thresholds
            
        Returns:
            Prediction results dictionary
        """
        try:
            # Preprocess audio
            spectrogram = self.preprocess_audio(audio_path)
            spectrogram = spectrogram.unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(spectrogram)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.class_names[predicted_idx.item()]
                confidence_score = confidence.item()
            
            # Apply confidence threshold if requested
            if apply_threshold:
                threshold = self.confidence_thresholds.get(predicted_class, 0.5)
                if confidence_score < threshold:
                    predicted_class = "unknown_species"
                    confidence_score = probabilities[0][self.class_names.index("unknown_species")].item()
            
            # Prepare result
            result = {
                "file_path": str(audio_path),
                "predicted_class": predicted_class,
                "confidence": confidence_score,
                "processing_time": time.time()
            }
            
            # Add all scores if requested
            if return_all_scores:
                all_scores = {}
                for i, class_name in enumerate(self.class_names):
                    all_scores[class_name] = probabilities[0][i].item()
                result["all_scores"] = all_scores
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting {audio_path}: {str(e)}")
            return {
                "file_path": str(audio_path),
                "error": str(e),
                "processing_time": time.time()
            }
    
    def predict_batch(
        self,
        audio_paths: List[Path],
        batch_size: int = 32,
        return_all_scores: bool = False,
        apply_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict classes for a batch of audio files.
        
        Args:
            audio_paths: List of audio file paths
            batch_size: Batch size for processing
            return_all_scores: Whether to return all class scores
            apply_threshold: Whether to apply confidence thresholds
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in tqdm(range(0, len(audio_paths), batch_size), desc="Processing batches"):
            batch_paths = audio_paths[i:i + batch_size]
            batch_spectrograms = []
            valid_indices = []
            
            # Preprocess batch
            for j, audio_path in enumerate(batch_paths):
                try:
                    spectrogram = self.preprocess_audio(audio_path)
                    batch_spectrograms.append(spectrogram)
                    valid_indices.append(j)
                except Exception as e:
                    logger.error(f"Error preprocessing {audio_path}: {str(e)}")
                    results.append({
                        "file_path": str(audio_path),
                        "error": str(e),
                        "processing_time": time.time()
                    })
            
            if not batch_spectrograms:
                continue
            
            # Create batch tensor
            batch_tensor = torch.stack(batch_spectrograms).to(self.device)
            
            # Predict
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probabilities, 1)
            
            processing_time = time.time() - start_time
            
            # Process results
            for j, valid_idx in enumerate(valid_indices):
                audio_path = batch_paths[valid_idx]
                predicted_idx = predicted_indices[j].item()
                predicted_class = self.class_names[predicted_idx]
                confidence_score = confidences[j].item()
                
                # Apply confidence threshold if requested
                if apply_threshold:
                    threshold = self.confidence_thresholds.get(predicted_class, 0.5)
                    if confidence_score < threshold:
                        predicted_class = "unknown_species"
                        confidence_score = probabilities[j][self.class_names.index("unknown_species")].item()
                
                # Prepare result
                result = {
                    "file_path": str(audio_path),
                    "predicted_class": predicted_class,
                    "confidence": confidence_score,
                    "processing_time": processing_time / len(valid_indices)
                }
                
                # Add all scores if requested
                if return_all_scores:
                    all_scores = {}
                    for k, class_name in enumerate(self.class_names):
                        all_scores[class_name] = probabilities[j][k].item()
                    result["all_scores"] = all_scores
                
                results.append(result)
        
        return results
    
    def predict_audio_segments(
        self,
        audio_path: Path,
        segment_duration_ms: int = 8000,
        overlap_ms: int = 1000,
        return_all_scores: bool = False,
        apply_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict classes for segments of a long audio file.
        
        Args:
            audio_path: Path to audio file
            segment_duration_ms: Duration of each segment in milliseconds
            overlap_ms: Overlap between segments in milliseconds
            return_all_scores: Whether to return all class scores
            apply_threshold: Whether to apply confidence thresholds
            
        Returns:
            List of prediction results for each segment
        """
        try:
            # Load audio with pydub for easy segmentation
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Resample if needed
            if audio.frame_rate != TARGET_SAMPLE_RATE:
                audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
            
            # Create segments
            segments = []
            step_size = segment_duration_ms - overlap_ms
            
            for start_ms in range(0, len(audio) - segment_duration_ms + 1, step_size):
                end_ms = start_ms + segment_duration_ms
                segment = audio[start_ms:end_ms]
                
                # Save segment to temporary file
                temp_path = Path(f"/tmp/temp_segment_{start_ms}.wav")
                segment.export(temp_path, format="wav")
                
                segments.append({
                    "path": temp_path,
                    "start_ms": start_ms,
                    "end_ms": end_ms
                })
            
            # Predict for each segment
            results = []
            for segment_info in tqdm(segments, desc="Processing segments"):
                try:
                    prediction = self.predict_single(
                        segment_info["path"],
                        return_all_scores=return_all_scores,
                        apply_threshold=apply_threshold
                    )
                    
                    # Add segment timing information
                    prediction["segment_start_ms"] = segment_info["start_ms"]
                    prediction["segment_end_ms"] = segment_info["end_ms"]
                    prediction["original_file"] = str(audio_path)
                    
                    results.append(prediction)
                    
                    # Clean up temporary file
                    segment_info["path"].unlink(missing_ok=True)
                    
                except Exception as e:
                    logger.error(f"Error processing segment {segment_info['start_ms']}-{segment_info['end_ms']}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio segments for {audio_path}: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "input_size": self.config.input_size,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "confidence_thresholds": self.confidence_thresholds
        }


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="EfficientNet prediction for wildlife audio")
    
    # Required arguments
    parser.add_argument("--model_path", type=Path, required=True,
                       help="Path to trained model file")
    parser.add_argument("--input", type=Path, required=True,
                       help="Input audio file or directory")
    
    # Optional arguments
    parser.add_argument("--config_path", type=Path, default=None,
                       help="Path to model configuration file")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--return_all_scores", action="store_true",
                       help="Return all class scores")
    parser.add_argument("--no_threshold", action="store_true",
                       help="Don't apply confidence thresholds")
    parser.add_argument("--segment_mode", action="store_true",
                       help="Process long audio files as segments")
    parser.add_argument("--segment_duration", type=int, default=8000,
                       help="Segment duration in milliseconds")
    parser.add_argument("--segment_overlap", type=int, default=1000,
                       help="Segment overlap in milliseconds")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, mps, cpu)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    
    # Setup device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Initialize predictor
    logger.info("Initializing EfficientNet predictor...")
    predictor = EfficientNetPredictor(
        model_path=args.model_path,
        config_path=args.config_path,
        device=device
    )
    
    # Show model info
    model_info = predictor.get_model_info()
    logger.info(f"Model: {model_info['model_name']}")
    logger.info(f"Device: {model_info['device']}")
    logger.info(f"Parameters: {model_info['total_parameters']:,}")
    
    # Determine input type and process
    if args.input.is_file():
        # Single file
        logger.info(f"Processing single file: {args.input}")
        
        if args.segment_mode:
            results = predictor.predict_audio_segments(
                args.input,
                segment_duration_ms=args.segment_duration,
                overlap_ms=args.segment_overlap,
                return_all_scores=args.return_all_scores,
                apply_threshold=not args.no_threshold
            )
        else:
            result = predictor.predict_single(
                args.input,
                return_all_scores=args.return_all_scores,
                apply_threshold=not args.no_threshold
            )
            results = [result]
    
    elif args.input.is_dir():
        # Directory of files
        logger.info(f"Processing directory: {args.input}")
        
        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(args.input.glob(f"**/*{ext}"))
        
        if not audio_files:
            logger.warning("No audio files found in directory")
            return
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process batch
        results = predictor.predict_batch(
            audio_files,
            batch_size=args.batch_size,
            return_all_scores=args.return_all_scores,
            apply_threshold=not args.no_threshold
        )
    
    else:
        raise ValueError(f"Input must be a file or directory: {args.input}")
    
    # Display results
    logger.info("\nPrediction Results:")
    logger.info("-" * 50)
    
    for result in results:
        if "error" in result:
            logger.error(f"Error: {result['file_path']} - {result['error']}")
        else:
            logger.info(f"File: {Path(result['file_path']).name}")
            logger.info(f"Predicted: {result['predicted_class']}")
            logger.info(f"Confidence: {result['confidence']:.3f}")
            
            if args.return_all_scores:
                logger.info("All scores:")
                for class_name, score in result['all_scores'].items():
                    logger.info(f"  {class_name}: {score:.3f}")
            
            logger.info("-" * 30)
    
    # Save results if output file specified
    if args.output:
        logger.info(f"Saving results to: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary statistics
    successful_predictions = [r for r in results if "error" not in r]
    error_count = len(results) - len(successful_predictions)
    
    logger.info(f"\nSummary:")
    logger.info(f"Total files processed: {len(results)}")
    logger.info(f"Successful predictions: {len(successful_predictions)}")
    logger.info(f"Errors: {error_count}")
    
    if successful_predictions:
        avg_confidence = np.mean([r['confidence'] for r in successful_predictions])
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        
        # Class distribution
        class_counts = {}
        for result in successful_predictions:
            class_name = result['predicted_class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        logger.info("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            logger.info(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()