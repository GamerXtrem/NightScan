"""Convert WAV recordings to spectrograms."""
from __future__ import annotations

from pathlib import Path
import os
import sys
import logging
from datetime import datetime, time as dtime

import numpy as np
import torchaudio
from torchaudio import transforms as T

# Ajouter le chemin vers les modules d'entraînement
sys.path.append(str(Path(__file__).parent.parent.parent / "audio_training_efficientnet"))

try:
    from spectrogram_config import SpectrogramConfig, get_config_for_animal
    from audio_augmentation import PreprocessingPipeline
    CONFIG_AVAILABLE = True
except ImportError:
    logging.warning("Configuration unifiée non disponible, utilisation des paramètres par défaut")
    CONFIG_AVAILABLE = False


TARGET_DURATION = 8  # Durée par défaut si config non disponible


def wav_to_spec(wav_path: Path, out_path: Path, sr: int = 22050, 
                config: SpectrogramConfig = None, animal_type: str = 'general') -> None:
    """Convert ``wav_path`` to a mel-spectrogram stored as ``out_path``.
    
    Args:
        wav_path: Chemin du fichier WAV
        out_path: Chemin de sortie pour le spectrogramme
        sr: Taux d'échantillonnage (ignoré si config fournie)
        config: Configuration de spectrogramme (optionnelle)
        animal_type: Type d'animal pour configuration automatique
    """
    # Utiliser la configuration unifiée si disponible
    if CONFIG_AVAILABLE and config is None:
        config = get_config_for_animal(animal_type)
        sr = config.sample_rate
        duration = config.duration
    else:
        duration = TARGET_DURATION
    
    # Charger l'audio
    waveform, original_sr = torchaudio.load(wav_path)
    
    # Convertir en mono si nécessaire
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Rééchantillonner si nécessaire
    if original_sr != sr:
        waveform = torchaudio.functional.resample(waveform, original_sr, sr)
    
    # Prétraitement si disponible
    if CONFIG_AVAILABLE and config:
        preprocessor = PreprocessingPipeline(sr, config.preprocessing_params)
        waveform_np = waveform.numpy().squeeze()
        waveform_np = preprocessor.process(waveform_np, config.fmin, config.fmax).numpy()
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    
    # Ajuster la durée
    target_length = int(sr * duration)
    if waveform.shape[1] < target_length:
        pad = target_length - waveform.shape[1]
        waveform = torchaudio.functional.pad(waveform, (0, pad))
    elif waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    
    # Créer le spectrogramme mel avec les paramètres optimisés
    if CONFIG_AVAILABLE and config:
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.fmin,
            f_max=config.fmax,
            window_fn=torch.hann_window if config.window == 'hann' else torch.hamming_window,
            center=config.center,
            pad_mode=config.pad_mode,
            power=config.power
        )
    else:
        # Paramètres par défaut si config non disponible
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=50,
            f_max=sr // 2
        )
    
    mel = mel_transform(waveform)
    
    # Conversion en dB
    if CONFIG_AVAILABLE and config:
        mel_db = T.AmplitudeToDB(top_db=config.top_db)(mel)
    else:
        mel_db = T.AmplitudeToDB(top_db=80)(mel)
    
    # Sauvegarder
    np.save(out_path, mel_db.squeeze(0).numpy())


def convert_directory(
    wav_dir: Path, out_dir: Path, remove: bool = False, *, sr: int = 22050,
    animal_type: str = 'general', config: SpectrogramConfig = None
) -> None:
    """Convert all WAV files in ``wav_dir`` to ``out_dir``.

    Any file that cannot be converted is skipped with a warning.
    
    Args:
        wav_dir: Répertoire contenant les fichiers WAV
        out_dir: Répertoire de sortie pour les spectrogrammes
        remove: Supprimer les WAV après conversion
        sr: Taux d'échantillonnage (ignoré si config fournie)
        animal_type: Type d'animal pour configuration automatique
        config: Configuration de spectrogramme personnalisée
    """
    wav_dir = Path(wav_dir)
    converted_count = 0
    error_count = 0
    
    # Utiliser la configuration appropriée
    if CONFIG_AVAILABLE and config is None:
        config = get_config_for_animal(animal_type)
        sr = config.sample_rate
        logging.info(f"Utilisation de la configuration '{animal_type}': sr={sr}, n_mels={config.n_mels}")
    
    for wav in wav_dir.rglob("*.wav"):
        rel = wav.relative_to(wav_dir)
        spec_path = out_dir / rel.with_suffix(".npy")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav_to_spec(wav, spec_path, sr=sr, config=config, animal_type=animal_type)
            converted_count += 1
        except Exception as exc:  # pragma: no cover - unexpected
            logging.warning("Could not convert %s: %s", wav, exc)
            error_count += 1
            continue
        if remove:
            wav.unlink()
    
    logging.info(f"Conversion terminée: {converted_count} réussis, {error_count} erreurs")


def disk_usage_percent(path: Path) -> float:
    """Return disk usage percentage for ``path``."""
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    return used / total * 100


def scheduled_conversion(
    wav_dir: Path,
    spec_dir: Path,
    *,
    threshold: float = 70.0,
    now: datetime | None = None,
    sr: int = 22050,
    animal_type: str = 'general',
    config: SpectrogramConfig = None
) -> None:
    """Convert WAV files after noon and delete them if disk usage is high.
    
    Args:
        wav_dir: Répertoire des fichiers WAV
        spec_dir: Répertoire de sortie pour les spectrogrammes
        threshold: Seuil d'utilisation disque pour suppression
        now: Date/heure actuelle (pour tests)
        sr: Taux d'échantillonnage (ignoré si config fournie)
        animal_type: Type d'animal pour configuration automatique
        config: Configuration de spectrogramme personnalisée
    """
    if now is None:
        now = datetime.now()
    if now.time() < dtime(12, 0):
        return
    convert_directory(wav_dir, spec_dir, sr=sr, animal_type=animal_type, config=config)
    if disk_usage_percent(wav_dir) >= threshold:
        for wav in Path(wav_dir).rglob("*.wav"):
            wav.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convertir des fichiers WAV en spectrogrammes mel optimisés"
    )
    parser.add_argument("wav_dir", type=Path, help="Répertoire des fichiers WAV")
    parser.add_argument("out_dir", type=Path, help="Répertoire de sortie")
    parser.add_argument("--remove", action="store_true",
                       help="Supprimer les WAV après conversion")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="Only run after noon and delete WAV when disk > threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Disk usage percentage triggering WAV deletion",
    )
    
    # Nouveaux arguments pour la configuration
    parser.add_argument(
        "--animal-type",
        type=str,
        default="general",
        choices=["general", "bat", "owl", "bird_song", "mammal", "amphibian", "insect"],
        help="Type d'animal pour optimisation automatique des paramètres"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Taux d'échantillonnage (ignoré si --animal-type est spécifié)"
    )
    
    args = parser.parse_args()
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.scheduled:
        scheduled_conversion(
            args.wav_dir, args.out_dir, 
            threshold=args.threshold,
            sr=args.sample_rate,
            animal_type=args.animal_type
        )
    else:
        convert_directory(
            args.wav_dir, args.out_dir, args.remove,
            sr=args.sample_rate,
            animal_type=args.animal_type
        )
