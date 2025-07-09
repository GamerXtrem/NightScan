"""
Détecteur de Type de Fichier pour NightScan

Ce module détecte automatiquement le type de fichier (audio, image, spectrogramme)
pour aiguiller vers le bon modèle de prédiction.
"""

import os
import re
import mimetypes
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from enum import Enum
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Types de fichiers supportés par NightScan."""
    AUDIO_RAW = "audio_raw"          # Fichiers WAV bruts
    AUDIO_SPECTROGRAM = "audio_spectrogram"  # Spectrogrammes NPY
    IMAGE = "image"                  # Images JPEG
    UNKNOWN = "unknown"              # Type non reconnu


class NightScanFile:
    """Classe représentant un fichier NightScan avec ses métadonnées."""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.file_type: FileType = FileType.UNKNOWN
        self.metadata: Dict = {}
        self.is_valid: bool = False
        self.error_message: Optional[str] = None
        
        # Analyser le fichier
        self._analyze_file()
    
    def _analyze_file(self):
        """Analyse le fichier pour déterminer son type et extraire les métadonnées."""
        try:
            # Vérifier l'existence du fichier
            if not self.file_path.exists():
                self.error_message = f"Fichier non trouvé: {self.file_path}"
                return
            
            # Analyser le nom de fichier
            name_analysis = self._analyze_filename()
            if not name_analysis["is_nightscan_format"]:
                self.error_message = f"Format de nom de fichier non conforme: {self.file_path.name}"
                return
            
            self.metadata.update(name_analysis)
            
            # Déterminer le type de fichier
            self.file_type = self._determine_file_type()
            
            # Validation spécifique au type
            if self.file_type != FileType.UNKNOWN:
                self.is_valid = self._validate_file_content()
                if not self.is_valid and not self.error_message:
                    self.error_message = f"Contenu du fichier invalide pour le type {self.file_type.value}"
            else:
                self.error_message = f"Type de fichier non reconnu: {self.file_path.suffix}"
                
        except Exception as e:
            self.error_message = f"Erreur lors de l'analyse: {str(e)}"
            logger.error(f"Erreur analyse fichier {self.file_path}: {e}")
    
    def _analyze_filename(self) -> Dict:
        """Analyse le nom de fichier selon la convention NightScan."""
        filename = self.file_path.name
        
        # Pattern NightScan: {TYPE}_{YYYYMMDD}_{HHMMSS}_{LAT}_{LON}.{ext}
        pattern = r'^(AUD|IMG)_(\d{8})_(\d{6})_(\d{4})_(\d{4})\.(\w+)$'
        match = re.match(pattern, filename)
        
        if not match:
            return {"is_nightscan_format": False}
        
        file_prefix, date_str, time_str, lat_str, lon_str, extension = match.groups()
        
        # Conversion des coordonnées GPS
        # Format: 4695 → 46.95 (latitude), 0745 → 07.45 (longitude)
        try:
            latitude = float(lat_str[:2] + '.' + lat_str[2:])
            longitude = float(lon_str[:2] + '.' + lon_str[2:])
        except ValueError:
            latitude = longitude = 0.0
        
        return {
            "is_nightscan_format": True,
            "file_prefix": file_prefix,
            "date": date_str,
            "time": time_str,
            "latitude": latitude,
            "longitude": longitude,
            "extension": extension.lower(),
            "timestamp": f"{date_str}_{time_str}"
        }
    
    def _determine_file_type(self) -> FileType:
        """Détermine le type de fichier basé sur l'extension et le préfixe."""
        if not self.metadata.get("is_nightscan_format"):
            return FileType.UNKNOWN
        
        extension = self.metadata["extension"]
        file_prefix = self.metadata["file_prefix"]
        
        # Spectrogrammes (fichiers NPY générés à partir d'audio)
        if extension == "npy" and file_prefix == "AUD":
            return FileType.AUDIO_SPECTROGRAM
        
        # Audio brut (fichiers WAV)
        elif extension == "wav" and file_prefix == "AUD":
            return FileType.AUDIO_RAW
        
        # Images (fichiers JPEG)
        elif extension in ["jpg", "jpeg"] and file_prefix == "IMG":
            return FileType.IMAGE
        
        return FileType.UNKNOWN
    
    def _validate_file_content(self) -> bool:
        """Valide le contenu du fichier selon son type."""
        try:
            if self.file_type == FileType.AUDIO_SPECTROGRAM:
                return self._validate_spectrogram()
            elif self.file_type == FileType.AUDIO_RAW:
                return self._validate_audio_wav()
            elif self.file_type == FileType.IMAGE:
                return self._validate_image()
            return False
        except Exception as e:
            self.error_message = f"Erreur validation contenu: {str(e)}"
            return False
    
    def _validate_spectrogram(self) -> bool:
        """Valide un fichier spectrogramme NPY."""
        try:
            # Charger le fichier NPY
            spectrogram = np.load(self.file_path)
            
            # Vérifier les dimensions (doit être 2D pour un spectrogramme)
            if len(spectrogram.shape) != 2:
                self.error_message = f"Spectrogramme doit être 2D, trouvé {len(spectrogram.shape)}D"
                return False
            
            # Vérifier les dimensions typiques d'un spectrogramme mel
            height, width = spectrogram.shape
            if height < 32 or height > 512 or width < 32 or width > 1024:
                self.error_message = f"Dimensions spectrogramme inhabituelles: {height}x{width}"
                return False
            
            # Vérifier le type de données
            if not np.issubdtype(spectrogram.dtype, np.floating):
                self.error_message = f"Type de données spectrogramme invalide: {spectrogram.dtype}"
                return False
            
            # Mettre à jour les métadonnées
            self.metadata.update({
                "spectrogram_shape": spectrogram.shape,
                "spectrogram_dtype": str(spectrogram.dtype),
                "spectrogram_range": (float(spectrogram.min()), float(spectrogram.max()))
            })
            
            return True
            
        except Exception as e:
            self.error_message = f"Erreur lecture spectrogramme: {str(e)}"
            return False
    
    def _validate_audio_wav(self) -> bool:
        """Valide un fichier audio WAV."""
        try:
            # Utiliser la bibliothèque wave pour valider
            import wave
            
            with wave.open(str(self.file_path), 'rb') as wav_file:
                # Vérifier les paramètres audio
                params = wav_file.getparams()
                
                # Métadonnées audio
                self.metadata.update({
                    "audio_channels": params.nchannels,
                    "audio_sample_width": params.sampwidth,
                    "audio_frame_rate": params.framerate,
                    "audio_frames": params.nframes,
                    "audio_duration": params.nframes / params.framerate
                })
                
                # Validations
                if params.nchannels not in [1, 2]:
                    self.error_message = f"Nombre de canaux audio invalide: {params.nchannels}"
                    return False
                
                if params.sampwidth not in [1, 2, 4]:
                    self.error_message = f"Largeur d'échantillon audio invalide: {params.sampwidth}"
                    return False
                
                if params.framerate < 8000 or params.framerate > 96000:
                    self.error_message = f"Fréquence d'échantillonnage inhabituelle: {params.framerate}Hz"
                    return False
                
                # Vérifier la durée (NightScan utilise généralement 8 secondes)
                duration = params.nframes / params.framerate
                if duration < 1.0 or duration > 30.0:
                    self.error_message = f"Durée audio inhabituelle: {duration:.2f}s"
                    return False
            
            return True
            
        except Exception as e:
            self.error_message = f"Erreur lecture audio: {str(e)}"
            return False
    
    def _validate_image(self) -> bool:
        """Valide un fichier image JPEG."""
        try:
            # Utiliser PIL pour valider l'image
            with Image.open(self.file_path) as img:
                # Vérifier le format
                if img.format != 'JPEG':
                    self.error_message = f"Format image invalide: {img.format}, attendu JPEG"
                    return False
                
                # Métadonnées image
                self.metadata.update({
                    "image_width": img.width,
                    "image_height": img.height,
                    "image_mode": img.mode,
                    "image_format": img.format
                })
                
                # Validations
                if img.width < 64 or img.height < 64:
                    self.error_message = f"Image trop petite: {img.width}x{img.height}"
                    return False
                
                if img.width > 4096 or img.height > 4096:
                    self.error_message = f"Image trop grande: {img.width}x{img.height}"
                    return False
                
                # Vérifier le mode couleur
                if img.mode not in ['RGB', 'L', 'P']:
                    self.error_message = f"Mode couleur non supporté: {img.mode}"
                    return False
            
            return True
            
        except Exception as e:
            self.error_message = f"Erreur lecture image: {str(e)}"
            return False
    
    def get_mime_type(self) -> str:
        """Retourne le type MIME du fichier."""
        if self.file_type == FileType.AUDIO_RAW:
            return "audio/wav"
        elif self.file_type == FileType.AUDIO_SPECTROGRAM:
            return "application/octet-stream"  # NPY
        elif self.file_type == FileType.IMAGE:
            return "image/jpeg"
        else:
            return "application/octet-stream"
    
    def get_processing_info(self) -> Dict:
        """Retourne les informations nécessaires pour le traitement."""
        return {
            "file_type": self.file_type.value,
            "file_path": str(self.file_path),
            "is_valid": self.is_valid,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "mime_type": self.get_mime_type()
        }


class FileTypeDetector:
    """Détecteur principal pour analyser les fichiers NightScan."""
    
    def __init__(self):
        self.supported_extensions = {
            '.wav': FileType.AUDIO_RAW,
            '.npy': FileType.AUDIO_SPECTROGRAM,
            '.jpg': FileType.IMAGE,
            '.jpeg': FileType.IMAGE
        }
    
    def detect_file_type(self, file_path: Union[str, Path]) -> NightScanFile:
        """
        Détecte le type d'un fichier et retourne un objet NightScanFile.
        
        Args:
            file_path: Chemin vers le fichier à analyser
            
        Returns:
            NightScanFile: Objet contenant les informations du fichier
        """
        return NightScanFile(file_path)
    
    def is_supported_extension(self, file_path: Union[str, Path]) -> bool:
        """Vérifie si l'extension du fichier est supportée."""
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions
    
    def get_expected_type_from_extension(self, file_path: Union[str, Path]) -> FileType:
        """Retourne le type de fichier attendu basé sur l'extension."""
        extension = Path(file_path).suffix.lower()
        return self.supported_extensions.get(extension, FileType.UNKNOWN)
    
    def batch_detect(self, file_paths: list) -> Dict[str, NightScanFile]:
        """
        Analyse plusieurs fichiers en lot.
        
        Args:
            file_paths: Liste des chemins de fichiers
            
        Returns:
            Dict: Dictionnaire {file_path: NightScanFile}
        """
        results = {}
        for file_path in file_paths:
            try:
                results[str(file_path)] = self.detect_file_type(file_path)
            except Exception as e:
                # Créer un objet avec erreur
                error_file = NightScanFile.__new__(NightScanFile)
                error_file.file_path = Path(file_path)
                error_file.file_type = FileType.UNKNOWN
                error_file.is_valid = False
                error_file.error_message = f"Erreur analyse: {str(e)}"
                error_file.metadata = {}
                results[str(file_path)] = error_file
        
        return results
    
    def get_file_stats(self, file_paths: list) -> Dict:
        """
        Retourne des statistiques sur un ensemble de fichiers.
        
        Args:
            file_paths: Liste des chemins de fichiers
            
        Returns:
            Dict: Statistiques par type de fichier
        """
        files = self.batch_detect(file_paths)
        stats = {
            "total_files": len(files),
            "valid_files": 0,
            "invalid_files": 0,
            "by_type": {
                FileType.AUDIO_RAW.value: 0,
                FileType.AUDIO_SPECTROGRAM.value: 0,
                FileType.IMAGE.value: 0,
                FileType.UNKNOWN.value: 0
            }
        }
        
        for file_obj in files.values():
            if file_obj.is_valid:
                stats["valid_files"] += 1
            else:
                stats["invalid_files"] += 1
            
            stats["by_type"][file_obj.file_type.value] += 1
        
        return stats


def main():
    """Fonction de test et démonstration."""
    detector = FileTypeDetector()
    
    # Exemples de fichiers NightScan
    test_files = [
        "AUD_20240109_143045_4695_0745.wav",
        "AUD_20240109_143045_4695_0745.npy",
        "IMG_20240109_143045_4695_0745.jpg",
        "invalid_file.txt"
    ]
    
    print("=== Test du Détecteur de Type de Fichier NightScan ===\n")
    
    for test_file in test_files:
        print(f"Analyse de: {test_file}")
        file_obj = detector.detect_file_type(test_file)
        
        info = file_obj.get_processing_info()
        print(f"  Type: {info['file_type']}")
        print(f"  Valide: {info['is_valid']}")
        print(f"  MIME: {info['mime_type']}")
        
        if info['error_message']:
            print(f"  Erreur: {info['error_message']}")
        
        if info['metadata']:
            print(f"  Métadonnées: {info['metadata']}")
        
        print()


if __name__ == "__main__":
    main()