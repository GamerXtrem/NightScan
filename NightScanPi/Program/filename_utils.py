#!/usr/bin/env python3
"""
Utilitaires de Nommage des Fichiers NightScan
Format unifié avec métadonnées GPS intégrées
"""

import re
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import time

# Configuration logging
logger = logging.getLogger(__name__)

# Configuration des formats
FILENAME_FORMAT = "{type}_{date}_{time}_{lat}_{lon}.{ext}"
DATE_FORMAT = "%Y%m%d"
TIME_FORMAT = "%H%M%S"

# Types de fichiers supportés
FILE_TYPES = {
    'audio': 'AUD',
    'image': 'IMG',
    'video': 'VID'
}

# Extensions supportées
EXTENSIONS = {
    'audio': ['wav', 'mp3', 'flac', 'aac'],
    'image': ['jpg', 'jpeg', 'png', 'tiff'],
    'video': ['mp4', 'avi', 'mov', 'mkv']
}



class FilenameGenerator:
    """Générateur de noms de fichiers avec métadonnées GPS"""
    
    def __init__(self, location_manager=None):
        """
        Initialise le générateur de noms de fichiers.
        
        Args:
            location_manager: Instance du gestionnaire de localisation
        """
        self.location_manager = location_manager
        self._load_current_location()
    
    def _load_current_location(self):
        """Charge la localisation actuelle"""
        try:
            if self.location_manager:
                location = self.location_manager.get_current_location()
                self.current_lat = location.get('latitude', 46.9480)
                self.current_lon = location.get('longitude', 7.4474)
            else:
                # Valeurs par défaut (Zurich)
                self.current_lat = 46.9480
                self.current_lon = 7.4474
        except Exception as e:
            logger.warning(f"Erreur lors du chargement de la localisation: {e}")
            self.current_lat = 46.9480
            self.current_lon = 7.4474
    
    
    def _format_coordinate(self, coord: float, is_longitude: bool = False) -> str:
        """
        Formate une coordonnée GPS en format compact.
        
        Args:
            coord: Coordonnée GPS
            is_longitude: True si c'est une longitude
            
        Returns:
            str: Coordonnée formatée (4 caractères)
        """
        try:
            # Convertir en format compact : 46.9480 -> 4695
            if is_longitude:
                # Longitude : -180 à 180 -> ajouter 180 pour rendre positif
                normalized = coord + 180
                formatted = f"{normalized:06.2f}".replace('.', '')[:4]
            else:
                # Latitude : -90 à 90 -> ajouter 90 pour rendre positif
                normalized = coord + 90
                formatted = f"{normalized:05.2f}".replace('.', '')[:4]
            
            return formatted.zfill(4)
        except Exception as e:
            logger.error(f"Erreur lors du formatage de coordonnée {coord}: {e}")
            return "0000"
    
    def generate_filename(self, 
                         file_type: str, 
                         extension: str,
                         timestamp: Optional[datetime] = None,
                         latitude: Optional[float] = None,
                         longitude: Optional[float] = None) -> str:
        """
        Génère un nom de fichier avec le format unifié.
        
        Args:
            file_type: Type de fichier ('audio', 'image', 'video')
            extension: Extension du fichier sans le point
            timestamp: Timestamp (par défaut: maintenant)
            latitude: Latitude GPS (par défaut: position actuelle)
            longitude: Longitude GPS (par défaut: position actuelle)
            
        Returns:
            str: Nom de fichier formaté
        """
        try:
            # Valeurs par défaut
            if timestamp is None:
                timestamp = datetime.now()
            if latitude is None:
                latitude = self.current_lat
            if longitude is None:
                longitude = self.current_lon
            
            # Validation du type de fichier
            if file_type not in FILE_TYPES:
                raise ValueError(f"Type de fichier non supporté: {file_type}")
            
            # Validation de l'extension
            if extension.lower() not in EXTENSIONS.get(file_type, []):
                logger.warning(f"Extension {extension} inhabituelle pour {file_type}")
            
            # Formatage des composants
            type_code = FILE_TYPES[file_type]
            date_str = timestamp.strftime(DATE_FORMAT)
            time_str = timestamp.strftime(TIME_FORMAT)
            lat_str = self._format_coordinate(latitude, False)
            lon_str = self._format_coordinate(longitude, True)
            
            # Génération du nom
            filename = FILENAME_FORMAT.format(
                type=type_code,
                date=date_str,
                time=time_str,
                lat=lat_str,
                lon=lon_str,
                ext=extension.lower()
            )
            
            logger.info(f"Nom généré: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du nom: {e}")
            # Fallback vers format simple
            fallback_name = f"{file_type}_{int(time.time())}.{extension}"
            logger.warning(f"Utilisation du fallback: {fallback_name}")
            return fallback_name
    
    def generate_audio_filename(self, 
                               timestamp: Optional[datetime] = None,
                               latitude: Optional[float] = None,
                               longitude: Optional[float] = None) -> str:
        """
        Génère un nom de fichier audio.
        
        Args:
            timestamp: Timestamp (par défaut: maintenant)
            latitude: Latitude GPS (par défaut: position actuelle)
            longitude: Longitude GPS (par défaut: position actuelle)
            
        Returns:
            str: Nom de fichier audio
        """
        return self.generate_filename('audio', 'wav', timestamp, latitude, longitude)
    
    def generate_image_filename(self, 
                               timestamp: Optional[datetime] = None,
                               latitude: Optional[float] = None,
                               longitude: Optional[float] = None) -> str:
        """
        Génère un nom de fichier image.
        
        Args:
            timestamp: Timestamp (par défaut: maintenant)
            latitude: Latitude GPS (par défaut: position actuelle)
            longitude: Longitude GPS (par défaut: position actuelle)
            
        Returns:
            str: Nom de fichier image
        """
        return self.generate_filename('image', 'jpg', timestamp, latitude, longitude)


class FilenameParser:
    """Parseur de noms de fichiers NightScan (nouveau format uniquement)"""
    
    # Pattern de reconnaissance du nouveau format
    NEW_FORMAT_PATTERN = re.compile(
        r'^(?P<type>AUD|IMG|VID)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<lat>\d{4})_(?P<lon>\d{4})\.(?P<ext>\w+)$'
    )
    
    @classmethod
    def parse_filename(cls, filename: str) -> Dict[str, Union[str, datetime, float, None]]:
        """
        Parse un nom de fichier et extrait les métadonnées.
        
        Args:
            filename: Nom de fichier à parser
            
        Returns:
            Dict: Métadonnées extraites
        """
        filename = Path(filename).name  # Enlever le chemin si présent
        
        # Essayer le nouveau format
        match = cls.NEW_FORMAT_PATTERN.match(filename)
        if match:
            return cls._parse_new_format(match)
        
        # Format non reconnu
        logger.warning(f"Format de fichier non reconnu: {filename}")
        return {
            'filename': filename,
            'type': 'unknown',
            'format': 'unknown',
            'timestamp': None,
            'latitude': None,
            'longitude': None,
            'extension': Path(filename).suffix.lstrip('.')
        }
    
    @classmethod
    def _parse_new_format(cls, match) -> Dict[str, Union[str, datetime, float, None]]:
        """Parse le nouveau format unifié"""
        try:
            groups = match.groupdict()
            
            # Reconstituer le timestamp
            date_str = groups['date']
            time_str = groups['time']
            timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            
            # Reconstituer les coordonnées
            lat = cls._decode_coordinate(groups['lat'], is_longitude=False)
            lon = cls._decode_coordinate(groups['lon'], is_longitude=True)
            
            # Déterminer le type de fichier
            type_map = {v: k for k, v in FILE_TYPES.items()}
            file_type = type_map.get(groups['type'], 'unknown')
            
            return {
                'filename': match.string,
                'type': file_type,
                'format': 'new',
                'timestamp': timestamp,
                'latitude': lat,
                'longitude': lon,
                'extension': groups['ext']
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du parsing du nouveau format: {e}")
            return cls._create_error_result(match.string)
    
    
    @classmethod
    def _decode_coordinate(cls, coord_str: str, is_longitude: bool = False) -> float:
        """
        Décode une coordonnée du format compact.
        
        Args:
            coord_str: Coordonnée en format compact
            is_longitude: True si c'est une longitude
            
        Returns:
            float: Coordonnée GPS
        """
        try:
            # Reconstituer la coordonnée
            if len(coord_str) == 4:
                # Format: 4695 -> 46.95
                coord_float = float(coord_str[:2] + '.' + coord_str[2:])
            else:
                coord_float = float(coord_str)
            
            # Dénormaliser
            if is_longitude:
                return coord_float - 180
            else:
                return coord_float - 90
                
        except Exception as e:
            logger.error(f"Erreur lors du décodage de coordonnée {coord_str}: {e}")
            return 0.0
    
    @classmethod
    def _create_error_result(cls, filename: str) -> Dict[str, Union[str, datetime, float, None]]:
        """Crée un résultat d'erreur"""
        return {
            'filename': filename,
            'type': 'unknown',
            'format': 'error',
            'timestamp': None,
            'latitude': None,
            'longitude': None,
            'extension': Path(filename).suffix.lstrip('.')
        }
    
    @classmethod
    def is_new_format(cls, filename: str) -> bool:
        """Vérifie si un fichier utilise le nouveau format"""
        return cls.NEW_FORMAT_PATTERN.match(Path(filename).name) is not None
    
    @classmethod
    def get_file_type(cls, filename: str) -> str:
        """Récupère le type de fichier d'un nom"""
        parsed = cls.parse_filename(filename)
        return parsed.get('type', 'unknown')
    
    @classmethod
    def get_timestamp(cls, filename: str) -> Optional[datetime]:
        """Récupère le timestamp d'un nom de fichier"""
        parsed = cls.parse_filename(filename)
        return parsed.get('timestamp')
    
    @classmethod
    def get_coordinates(cls, filename: str) -> Tuple[Optional[float], Optional[float]]:
        """Récupère les coordonnées GPS d'un nom de fichier"""
        parsed = cls.parse_filename(filename)
        return parsed.get('latitude'), parsed.get('longitude')


# Fonctions utilitaires pour compatibilité
def create_audio_filename(timestamp: Optional[datetime] = None, 
                         latitude: Optional[float] = None,
                         longitude: Optional[float] = None) -> str:
    """
    Crée un nom de fichier audio avec le nouveau format.
    
    Args:
        timestamp: Timestamp (par défaut: maintenant)
        latitude: Latitude GPS (par défaut: position actuelle)
        longitude: Longitude GPS (par défaut: position actuelle)
        
    Returns:
        str: Nom de fichier audio
    """
    generator = FilenameGenerator()
    return generator.generate_audio_filename(timestamp, latitude, longitude)


def create_image_filename(timestamp: Optional[datetime] = None,
                         latitude: Optional[float] = None,
                         longitude: Optional[float] = None) -> str:
    """
    Crée un nom de fichier image avec le nouveau format.
    
    Args:
        timestamp: Timestamp (par défaut: maintenant)
        latitude: Latitude GPS (par défaut: position actuelle)
        longitude: Longitude GPS (par défaut: position actuelle)
        
    Returns:
        str: Nom de fichier image
    """
    generator = FilenameGenerator()
    return generator.generate_image_filename(timestamp, latitude, longitude)


def parse_any_filename(filename: str) -> Dict[str, Union[str, datetime, float, None]]:
    """
    Parse n'importe quel nom de fichier NightScan.
    
    Args:
        filename: Nom de fichier à parser
        
    Returns:
        Dict: Métadonnées extraites
    """
    return FilenameParser.parse_filename(filename)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration logging
    logging.basicConfig(level=logging.INFO)
    
    # Test du générateur
    generator = FilenameGenerator()
    
    # Génération de noms
    audio_name = generator.generate_audio_filename()
    image_name = generator.generate_image_filename()
    
    print(f"Audio: {audio_name}")
    print(f"Image: {image_name}")
    
    # Test du parseur
    parser = FilenameParser()
    
    # Test des formats
    test_files = [
        audio_name,
        image_name,
        "1625097600.wav",  # Ancien format audio
        "20240109_143045.jpg",  # Ancien format image
        "invalid_file.txt"  # Format invalide
    ]
    
    for filename in test_files:
        parsed = parser.parse_filename(filename)
        print(f"\nFichier: {filename}")
        print(f"Type: {parsed['type']}")
        print(f"Format: {parsed['format']}")
        print(f"Timestamp: {parsed['timestamp']}")
        print(f"GPS: {parsed['latitude']}, {parsed['longitude']}")
        print(f"Zone: {parsed['zone']}")