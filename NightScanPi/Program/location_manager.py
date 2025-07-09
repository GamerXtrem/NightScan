#!/usr/bin/env python3
"""
Location Manager pour NightScan Pi
Gère la localisation géographique du Pi avec API et sauvegarde persistante
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path
import requests
from dataclasses import dataclass

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LocationData:
    """Données de localisation du Pi"""
    latitude: float
    longitude: float
    address: str = ""
    zone: str = ""
    timezone: str = ""
    updated_at: str = ""
    source: str = "manual"  # manual, gps, phone

class LocationManager:
    """Gestionnaire de localisation pour NightScan Pi"""
    
    def __init__(self, db_path: str = "nightscan_config.db"):
        self.db_path = Path(db_path)
        self.current_location: Optional[LocationData] = None
        self._init_database()
        self._load_current_location()
    
    def _init_database(self):
        """Initialise la base de données de configuration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Créer la table de configuration de localisation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pi_location (
                    id INTEGER PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    address TEXT DEFAULT '',
                    zone TEXT DEFAULT '',
                    timezone TEXT DEFAULT '',
                    source TEXT DEFAULT 'manual',
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            # Créer la table d'historique
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS location_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    address TEXT DEFAULT '',
                    zone TEXT DEFAULT '',
                    source TEXT DEFAULT 'manual',
                    changed_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Base de données de localisation initialisée")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la DB: {e}")
    
    def _load_current_location(self):
        """Charge la localisation actuelle depuis la base de données"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT latitude, longitude, address, zone, timezone, source, updated_at
                FROM pi_location 
                WHERE is_active = 1 
                ORDER BY updated_at DESC 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                self.current_location = LocationData(
                    latitude=result[0],
                    longitude=result[1],
                    address=result[2] or "",
                    zone=result[3] or "",
                    timezone=result[4] or "",
                    source=result[5] or "manual",
                    updated_at=result[6] or ""
                )
                logger.info(f"Localisation chargée: {self.current_location.latitude}, {self.current_location.longitude}")
            else:
                # Localisation par défaut (Zurich)
                self._set_default_location()
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la localisation: {e}")
            self._set_default_location()
    
    def _set_default_location(self):
        """Configure la localisation par défaut"""
        self.current_location = LocationData(
            latitude=46.9480,
            longitude=7.4474,
            address="Zurich, Switzerland",
            zone="Europe/Zurich",
            timezone="Europe/Zurich",
            source="default"
        )
        logger.info("Localisation par défaut configurée (Zurich)")
    
    def get_current_location(self) -> Dict:
        """Récupère la localisation actuelle"""
        if not self.current_location:
            self._load_current_location()
        
        return {
            "latitude": self.current_location.latitude,
            "longitude": self.current_location.longitude,
            "address": self.current_location.address,
            "zone": self.current_location.zone,
            "timezone": self.current_location.timezone,
            "source": self.current_location.source,
            "updated_at": self.current_location.updated_at
        }
    
    def update_location(self, latitude: float, longitude: float, 
                       address: str = "", zone: str = "", 
                       source: str = "manual") -> bool:
        """
        Met à jour la localisation du Pi
        
        Args:
            latitude: Latitude en degrés décimaux
            longitude: Longitude en degrés décimaux
            address: Adresse textuelle (optionnel)
            zone: Zone/région (optionnel)
            source: Source de la localisation (manual, gps, phone)
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        try:
            # Validation des coordonnées
            if not self._validate_coordinates(latitude, longitude):
                logger.error("Coordonnées invalides")
                return False
            
            # Résolution inverse de géolocalisation pour obtenir l'adresse
            if not address:
                address = self._reverse_geocode(latitude, longitude)
            
            # Détection du fuseau horaire
            if not zone:
                zone = self._get_timezone(latitude, longitude)
            
            # Sauvegarder l'ancienne localisation dans l'historique
            if self.current_location:
                self._save_to_history(self.current_location)
            
            # Mettre à jour la base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Désactiver l'ancienne localisation
            cursor.execute("UPDATE pi_location SET is_active = 0")
            
            # Insérer la nouvelle localisation
            cursor.execute("""
                INSERT INTO pi_location 
                (latitude, longitude, address, zone, timezone, source, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                latitude, longitude, address, zone, zone, source, 
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Mettre à jour l'objet actuel
            self.current_location = LocationData(
                latitude=latitude,
                longitude=longitude,
                address=address,
                zone=zone,
                timezone=zone,
                source=source,
                updated_at=datetime.now().isoformat()
            )
            
            logger.info(f"Localisation mise à jour: {latitude}, {longitude} ({source})")
            
            # Déclencher la mise à jour des calculs astronomiques
            self._update_sun_times()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la localisation: {e}")
            return False
    
    def update_from_phone(self, phone_location: Dict) -> bool:
        """
        Met à jour la localisation depuis les données du téléphone
        
        Args:
            phone_location: Dict contenant latitude, longitude, accuracy, etc.
            
        Returns:
            bool: True si la mise à jour a réussi
        """
        try:
            latitude = phone_location.get("latitude")
            longitude = phone_location.get("longitude")
            accuracy = phone_location.get("accuracy", 0)
            
            if not latitude or not longitude:
                logger.error("Coordonnées manquantes dans les données du téléphone")
                return False
            
            # Vérifier la précision (rejeter si > 100m)
            if accuracy > 100:
                logger.warning(f"Précision GPS faible: {accuracy}m")
                return False
            
            logger.info(f"Mise à jour depuis téléphone: {latitude}, {longitude} (précision: {accuracy}m)")
            
            return self.update_location(
                latitude=latitude,
                longitude=longitude,
                source="phone"
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour depuis téléphone: {e}")
            return False
    
    def _validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Valide les coordonnées GPS"""
        try:
            if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
                return False
            
            if not (-90 <= latitude <= 90):
                return False
            
            if not (-180 <= longitude <= 180):
                return False
            
            return True
            
        except:
            return False
    
    def _reverse_geocode(self, latitude: float, longitude: float) -> str:
        """Résolution inverse de géolocalisation"""
        try:
            # Utiliser le service de géocodage gratuit OpenStreetMap
            url = f"https://nominatim.openstreetmap.org/reverse"
            params = {
                "lat": latitude,
                "lon": longitude,
                "format": "json",
                "addressdetails": 1
            }
            
            headers = {
                "User-Agent": "NightScan-Pi/1.0"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Construire une adresse lisible
                address_parts = []
                if data.get("address"):
                    addr = data["address"]
                    if addr.get("house_number") and addr.get("road"):
                        address_parts.append(f"{addr['house_number']} {addr['road']}")
                    elif addr.get("road"):
                        address_parts.append(addr["road"])
                    
                    if addr.get("city"):
                        address_parts.append(addr["city"])
                    elif addr.get("town"):
                        address_parts.append(addr["town"])
                    elif addr.get("village"):
                        address_parts.append(addr["village"])
                    
                    if addr.get("country"):
                        address_parts.append(addr["country"])
                
                return ", ".join(address_parts) if address_parts else "Adresse inconnue"
            
        except Exception as e:
            logger.warning(f"Erreur de géocodage inverse: {e}")
        
        return "Adresse inconnue"
    
    def _get_timezone(self, latitude: float, longitude: float) -> str:
        """Détermine le fuseau horaire basé sur les coordonnées"""
        try:
            # Utiliser un service de fuseau horaire gratuit
            url = f"http://api.timezonedb.com/v2.1/get-time-zone"
            params = {
                "key": "demo",  # Clé démo limitée
                "format": "json",
                "by": "position",
                "lat": latitude,
                "lng": longitude
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("zoneName"):
                    return data["zoneName"]
            
        except Exception as e:
            logger.warning(f"Erreur de détection du fuseau horaire: {e}")
        
        # Fuseau horaire par défaut
        return "Europe/Zurich"
    
    def _save_to_history(self, location: LocationData):
        """Sauvegarde la localisation dans l'historique"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO location_history 
                (latitude, longitude, address, zone, source, changed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                location.latitude,
                location.longitude,
                location.address,
                location.zone,
                location.source,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {e}")
    
    def _update_sun_times(self):
        """Met à jour les calculs des heures de lever/coucher du soleil"""
        try:
            # Importer et mettre à jour sun_times.py
            import sys
            sys.path.append(str(Path(__file__).parent))
            
            from sun_times import SunTimes
            
            sun_times = SunTimes(
                latitude=self.current_location.latitude,
                longitude=self.current_location.longitude
            )
            
            # Calculer pour aujourd'hui
            today = datetime.now().date()
            sunrise, sunset = sun_times.get_sun_times(today)
            
            logger.info(f"Heures mises à jour - Lever: {sunrise}, Coucher: {sunset}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des heures: {e}")
    
    def get_location_history(self, limit: int = 10) -> list:
        """Récupère l'historique des localisations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT latitude, longitude, address, zone, source, changed_at
                FROM location_history 
                ORDER BY changed_at DESC 
                LIMIT ?
            """, (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                history.append({
                    "latitude": row[0],
                    "longitude": row[1],
                    "address": row[2],
                    "zone": row[3],
                    "source": row[4],
                    "changed_at": row[5]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique: {e}")
            return []
    
    def get_coordinates(self) -> Tuple[float, float]:
        """Récupère les coordonnées actuelles (format tuple)"""
        if not self.current_location:
            self._load_current_location()
        
        return (self.current_location.latitude, self.current_location.longitude)
    
    def export_location_data(self) -> Dict:
        """Exporte toutes les données de localisation"""
        return {
            "current_location": self.get_current_location(),
            "location_history": self.get_location_history(50)
        }


# Instance globale pour l'application
location_manager = LocationManager()

# Fonctions utilitaires pour l'intégration
def get_current_coordinates() -> Tuple[float, float]:
    """Récupère les coordonnées actuelles (compatible avec l'API existante)"""
    return location_manager.get_coordinates()

def update_pi_location(latitude: float, longitude: float, source: str = "manual") -> bool:
    """Met à jour la localisation du Pi"""
    return location_manager.update_location(latitude, longitude, source=source)

def update_from_phone_location(phone_data: Dict) -> bool:
    """Met à jour depuis les données du téléphone"""
    return location_manager.update_from_phone(phone_data)


if __name__ == "__main__":
    # Test du gestionnaire de localisation
    manager = LocationManager()
    
    print("Localisation actuelle:", manager.get_current_location())
    
    # Test de mise à jour
    success = manager.update_location(
        latitude=47.3769,
        longitude=8.5417,
        source="manual"
    )
    print(f"Mise à jour: {'✅' if success else '❌'}")
    
    # Test depuis téléphone
    phone_location = {
        "latitude": 47.3769,
        "longitude": 8.5417,
        "accuracy": 10
    }
    success = manager.update_from_phone(phone_location)
    print(f"Mise à jour depuis téléphone: {'✅' if success else '❌'}")
    
    print("Historique:", manager.get_location_history(3))