#!/usr/bin/env python3
"""
Test du système de nommage des fichiers NightScan
Vérifie la génération, le parsing et la compatibilité
"""

import os
import sys
import logging
import tempfile
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from .filename_utils import FilenameGenerator, FilenameParser, create_audio_filename, create_image_filename
from .location_manager import LocationManager

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FilenameSystemTester:
    """Testeur complet du système de nommage"""
    
    def __init__(self):
        self.results = []
        self.location_manager = LocationManager()
    
    def test_filename_generation(self):
        """Test de génération de noms de fichiers"""
        logger.info("🧪 Test de génération de noms de fichiers")
        
        # Test avec location manager
        generator = FilenameGenerator(self.location_manager)
        
        # Test audio
        audio_name = generator.generate_audio_filename()
        logger.info(f"Audio généré: {audio_name}")
        
        # Test image
        image_name = generator.generate_image_filename()
        logger.info(f"Image généré: {image_name}")
        
        # Test avec coordonnées personnalisées
        custom_audio = generator.generate_audio_filename(
            latitude=47.3769,
            longitude=8.5417
        )
        logger.info(f"Audio personnalisé: {custom_audio}")
        
        # Vérifier le format
        self.results.append(("Audio format", audio_name.startswith("AUD_")))
        self.results.append(("Image format", image_name.startswith("IMG_")))
        self.results.append(("Extension audio", audio_name.endswith(".wav")))
        self.results.append(("Extension image", image_name.endswith(".jpg")))
        
        return True
    
    def test_filename_parsing(self):
        """Test de parsing de noms de fichiers"""
        logger.info("🧪 Test de parsing de noms de fichiers")
        
        parser = FilenameParser()
        
        # Test du nouveau format
        test_files = [
            "AUD_20240109_143045_4695_0745.wav",
            "IMG_20240109_143045_4695_0745.jpg",
            "VID_20240109_143045_4695_0745.mp4",
            # Format invalide
            "invalid_file.txt"
        ]
        
        for filename in test_files:
            parsed = parser.parse_filename(filename)
            logger.info(f"Fichier: {filename}")
            logger.info(f"  Type: {parsed['type']}")
            logger.info(f"  Format: {parsed['format']}")
            logger.info(f"  Timestamp: {parsed['timestamp']}")
            logger.info(f"  GPS: {parsed['latitude']}, {parsed['longitude']}")
            
            # Vérifications
            if filename.startswith("AUD_"):
                self.results.append((f"Parse nouveau audio {filename}", parsed['type'] == 'audio'))
            elif filename.startswith("IMG_"):
                self.results.append((f"Parse nouveau image {filename}", parsed['type'] == 'image'))
            elif filename.startswith("VID_"):
                self.results.append((f"Parse nouveau video {filename}", parsed['type'] == 'video'))
        
        return True
    
    def test_coordinate_encoding(self):
        """Test d'encodage/décodage des coordonnées"""
        logger.info("🧪 Test d'encodage/décodage des coordonnées")
        
        generator = FilenameGenerator()
        
        # Test de coordonnées connues
        test_coords = [
            (46.9480, 7.4474),  # Zurich
            (47.3769, 8.5417),  # Zurich Nord
            (45.5017, 6.0863),  # Chamonix
            (0.0, 0.0),         # Équateur/Greenwich
            (-33.8688, 151.2093),  # Sydney
            (90.0, 180.0),      # Pôle Nord, longitude max
            (-90.0, -180.0)     # Pôle Sud, longitude min
        ]
        
        for lat, lon in test_coords:
            # Générer un nom avec ces coordonnées
            filename = generator.generate_filename('audio', 'wav', latitude=lat, longitude=lon)
            logger.info(f"Coordonnées ({lat}, {lon}) -> {filename}")
            
            # Parser et vérifier
            parsed = FilenameParser.parse_filename(filename)
            parsed_lat = parsed['latitude']
            parsed_lon = parsed['longitude']
            
            if parsed_lat is not None and parsed_lon is not None:
                lat_diff = abs(lat - parsed_lat)
                lon_diff = abs(lon - parsed_lon)
                
                # Tolérance de 0.1 degrés (environ 10km)
                lat_ok = lat_diff < 0.1
                lon_ok = lon_diff < 0.1
                
                logger.info(f"  Décodé: ({parsed_lat}, {parsed_lon})")
                logger.info(f"  Différence: ({lat_diff:.4f}, {lon_diff:.4f})")
                logger.info(f"  Précision: {'✅' if lat_ok and lon_ok else '❌'}")
                
                self.results.append((f"Coordonnées {lat},{lon}", lat_ok and lon_ok))
            else:
                logger.warning(f"  Échec du décodage")
                self.results.append((f"Coordonnées {lat},{lon}", False))
        
        return True
    
    def test_compatibility_functions(self):
        """Test des fonctions de compatibilité"""
        logger.info("🧪 Test des fonctions de compatibilité")
        
        # Test des fonctions utilitaires
        audio_name = create_audio_filename()
        image_name = create_image_filename()
        
        logger.info(f"Fonction utilitaire audio: {audio_name}")
        logger.info(f"Fonction utilitaire image: {image_name}")
        
        self.results.append(("Fonction audio", audio_name.startswith("AUD_")))
        self.results.append(("Fonction image", image_name.startswith("IMG_")))
        
        return True
    
    def test_location_integration(self):
        """Test d'intégration avec le système de localisation"""
        logger.info("🧪 Test d'intégration avec le système de localisation")
        
        try:
            # Test avec location manager
            generator = FilenameGenerator(self.location_manager)
            
            # Générer des noms avec la localisation actuelle
            audio_name = generator.generate_audio_filename()
            image_name = generator.generate_image_filename()
            
            logger.info(f"Audio avec localisation: {audio_name}")
            logger.info(f"Image avec localisation: {image_name}")
            
            # Vérifier que les coordonnées sont intégrées
            parsed_audio = FilenameParser.parse_filename(audio_name)
            parsed_image = FilenameParser.parse_filename(image_name)
            
            has_gps_audio = parsed_audio['latitude'] is not None and parsed_audio['longitude'] is not None
            has_gps_image = parsed_image['latitude'] is not None and parsed_image['longitude'] is not None
            
            self.results.append(("GPS intégré audio", has_gps_audio))
            self.results.append(("GPS intégré image", has_gps_image))
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur d'intégration localisation: {e}")
            self.results.append(("Intégration localisation", False))
            return False
    
    def test_file_operations(self):
        """Test avec de vrais fichiers"""
        logger.info("🧪 Test avec opérations fichiers")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Créer des fichiers avec nouveaux noms
                generator = FilenameGenerator(self.location_manager)
                
                audio_filename = generator.generate_audio_filename()
                image_filename = generator.generate_image_filename()
                
                audio_path = temp_path / audio_filename
                image_path = temp_path / image_filename
                
                # Créer des fichiers factices
                audio_path.write_text("fake audio content")
                image_path.write_text("fake image content")
                
                # Vérifier que les fichiers existent
                audio_exists = audio_path.exists()
                image_exists = image_path.exists()
                
                logger.info(f"Fichier audio créé: {audio_exists}")
                logger.info(f"Fichier image créé: {image_exists}")
                
                # Lister les fichiers et les parser
                for file_path in temp_path.iterdir():
                    parsed = FilenameParser.parse_filename(file_path.name)
                    logger.info(f"Fichier: {file_path.name} -> Type: {parsed['type']}")
                
                self.results.append(("Création fichier audio", audio_exists))
                self.results.append(("Création fichier image", image_exists))
                
                return True
                
        except Exception as e:
            logger.error(f"Erreur test fichiers: {e}")
            self.results.append(("Test fichiers", False))
            return False
    
    def run_all_tests(self):
        """Exécute tous les tests"""
        logger.info("🚀 Démarrage des tests du système de nommage")
        
        tests = [
            ("Génération de noms", self.test_filename_generation),
            ("Parsing de noms", self.test_filename_parsing),
            ("Encodage coordonnées", self.test_coordinate_encoding),
            ("Fonctions compatibilité", self.test_compatibility_functions),
            ("Intégration localisation", self.test_location_integration),
            ("Opérations fichiers", self.test_file_operations)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Test: {test_name}")
                logger.info(f"{'='*50}")
                
                result = test_func()
                if result:
                    logger.info(f"✅ {test_name} PASSÉ")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} ÉCHOUÉ")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"❌ {test_name} ERREUR: {e}")
                failed += 1
        
        # Résumé des résultats détaillés
        logger.info(f"\n{'='*50}")
        logger.info("RÉSULTATS DÉTAILLÉS")
        logger.info(f"{'='*50}")
        
        detail_passed = 0
        detail_failed = 0
        
        for test_name, result in self.results:
            status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
            logger.info(f"{test_name}: {status}")
            
            if result:
                detail_passed += 1
            else:
                detail_failed += 1
        
        # Résumé final
        logger.info(f"\n{'='*50}")
        logger.info("RÉSUMÉ FINAL")
        logger.info(f"{'='*50}")
        logger.info(f"Tests généraux: {passed} passés, {failed} échoués")
        logger.info(f"Tests détaillés: {detail_passed} passés, {detail_failed} échoués")
        
        overall_success = failed == 0 and detail_failed == 0
        
        if overall_success:
            logger.info("🎉 TOUS LES TESTS SONT PASSÉS!")
        else:
            logger.error("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        
        return overall_success
    
    def generate_test_report(self):
        """Génère un rapport de test"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'summary': {
                'total_tests': len(self.results),
                'passed': sum(1 for _, result in self.results if result),
                'failed': sum(1 for _, result in self.results if not result)
            }
        }
        
        return report


def main():
    """Fonction principale"""
    print("🧪 Test du Système de Nommage des Fichiers NightScan")
    print("="*60)
    
    tester = FilenameSystemTester()
    success = tester.run_all_tests()
    
    # Générer le rapport
    report = tester.generate_test_report()
    
    # Sauvegarder le rapport
    try:
        report_path = Path("filename_test_report.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Rapport sauvegardé: {report_path}")
    except Exception as e:
        print(f"⚠️  Erreur sauvegarde rapport: {e}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())