#!/usr/bin/env python3
"""
API REST pour la Gestion de Localisation NightScan Pi
Endpoints pour gérer la localisation du Pi via l'application mobile
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from pathlib import Path
import json

# Import du gestionnaire de localisation
from location_manager import location_manager

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application Flask
app = Flask(__name__)
CORS(app)  # Autoriser les requêtes cross-origin pour l'app mobile

# Configuration
app.config['SECRET_KEY'] = os.environ.get('LOCATION_API_SECRET', 'dev-location-key-change-in-prod')
app.config['JSON_SORT_KEYS'] = False

def create_response(success=True, data=None, message="", error=None):
    """Crée une réponse JSON standardisée"""
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat(),
        "message": message
    }
    
    if data is not None:
        response["data"] = data
    
    if error is not None:
        response["error"] = error
    
    return jsonify(response)

@app.route('/api/location', methods=['GET'])
def get_location():
    """Récupère la localisation actuelle du Pi"""
    try:
        location_data = location_manager.get_current_location()
        
        return create_response(
            success=True,
            data=location_data,
            message="Localisation récupérée avec succès"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la localisation: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

@app.route('/api/location', methods=['POST'])
def update_location():
    """Met à jour la localisation du Pi"""
    try:
        data = request.get_json()
        
        if not data:
            return create_response(
                success=False,
                error="Données JSON manquantes"
            ), 400
        
        # Validation des champs requis
        required_fields = ['latitude', 'longitude']
        for field in required_fields:
            if field not in data:
                return create_response(
                    success=False,
                    error=f"Champ requis manquant: {field}"
                ), 400
        
        # Extraction des données
        latitude = data['latitude']
        longitude = data['longitude']
        address = data.get('address', '')
        zone = data.get('zone', '')
        source = data.get('source', 'manual')
        
        # Validation des types
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return create_response(
                success=False,
                error="Latitude et longitude doivent être des nombres"
            ), 400
        
        # Mise à jour
        success = location_manager.update_location(
            latitude=latitude,
            longitude=longitude,
            address=address,
            zone=zone,
            source=source
        )
        
        if success:
            new_location = location_manager.get_current_location()
            return create_response(
                success=True,
                data=new_location,
                message="Localisation mise à jour avec succès"
            )
        else:
            return create_response(
                success=False,
                error="Échec de la mise à jour de la localisation"
            ), 500
            
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la localisation: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

@app.route('/api/location/phone', methods=['POST'])
def update_from_phone():
    """Met à jour la localisation depuis les données du téléphone"""
    try:
        data = request.get_json()
        
        if not data:
            return create_response(
                success=False,
                error="Données JSON manquantes"
            ), 400
        
        # Validation des champs requis
        required_fields = ['latitude', 'longitude']
        for field in required_fields:
            if field not in data:
                return create_response(
                    success=False,
                    error=f"Champ requis manquant: {field}"
                ), 400
        
        # Log des données reçues
        logger.info(f"Données de géolocalisation reçues du téléphone: {data}")
        
        # Mise à jour depuis les données du téléphone
        success = location_manager.update_from_phone(data)
        
        if success:
            new_location = location_manager.get_current_location()
            return create_response(
                success=True,
                data=new_location,
                message="Localisation mise à jour depuis le téléphone"
            )
        else:
            return create_response(
                success=False,
                error="Échec de la mise à jour depuis le téléphone"
            ), 500
            
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour depuis téléphone: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

@app.route('/api/location/history', methods=['GET'])
def get_location_history():
    """Récupère l'historique des localisations"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        if limit > 100:
            limit = 100  # Limite maximale
        
        history = location_manager.get_location_history(limit)
        
        return create_response(
            success=True,
            data={
                "history": history,
                "count": len(history)
            },
            message="Historique récupéré avec succès"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

@app.route('/api/location/coordinates', methods=['GET'])
def get_coordinates():
    """Récupère uniquement les coordonnées actuelles (endpoint simple)"""
    try:
        latitude, longitude = location_manager.get_coordinates()
        
        return create_response(
            success=True,
            data={
                "latitude": latitude,
                "longitude": longitude
            },
            message="Coordonnées récupérées"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des coordonnées: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

@app.route('/api/location/validate', methods=['POST'])
def validate_coordinates():
    """Valide des coordonnées GPS"""
    try:
        data = request.get_json()
        
        if not data:
            return create_response(
                success=False,
                error="Données JSON manquantes"
            ), 400
        
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if latitude is None or longitude is None:
            return create_response(
                success=False,
                error="Latitude et longitude requises"
            ), 400
        
        # Validation
        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except (ValueError, TypeError):
            return create_response(
                success=False,
                error="Latitude et longitude doivent être des nombres"
            ), 400
        
        # Vérification des limites
        valid = True
        errors = []
        
        if not (-90 <= latitude <= 90):
            valid = False
            errors.append("Latitude doit être entre -90 et 90")
        
        if not (-180 <= longitude <= 180):
            valid = False
            errors.append("Longitude doit être entre -180 et 180")
        
        return create_response(
            success=valid,
            data={
                "latitude": latitude,
                "longitude": longitude,
                "valid": valid,
                "errors": errors
            },
            message="Validation terminée"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la validation: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

@app.route('/api/location/export', methods=['GET'])
def export_location_data():
    """Exporte toutes les données de localisation"""
    try:
        data = location_manager.export_location_data()
        
        return create_response(
            success=True,
            data=data,
            message="Données exportées avec succès"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de l'export: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

@app.route('/api/location/status', methods=['GET'])
def get_location_status():
    """Récupère le statut complet de la localisation"""
    try:
        current_location = location_manager.get_current_location()
        history_count = len(location_manager.get_location_history(1))
        
        status = {
            "current_location": current_location,
            "has_history": history_count > 0,
            "last_updated": current_location.get("updated_at", ""),
            "source": current_location.get("source", "unknown"),
            "coordinates_valid": bool(current_location.get("latitude") and current_location.get("longitude"))
        }
        
        return create_response(
            success=True,
            data=status,
            message="Statut récupéré avec succès"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

@app.route('/api/location/reset', methods=['POST'])
def reset_location():
    """Remet la localisation à la valeur par défaut"""
    try:
        # Utiliser la localisation par défaut (Zurich)
        success = location_manager.update_location(
            latitude=46.9480,
            longitude=7.4474,
            address="Zurich, Switzerland",
            zone="Europe/Zurich",
            source="reset"
        )
        
        if success:
            new_location = location_manager.get_current_location()
            return create_response(
                success=True,
                data=new_location,
                message="Localisation remise à la valeur par défaut"
            )
        else:
            return create_response(
                success=False,
                error="Échec de la remise à zéro"
            ), 500
            
    except Exception as e:
        logger.error(f"Erreur lors de la remise à zéro: {e}")
        return create_response(
            success=False,
            error=str(e)
        ), 500

# Gestion des erreurs
@app.errorhandler(404)
def not_found(error):
    return create_response(
        success=False,
        error="Endpoint non trouvé"
    ), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return create_response(
        success=False,
        error="Méthode non autorisée"
    ), 405

@app.errorhandler(500)
def internal_error(error):
    return create_response(
        success=False,
        error="Erreur interne du serveur"
    ), 500

# Point d'entrée pour tests
@app.route('/api/location/test', methods=['GET'])
def test_endpoint():
    """Endpoint de test pour vérifier la connectivité"""
    return create_response(
        success=True,
        data={"version": "1.0", "service": "location_api"},
        message="API de localisation fonctionnelle"
    )

if __name__ == '__main__':
    logger.info("Démarrage de l'API de localisation NightScan Pi")
    
    # Afficher la localisation actuelle au démarrage
    current_location = location_manager.get_current_location()
    logger.info(f"Localisation actuelle: {current_location['latitude']}, {current_location['longitude']}")
    
    # Démarrer le serveur
    app.run(
        host='0.0.0.0',  # Accessible depuis l'app mobile
        port=5001,       # Port différent du service principal
        debug=True
    )