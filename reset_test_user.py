#!/usr/bin/env python3
"""
Script pour réinitialiser l'utilisateur test avec un hash de mot de passe fonctionnel
"""
import os
import sys
from werkzeug.security import generate_password_hash
from datetime import datetime, timezone

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration des variables d'environnement
os.environ.setdefault('SECRET_KEY', '296cff66ff3bf5591376b733dde9ff0f894e3c6ccf5d19faeb75ba3c145b61cc')
os.environ.setdefault('SQLALCHEMY_DATABASE_URI', 'postgresql://nightscan:nightscan_secure_password_2025@localhost:5432/nightscan')

from web.app import application, db, User

def reset_test_user():
    """Réinitialiser l'utilisateur test"""
    with application.app_context():
        # Supprimer l'utilisateur existant s'il existe
        existing_user = User.query.filter_by(username='testuser').first()
        if existing_user:
            db.session.delete(existing_user)
            db.session.commit()
            print("Utilisateur test existant supprimé.")
        
        # Créer un nouvel utilisateur avec un hash pbkdf2 qui rentre dans 128 caractères
        test_user = User(
            username='testuser',
            password_hash=generate_password_hash('testpass123', method='pbkdf2:sha256')
        )
        
        try:
            db.session.add(test_user)
            db.session.commit()
            print("\nUtilisateur test créé avec succès!")
            print("Username: testuser")
            print("Password: testpass123")
            print(f"Hash format: {test_user.password_hash[:30]}...")
        except Exception as e:
            db.session.rollback()
            print(f"Erreur lors de la création de l'utilisateur: {e}")
            return False
    
    return True

if __name__ == "__main__":
    if reset_test_user():
        print("\nVous pouvez maintenant vous connecter avec testuser/testpass123")
    else:
        print("\nÉchec de la réinitialisation de l'utilisateur test")