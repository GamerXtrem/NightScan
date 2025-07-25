#!/usr/bin/env python3
"""
Script to create a test user in the NightScan database
"""
import os
import sys

from werkzeug.security import generate_password_hash
from datetime import datetime, timezone

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables if not already set
os.environ.setdefault('SECRET_KEY', '296cff66ff3bf5591376b733dde9ff0f894e3c6ccf5d19faeb75ba3c145b61cc')
os.environ.setdefault('SQLALCHEMY_DATABASE_URI', 'postgresql://nightscan:nightscan_secure_password_2025@localhost:5432/nightscan')

from web.app import application, db, User

def create_test_user():
    """Create a test user account"""
    with application.app_context():
        # Check if user already exists
        existing_user = User.query.filter_by(username='testuser').first()
        if existing_user:
            print("User 'testuser' already exists!")
            return
        
        # Create new user with pbkdf2:sha256 hash method
        test_user = User(
            username='testuser',
            password_hash=generate_password_hash('testpass123', method='pbkdf2:sha256')
        )
        
        try:
            db.session.add(test_user)
            db.session.commit()
            print("Successfully created test user!")
            print("Username: testuser")
            print("Password: testpass123")
            print("Email: testuser@example.com")
        except Exception as e:
            db.session.rollback()
            print(f"Error creating user: {e}")

if __name__ == "__main__":
    create_test_user()