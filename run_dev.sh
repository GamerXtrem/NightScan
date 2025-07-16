#!/bin/bash
# Development server script for NightScan

echo "Starting NightScan development server..."

# Kill any existing gunicorn processes
pkill -f gunicorn

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
export SQLALCHEMY_DATABASE_URI="sqlite:///nightscan.db"
export PREDICT_API_URL="http://localhost:8002/api/predict"
export FLASK_ENV=development
export NIGHTSCAN_ENV=development
export TALISMAN_FORCE_HTTPS=False

echo "Environment configured."
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop the server."

# Start Flask development server instead of gunicorn for easier debugging
export FLASK_APP=web.app:application
flask run --host=0.0.0.0 --port=8000