"""
Application extensions configuration for NightScan.
This file handles initialization of Flask extensions that need to be imported in multiple places.
"""

from flask_mail import Mail

# Initialize Flask-Mail
mail = Mail()

def init_mail(app):
    """Initialize Flask-Mail with app configuration."""
    # Mail configuration
    app.config['MAIL_SERVER'] = app.config.get('MAIL_SERVER', 'localhost')
    app.config['MAIL_PORT'] = app.config.get('MAIL_PORT', 587)
    app.config['MAIL_USE_TLS'] = app.config.get('MAIL_USE_TLS', True)
    app.config['MAIL_USE_SSL'] = app.config.get('MAIL_USE_SSL', False)
    app.config['MAIL_USERNAME'] = app.config.get('MAIL_USERNAME')
    app.config['MAIL_PASSWORD'] = app.config.get('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = app.config.get('MAIL_DEFAULT_SENDER', 'noreply@nightscan.app')
    
    mail.init_app(app)