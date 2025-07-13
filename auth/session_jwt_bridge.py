"""JWT authentication integration for Flask application"""

from functools import wraps
from flask import request, g, current_app
from flask_login import current_user
import logging

from auth.jwt_manager import get_jwt_manager

logger = logging.getLogger(__name__)


class SessionJWTBridge:
    """Handles JWT and session authentication for Flask routes"""
    
    @staticmethod
    def get_current_user():
        """Get current user from either JWT or session
        
        Returns:
            User object or None
        """
        # First, check if we have a JWT user in g
        if hasattr(g, 'jwt_user') and g.jwt_user:
            # Convert JWT user data to User object
            from web.app import User
            return User.query.get(g.jwt_user['id'])
        
        # Fall back to Flask-Login current_user
        if current_user and current_user.is_authenticated:
            return current_user
        
        # Try to extract and validate JWT
        jwt_manager = get_jwt_manager()
        token = jwt_manager.extract_token_from_request(request)
        
        if token:
            token_payload = jwt_manager.validate_token(token)
            if token_payload:
                from web.app import User
                user = User.query.get(token_payload.user_id)
                if user:
                    # Store in g for this request
                    g.jwt_user = {
                        'id': user.id,
                        'username': user.username,
                        'email': getattr(user, 'email', ''),
                        'roles': getattr(user, 'roles', ['user']),
                        'plan_type': getattr(user, 'plan_type', 'free')
                    }
                    return user
        
        return None
    
    @staticmethod
    def unified_login_required(f):
        """Decorator that accepts either session or JWT authentication"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user = SessionJWTBridge.get_current_user()
            
            if not user:
                # Check if this is an API request (wants JSON)
                if request.is_json or request.path.startswith('/api/'):
                    return {
                        'success': False,
                        'error': 'Authentication required'
                    }, 401
                else:
                    # Redirect to login for web requests
                    from flask import redirect, url_for
                    return redirect(url_for('login'))
            
            # Store user in g for easy access
            g.current_user = user
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    @staticmethod
    def convert_session_to_jwt():
        """Convert an authenticated session to JWT tokens
        
        Returns:
            Dict with tokens or None
        """
        if not current_user or not current_user.is_authenticated:
            return None
        
        jwt_manager = get_jwt_manager()
        
        user_data = {
            'id': current_user.id,
            'username': current_user.username,
            'email': getattr(current_user, 'email', ''),
            'roles': getattr(current_user, 'roles', ['user']),
            'plan_type': getattr(current_user, 'plan_type', 'free')
        }
        
        access_token, refresh_token = jwt_manager.generate_tokens(user_data)
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': current_app.config.get('JWT_ACCESS_TOKEN_EXPIRES', 3600)
        }
    
    @staticmethod
    def inject_jwt_in_response(response):
        """Middleware to inject JWT token in response if user is authenticated via session
        
        Args:
            response: Flask response object
            
        Returns:
            Modified response
        """
        # Only for API requests from authenticated sessions
        if (request.path.startswith('/api/') and 
            current_user and 
            current_user.is_authenticated and
            'Authorization' not in request.headers):
            
            # Generate JWT for this session
            tokens = SessionJWTBridge.convert_session_to_jwt()
            
            if tokens:
                # Add token to response headers
                response.headers['X-Auth-Token'] = tokens['access_token']
                response.headers['X-Auth-Token-Type'] = 'Bearer'
                response.headers['X-Auth-Token-Expires'] = str(tokens['expires_in'])
                
                logger.debug(f"Injected JWT for session user: {current_user.username}")
        
        return response


def init_session_jwt_bridge(app):
    """Initialize the session-JWT bridge for the Flask app
    
    Args:
        app: Flask application instance
    """
    @app.before_request
    def before_request():
        """Extract user from JWT if present"""
        # Skip for static files
        if request.path.startswith('/static/'):
            return
        
        jwt_manager = get_jwt_manager()
        token = jwt_manager.extract_token_from_request(request)
        
        if token:
            token_payload = jwt_manager.validate_token(token)
            if token_payload:
                g.jwt_user = {
                    'id': token_payload.user_id,
                    'username': token_payload.username,
                    'email': token_payload.email,
                    'roles': token_payload.roles,
                    'plan_type': token_payload.plan_type
                }
    
    @app.after_request
    def after_request(response):
        """Inject JWT token for session-authenticated API requests"""
        return SessionJWTBridge.inject_jwt_in_response(response)
    
    logger.info("Session-JWT bridge initialized")