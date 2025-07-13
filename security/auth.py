"""
Authentication and Authorization Module

Handles user authentication, password management, and authorization.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from functools import wraps
import jwt
from flask import request, session, current_app, g, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import logging

logger = logging.getLogger(__name__)


class SecurityAuth:
    """Handles authentication and authorization."""
    
    def __init__(self, config):
        self.config = config
        self.failed_attempts = {}  # Track failed login attempts
        self.locked_accounts = {}  # Track locked accounts
    
    def init_app(self, app):
        """Initialize authentication with Flask app."""
        self.app = app
        
        # Set up before request handler
        @app.before_request
        def load_user():
            """Load user for each request."""
            g.user = self.get_current_user()
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return generate_password_hash(password, method='pbkdf2:sha256')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            password_hash: Hashed password
            
        Returns:
            True if password matches
        """
        return check_password_hash(password_hash, password)
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            Dict with 'valid' boolean and 'errors' list
        """
        errors = []
        min_length = self.config.security.password_min_length
        
        if len(password) < min_length:
            errors.append(f"Password must be at least {min_length} characters")
        
        # Additional checks if configured
        if hasattr(self.config.security, 'password_require_uppercase') and \
           self.config.security.password_require_uppercase:
            if not any(c.isupper() for c in password):
                errors.append("Password must contain at least one uppercase letter")
        
        if hasattr(self.config.security, 'password_require_lowercase') and \
           self.config.security.password_require_lowercase:
            if not any(c.islower() for c in password):
                errors.append("Password must contain at least one lowercase letter")
        
        if hasattr(self.config.security, 'password_require_digits') and \
           self.config.security.password_require_digits:
            if not any(c.isdigit() for c in password):
                errors.append("Password must contain at least one digit")
        
        if hasattr(self.config.security, 'password_require_special') and \
           self.config.security.password_require_special:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                errors.append("Password must contain at least one special character")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def create_session(self, user_id: str, remember: bool = False) -> str:
        """
        Create a new session for a user.
        
        Args:
            user_id: User identifier
            remember: Whether to create a persistent session
            
        Returns:
            Session token
        """
        # Generate session token
        session_token = secrets.token_urlsafe(32)
        
        # Store in session
        session['user_id'] = user_id
        session['session_token'] = session_token
        session['created_at'] = datetime.utcnow().isoformat()
        
        if remember:
            session.permanent = True
        
        # Log session creation
        logger.info(f"Session created for user {user_id}")
        
        return session_token
    
    def destroy_session(self) -> None:
        """Destroy the current session."""
        if 'user_id' in session:
            logger.info(f"Session destroyed for user {session['user_id']}")
        
        session.clear()
    
    def is_authenticated(self) -> bool:
        """Check if current user is authenticated."""
        return 'user_id' in session and 'session_token' in session
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Get current authenticated user.
        
        Returns:
            User dict or None
        """
        if not self.is_authenticated():
            return None
        
        # Return basic user info from session
        return {
            'id': session.get('user_id'),
            'session_token': session.get('session_token'),
            'created_at': session.get('created_at')
        }
    
    def check_lockout(self, identifier: str) -> bool:
        """
        Check if an account/IP is locked out.
        
        Args:
            identifier: Username or IP address
            
        Returns:
            True if locked out
        """
        if identifier in self.locked_accounts:
            lockout_until = self.locked_accounts[identifier]
            if datetime.utcnow() < lockout_until:
                return True
            else:
                # Lockout expired
                del self.locked_accounts[identifier]
                if identifier in self.failed_attempts:
                    del self.failed_attempts[identifier]
        
        return False
    
    def record_failed_attempt(self, identifier: str) -> None:
        """
        Record a failed login attempt.
        
        Args:
            identifier: Username or IP address
        """
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append(datetime.utcnow())
        
        # Clean old attempts
        cutoff = datetime.utcnow() - timedelta(seconds=self.config.security.lockout_window)
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        # Check if should lock account
        if len(self.failed_attempts[identifier]) >= self.config.security.lockout_threshold:
            lockout_duration = timedelta(seconds=self.config.security.lockout_window)
            self.locked_accounts[identifier] = datetime.utcnow() + lockout_duration
            logger.warning(f"Account locked: {identifier}")
    
    def clear_failed_attempts(self, identifier: str) -> None:
        """Clear failed attempts for an identifier."""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]
        if identifier in self.locked_accounts:
            del self.locked_accounts[identifier]
    
    def generate_csrf_token(self) -> str:
        """Generate a new CSRF token."""
        if 'csrf_token' not in session:
            session['csrf_token'] = secrets.token_urlsafe(32)
        return session['csrf_token']
    
    def validate_csrf_token(self, token: str) -> bool:
        """
        Validate a CSRF token.
        
        Args:
            token: Token to validate
            
        Returns:
            True if valid
        """
        return 'csrf_token' in session and secrets.compare_digest(
            session['csrf_token'], token
        )
    
    def generate_jwt_token(self, user_id: str, expiry_hours: int = 24) -> str:
        """
        Generate a JWT token for API authentication.
        
        Args:
            user_id: User identifier
            expiry_hours: Token validity in hours
            
        Returns:
            JWT token
        """
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expiry_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(
            payload,
            self.config.security.secret_key,
            algorithm='HS256'
        )
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Decoded payload or None
        """
        try:
            payload = jwt.decode(
                token,
                self.config.security.secret_key,
                algorithms=['HS256']
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
    
    # Decorators
    def require_auth(self, f: Callable) -> Callable:
        """Decorator to require authentication."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not self.is_authenticated():
                # Check for API token
                auth_header = request.headers.get('Authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    token = auth_header.split(' ')[1]
                    payload = self.validate_jwt_token(token)
                    if payload:
                        g.user = {'id': payload['user_id'], 'from_jwt': True}
                        return f(*args, **kwargs)
                
                return jsonify({'error': 'Authentication required'}), 401
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def require_csrf(self, f: Callable) -> Callable:
        """Decorator to require CSRF protection."""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.method in ['POST', 'PUT', 'DELETE']:
                token = request.form.get('csrf_token') or \
                        request.headers.get('X-CSRF-Token')
                
                if not token or not self.validate_csrf_token(token):
                    return jsonify({'error': 'Invalid CSRF token'}), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def require_role(self, role: str) -> Callable:
        """
        Decorator to require a specific role.
        
        Args:
            role: Required role
        """
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            @self.require_auth
            def decorated_function(*args, **kwargs):
                user = g.get('user')
                if not user or user.get('role') != role:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                return f(*args, **kwargs)
            
            return decorated_function
        
        return decorator