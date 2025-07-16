"""
Secure Authentication System for NightScan
Implements modern authentication security practices.
"""

import hashlib
import secrets
import time
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import bcrypt
import jwt
from functools import wraps

class SecureAuth:
    """Secure authentication handler."""
    
    def __init__(self, jwt_secret: str, session_timeout: int = 3600):
        self.jwt_secret = jwt_secret
        self.session_timeout = session_timeout
        self.failed_attempts = {}  # In production, use Redis
        self.max_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
        
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
            
    def generate_secure_token(self, user_id: int, additional_claims: Optional[Dict] = None) -> str:
        """Generate secure JWT token."""
        payload = {
            'user_id': user_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.session_timeout),
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }
        
        if additional_claims:
            payload.update(additional_claims)
            
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
            
    def check_rate_limit(self, identifier: str) -> Tuple[bool, int]:
        """Check if identifier is rate limited."""
        current_time = time.time()
        
        if identifier in self.failed_attempts:
            attempts, first_attempt = self.failed_attempts[identifier]
            
            # Reset if lockout period has passed
            if current_time - first_attempt > self.lockout_duration:
                del self.failed_attempts[identifier]
                return True, 0
                
            # Check if still locked out
            if attempts >= self.max_attempts:
                remaining = self.lockout_duration - (current_time - first_attempt)
                return False, int(remaining)
                
        return True, 0
        
    def record_failed_attempt(self, identifier: str):
        """Record failed authentication attempt."""
        current_time = time.time()
        
        if identifier in self.failed_attempts:
            attempts, first_attempt = self.failed_attempts[identifier]
            # Reset if enough time has passed
            if current_time - first_attempt > self.lockout_duration:
                self.failed_attempts[identifier] = (1, current_time)
            else:
                self.failed_attempts[identifier] = (attempts + 1, first_attempt)
        else:
            self.failed_attempts[identifier] = (1, current_time)
            
    def clear_failed_attempts(self, identifier: str):
        """Clear failed attempts for identifier."""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]
            
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token."""
        timestamp = str(int(time.time()))
        data = f"{session_id}:{timestamp}"
        signature = hashlib.sha256((data + self.jwt_secret).encode()).hexdigest()
        return f"{timestamp}.{signature}"
        
    def verify_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Verify CSRF token."""
        try:
            timestamp_str, signature = token.split('.', 1)
            timestamp = int(timestamp_str)
            
            # Check token age
            if time.time() - timestamp > max_age:
                return False
                
            # Verify signature
            data = f"{session_id}:{timestamp_str}"
            expected_signature = hashlib.sha256((data + self.jwt_secret).encode()).hexdigest()
            
            return secrets.compare_digest(signature, expected_signature)
        except (ValueError, TypeError):
            return False

class SecureSession:
    """Secure session management."""
    
    def __init__(self):
        self.sessions = {}  # In production, use Redis
        
    def create_session(self, user_id: int) -> str:
        """Create new session."""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'csrf_token': secrets.token_urlsafe(32)
        }
        return session_id
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        session = self.sessions.get(session_id)
        if session:
            # Check if session is expired (24 hours)
            if datetime.utcnow() - session['last_activity'] > timedelta(hours=24):
                self.destroy_session(session_id)
                return None
            # Update last activity
            session['last_activity'] = datetime.utcnow()
        return session
        
    def destroy_session(self, session_id: str):
        """Destroy session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
    def regenerate_session_id(self, old_session_id: str) -> Optional[str]:
        """Regenerate session ID (prevent session fixation)."""
        session_data = self.sessions.get(old_session_id)
        if session_data:
            new_session_id = secrets.token_urlsafe(32)
            self.sessions[new_session_id] = session_data
            del self.sessions[old_session_id]
            return new_session_id
        return None

# Decorators for Flask routes
def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import request, jsonify, current_app
        
        # Get token from header
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization header'}), 401
            
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        # Verify token
        auth = SecureAuth(current_app.config['JWT_SECRET_KEY'])
        payload = auth.verify_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
            
        # Add user info to request context
        request.user_id = payload['user_id']
        
        return f(*args, **kwargs)
    return decorated_function

def require_csrf(f):
    """Decorator to require CSRF token."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import request, jsonify, session, current_app
        
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            csrf_token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
            
            if not csrf_token:
                return jsonify({'error': 'CSRF token required'}), 403
                
            auth = SecureAuth(current_app.config['JWT_SECRET_KEY'])
            if not auth.verify_csrf_token(csrf_token, session.get('session_id', '')):
                return jsonify({'error': 'Invalid CSRF token'}), 403
                
        return f(*args, **kwargs)
    return decorated_function

# Global instances
_auth = None
_session_manager = None

def get_auth() -> SecureAuth:
    """Get global auth instance."""
    global _auth
    if _auth is None:
        from secure_secrets import get_secret
        jwt_secret = get_secret('JWT_SECRET_KEY')
        if not jwt_secret:
            raise ValueError("JWT_SECRET_KEY not configured")
        _auth = SecureAuth(jwt_secret)
    return _auth

def get_session_manager() -> SecureSession:
    """Get global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SecureSession()
    return _session_manager
