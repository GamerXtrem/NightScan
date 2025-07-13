"""JWT Token Management for NightScan API Gateway"""

import os
import jwt
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, Any
from functools import wraps
from flask import request, jsonify, current_app, g
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT Token payload structure"""
    user_id: int
    username: str
    email: str
    roles: list
    plan_type: str
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation
    token_type: str  # 'access' or 'refresh'


class JWTManager:
    """Manages JWT token generation, validation, and revocation"""
    
    def __init__(self, secret_key: Optional[str] = None, 
                 access_token_expires: int = 3600,  # 1 hour
                 refresh_token_expires: int = 2592000):  # 30 days
        """Initialize JWT Manager
        
        Args:
            secret_key: Secret key for signing tokens
            access_token_expires: Access token expiration in seconds
            refresh_token_expires: Refresh token expiration in seconds
        """
        self.secret_key = secret_key or os.environ.get('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        self.access_token_expires = timedelta(seconds=access_token_expires)
        self.refresh_token_expires = timedelta(seconds=refresh_token_expires)
        self.algorithm = 'HS256'
        self.revoked_tokens = set()  # In production, use Redis
    
    def generate_tokens(self, user_data: Dict[str, Any]) -> Tuple[str, str]:
        """Generate both access and refresh tokens
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            Tuple of (access_token, refresh_token)
        """
        now = datetime.now(timezone.utc)
        
        # Common claims
        base_claims = {
            'user_id': user_data['id'],
            'username': user_data['username'],
            'email': user_data.get('email', ''),
            'roles': user_data.get('roles', ['user']),
            'plan_type': user_data.get('plan_type', 'free'),
            'iat': now,
        }
        
        # Access token
        access_jti = secrets.token_urlsafe(16)
        access_claims = {
            **base_claims,
            'exp': now + self.access_token_expires,
            'jti': access_jti,
            'token_type': 'access'
        }
        access_token = jwt.encode(access_claims, self.secret_key, algorithm=self.algorithm)
        
        # Refresh token
        refresh_jti = secrets.token_urlsafe(16)
        refresh_claims = {
            **base_claims,
            'exp': now + self.refresh_token_expires,
            'jti': refresh_jti,
            'token_type': 'refresh'
        }
        refresh_token = jwt.encode(refresh_claims, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"Generated tokens for user {user_data['username']} (ID: {user_data['id']})")
        
        return access_token, refresh_token
    
    def validate_token(self, token: str, token_type: str = 'access') -> Optional[TokenPayload]:
        """Validate and decode a JWT token
        
        Args:
            token: JWT token string
            token_type: Expected token type ('access' or 'refresh')
            
        Returns:
            TokenPayload if valid, None otherwise
        """
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            if payload.get('jti') in self.revoked_tokens:
                logger.warning(f"Attempted to use revoked token {payload.get('jti')}")
                return None
            
            # Verify token type
            if payload.get('token_type') != token_type:
                logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('token_type')}")
                return None
            
            # Create TokenPayload object
            token_payload = TokenPayload(
                user_id=payload['user_id'],
                username=payload['username'],
                email=payload.get('email', ''),
                roles=payload.get('roles', []),
                plan_type=payload.get('plan_type', 'free'),
                exp=datetime.fromtimestamp(payload['exp'], tz=timezone.utc),
                iat=datetime.fromtimestamp(payload['iat'], tz=timezone.utc),
                jti=payload['jti'],
                token_type=payload['token_type']
            )
            
            return token_payload
            
        except jwt.ExpiredSignatureError:
            logger.info("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token if valid, None otherwise
        """
        # Validate refresh token
        token_payload = self.validate_token(refresh_token, token_type='refresh')
        if not token_payload:
            return None
        
        # Generate new access token
        now = datetime.now(timezone.utc)
        access_jti = secrets.token_urlsafe(16)
        
        access_claims = {
            'user_id': token_payload.user_id,
            'username': token_payload.username,
            'email': token_payload.email,
            'roles': token_payload.roles,
            'plan_type': token_payload.plan_type,
            'iat': now,
            'exp': now + self.access_token_expires,
            'jti': access_jti,
            'token_type': 'access'
        }
        
        access_token = jwt.encode(access_claims, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Refreshed access token for user {token_payload.username}")
        
        return access_token
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token by adding its JTI to revocation list
        
        Args:
            token: JWT token to revoke
            
        Returns:
            True if successfully revoked, False otherwise
        """
        try:
            # Decode without verification to get JTI
            unverified = jwt.decode(token, options={"verify_signature": False})
            jti = unverified.get('jti')
            
            if jti:
                self.revoked_tokens.add(jti)
                # In production, store in Redis with expiration
                logger.info(f"Revoked token {jti}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False
    
    def extract_token_from_request(self, request) -> Optional[str]:
        """Extract JWT token from request
        
        Args:
            request: Flask request object
            
        Returns:
            Token string if found, None otherwise
        """
        # Check Authorization header
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            return auth_header[7:]
        
        # Check cookie
        token = request.cookies.get('jwt_token')
        if token:
            return token
        
        # Check query parameter (not recommended for production)
        token = request.args.get('jwt')
        if token:
            return token
        
        return None


# Global JWT manager instance
_jwt_manager: Optional[JWTManager] = None


def get_jwt_manager() -> JWTManager:
    """Get or create global JWT manager instance"""
    global _jwt_manager
    
    if _jwt_manager is None:
        config = current_app.config if current_app else {}
        _jwt_manager = JWTManager(
            secret_key=config.get('JWT_SECRET_KEY'),
            access_token_expires=config.get('JWT_ACCESS_TOKEN_EXPIRES', 3600),
            refresh_token_expires=config.get('JWT_REFRESH_TOKEN_EXPIRES', 2592000)
        )
    
    return _jwt_manager


def jwt_required(f):
    """Decorator to require valid JWT token for route access"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        jwt_manager = get_jwt_manager()
        
        # Extract token from request
        token = jwt_manager.extract_token_from_request(request)
        if not token:
            return jsonify({
                'success': False,
                'error': 'Missing authentication token'
            }), 401
        
        # Validate token
        token_payload = jwt_manager.validate_token(token)
        if not token_payload:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired token'
            }), 401
        
        # Store user info in g for route access
        g.jwt_user = {
            'id': token_payload.user_id,
            'username': token_payload.username,
            'email': token_payload.email,
            'roles': token_payload.roles,
            'plan_type': token_payload.plan_type
        }
        
        return f(*args, **kwargs)
    
    return decorated_function


def jwt_optional(f):
    """Decorator to optionally validate JWT token if present"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        jwt_manager = get_jwt_manager()
        
        # Extract token from request
        token = jwt_manager.extract_token_from_request(request)
        if token:
            # Validate token but don't fail if invalid
            token_payload = jwt_manager.validate_token(token)
            if token_payload:
                g.jwt_user = {
                    'id': token_payload.user_id,
                    'username': token_payload.username,
                    'email': token_payload.email,
                    'roles': token_payload.roles,
                    'plan_type': token_payload.plan_type
                }
            else:
                g.jwt_user = None
        else:
            g.jwt_user = None
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_roles(*required_roles):
    """Decorator to require specific roles for route access"""
    def decorator(f):
        @wraps(f)
        @jwt_required
        def decorated_function(*args, **kwargs):
            user_roles = g.jwt_user.get('roles', [])
            
            # Check if user has any of the required roles
            if not any(role in user_roles for role in required_roles):
                return jsonify({
                    'success': False,
                    'error': 'Insufficient permissions'
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator