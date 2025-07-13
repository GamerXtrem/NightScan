"""JWT Authentication routes for NightScan API"""

from flask import Blueprint, request, jsonify, current_app
from flask_login import login_user, logout_user
from werkzeug.security import check_password_hash
from marshmallow import Schema, fields, ValidationError
import logging
from datetime import datetime

from auth.jwt_manager import get_jwt_manager
from metrics import track_request_metrics, record_failed_login
from cache_manager import invalidate_user_cache

logger = logging.getLogger(__name__)

# Create auth blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


# Request/Response Schemas
class LoginSchema(Schema):
    """Schema for login request"""
    username = fields.Str(required=True, validate=lambda x: len(x) >= 3)
    password = fields.Str(required=True, validate=lambda x: len(x) >= 6)
    remember_me = fields.Bool(load_default=False)


class RegisterSchema(Schema):
    """Schema for registration request"""
    username = fields.Str(required=True, validate=lambda x: 3 <= len(x) <= 50)
    password = fields.Str(required=True, validate=lambda x: len(x) >= 8)
    email = fields.Email(required=True)
    plan_type = fields.Str(load_default='free', validate=lambda x: x in ['free', 'premium', 'enterprise'])


class RefreshTokenSchema(Schema):
    """Schema for token refresh request"""
    refresh_token = fields.Str(required=True)


class TokenResponseSchema(Schema):
    """Schema for token response"""
    access_token = fields.Str(required=True)
    refresh_token = fields.Str(required=True)
    token_type = fields.Str(dump_default='Bearer')
    expires_in = fields.Int(required=True)
    user = fields.Dict(required=True)


@auth_bp.route('/login', methods=['POST'])
@track_request_metrics
def login():
    """Authenticate user and return JWT tokens
    
    ---
    tags:
      - Authentication
    summary: User login
    description: Authenticate with username/password and receive JWT tokens
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/LoginSchema'
    responses:
      200:
        description: Successful authentication
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TokenResponseSchema'
      401:
        description: Invalid credentials
      429:
        description: Too many failed attempts
    """
    try:
        # Validate request data
        schema = LoginSchema()
        data = schema.load(request.get_json())
        
        # Import here to avoid circular imports
        from web.app import User, db
        
        # Find user
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not user.check_password(data['password']):
            # Record failed login attempt
            record_failed_login(data['username'])
            logger.warning(f"Failed login attempt for username: {data['username']}")
            return jsonify({
                'success': False,
                'error': 'Invalid username or password'
            }), 401
        
        # Check if account is locked
        if hasattr(user, 'is_locked') and user.is_locked:
            return jsonify({
                'success': False,
                'error': 'Account is locked. Please contact support.'
            }), 401
        
        # Generate JWT tokens
        jwt_manager = get_jwt_manager()
        user_data = {
            'id': user.id,
            'username': user.username,
            'email': getattr(user, 'email', ''),
            'roles': getattr(user, 'roles', ['user']),
            'plan_type': getattr(user, 'plan_type', 'free')
        }
        
        access_token, refresh_token = jwt_manager.generate_tokens(user_data)
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Optionally maintain session for backward compatibility
        if data.get('remember_me'):
            login_user(user, remember=True)
        
        # Clear any cached user data
        invalidate_user_cache(user.id)
        
        logger.info(f"Successful login for user: {user.username}")
        
        return jsonify({
            'success': True,
            'data': {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'Bearer',
                'expires_in': current_app.config.get('JWT_ACCESS_TOKEN_EXPIRES', 3600),
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user_data['email'],
                    'plan_type': user_data['plan_type']
                }
            }
        }), 200
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid request data',
            'details': e.messages
        }), 400
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during login'
        }), 500


@auth_bp.route('/register', methods=['POST'])
@track_request_metrics
def register():
    """Register new user and return JWT tokens
    
    ---
    tags:
      - Authentication
    summary: User registration
    description: Create new user account and receive JWT tokens
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/RegisterSchema'
    responses:
      201:
        description: User created successfully
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TokenResponseSchema'
      400:
        description: Invalid data or username already exists
    """
    try:
        # Validate request data
        schema = RegisterSchema()
        data = schema.load(request.get_json())
        
        # Import here to avoid circular imports
        from web.app import User, db
        
        # Check if username already exists
        if User.query.filter_by(username=data['username']).first():
            return jsonify({
                'success': False,
                'error': 'Username already exists'
            }), 400
        
        # Check if email already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({
                'success': False,
                'error': 'Email already registered'
            }), 400
        
        # Create new user
        user = User(
            username=data['username'],
            email=data['email']
        )
        user.set_password(data['password'])
        
        # Set additional attributes if model supports them
        if hasattr(user, 'plan_type'):
            user.plan_type = data['plan_type']
        if hasattr(user, 'created_at'):
            user.created_at = datetime.utcnow()
        
        db.session.add(user)
        db.session.commit()
        
        # Generate JWT tokens
        jwt_manager = get_jwt_manager()
        user_data = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'roles': ['user'],
            'plan_type': data['plan_type']
        }
        
        access_token, refresh_token = jwt_manager.generate_tokens(user_data)
        
        logger.info(f"New user registered: {user.username}")
        
        return jsonify({
            'success': True,
            'data': {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'token_type': 'Bearer',
                'expires_in': current_app.config.get('JWT_ACCESS_TOKEN_EXPIRES', 3600),
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'plan_type': data['plan_type']
                }
            }
        }), 201
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid request data',
            'details': e.messages
        }), 400
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during registration'
        }), 500


@auth_bp.route('/refresh', methods=['POST'])
@track_request_metrics
def refresh_token():
    """Refresh access token using refresh token
    
    ---
    tags:
      - Authentication
    summary: Refresh access token
    description: Get new access token using valid refresh token
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/RefreshTokenSchema'
    responses:
      200:
        description: Token refreshed successfully
        content:
          application/json:
            schema:
              type: object
              properties:
                access_token:
                  type: string
                token_type:
                  type: string
                expires_in:
                  type: integer
      401:
        description: Invalid or expired refresh token
    """
    try:
        # Validate request data
        schema = RefreshTokenSchema()
        data = schema.load(request.get_json())
        
        # Get JWT manager
        jwt_manager = get_jwt_manager()
        
        # Refresh the token
        new_access_token = jwt_manager.refresh_access_token(data['refresh_token'])
        
        if not new_access_token:
            return jsonify({
                'success': False,
                'error': 'Invalid or expired refresh token'
            }), 401
        
        return jsonify({
            'success': True,
            'data': {
                'access_token': new_access_token,
                'token_type': 'Bearer',
                'expires_in': current_app.config.get('JWT_ACCESS_TOKEN_EXPIRES', 3600)
            }
        }), 200
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid request data',
            'details': e.messages
        }), 400
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during token refresh'
        }), 500


@auth_bp.route('/logout', methods=['POST'])
@track_request_metrics
def logout():
    """Logout user and revoke tokens
    
    ---
    tags:
      - Authentication
    summary: User logout
    description: Logout user and revoke JWT tokens
    security:
      - BearerAuth: []
    responses:
      200:
        description: Successfully logged out
      401:
        description: Not authenticated
    """
    try:
        jwt_manager = get_jwt_manager()
        
        # Get token from request
        token = jwt_manager.extract_token_from_request(request)
        
        if token:
            # Revoke the token
            jwt_manager.revoke_token(token)
        
        # Also handle session logout for backward compatibility
        logout_user()
        
        return jsonify({
            'success': True,
            'message': 'Successfully logged out'
        }), 200
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during logout'
        }), 500


@auth_bp.route('/verify', methods=['GET'])
@track_request_metrics
def verify_token():
    """Verify if current token is valid
    
    ---
    tags:
      - Authentication
    summary: Verify token
    description: Check if the provided JWT token is valid
    security:
      - BearerAuth: []
    responses:
      200:
        description: Token is valid
        content:
          application/json:
            schema:
              type: object
              properties:
                valid:
                  type: boolean
                user:
                  type: object
      401:
        description: Token is invalid or missing
    """
    try:
        jwt_manager = get_jwt_manager()
        
        # Get token from request
        token = jwt_manager.extract_token_from_request(request)
        
        if not token:
            return jsonify({
                'success': False,
                'valid': False,
                'error': 'No token provided'
            }), 401
        
        # Validate token
        token_payload = jwt_manager.validate_token(token)
        
        if not token_payload:
            return jsonify({
                'success': False,
                'valid': False,
                'error': 'Invalid or expired token'
            }), 401
        
        return jsonify({
            'success': True,
            'valid': True,
            'user': {
                'id': token_payload.user_id,
                'username': token_payload.username,
                'email': token_payload.email,
                'plan_type': token_payload.plan_type,
                'roles': token_payload.roles
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during token verification'
        }), 500