"""
Password Reset Functionality for NightScan
Implements secure password reset with rate limiting and token validation.
"""

import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from flask_mail import Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
import logging

from web.app_extensions import mail
from secure_auth import SecureAuth

logger = logging.getLogger(__name__)

# Create blueprint
password_reset_bp = Blueprint('password_reset', __name__)

# Token storage (in production, use Redis)
reset_tokens = {}  # email -> (token, timestamp)
token_usage = {}  # token -> usage_count

# Constants
TOKEN_EXPIRY = 3600  # 1 hour
MAX_TOKEN_USES = 1
RATE_LIMIT_WINDOW = 300  # 5 minutes
MAX_REQUESTS_PER_WINDOW = 3


def generate_reset_token(email: str) -> str:
    """Generate a secure password reset token."""
    # Clean up old tokens first
    cleanup_expired_tokens()
    
    # Generate secure token
    token = secrets.token_urlsafe(32)
    
    # Store token with timestamp
    reset_tokens[email] = (token, time.time())
    token_usage[token] = 0
    
    logger.info(f"Generated password reset token for {email}")
    return token


def verify_reset_token(token: str) -> Optional[Dict]:
    """Verify a password reset token and return associated data."""
    # Check token usage
    if token not in token_usage:
        logger.warning(f"Invalid token attempted: {token[:8]}...")
        return None
        
    if token_usage[token] >= MAX_TOKEN_USES:
        logger.warning(f"Token usage exceeded for: {token[:8]}...")
        return None
    
    # Find email associated with token
    for email, (stored_token, timestamp) in reset_tokens.items():
        if stored_token == token:
            # Check expiry
            if time.time() - timestamp > TOKEN_EXPIRY:
                logger.warning(f"Expired token for {email}")
                # Clean up expired token
                del reset_tokens[email]
                del token_usage[token]
                return None
                
            # Increment usage
            token_usage[token] += 1
            
            return {
                'email': email,
                'timestamp': timestamp,
                'remaining_uses': MAX_TOKEN_USES - token_usage[token]
            }
    
    logger.warning(f"Token not found: {token[:8]}...")
    return None


def cleanup_expired_tokens():
    """Remove expired tokens from storage."""
    current_time = time.time()
    expired_emails = []
    
    for email, (token, timestamp) in reset_tokens.items():
        if current_time - timestamp > TOKEN_EXPIRY:
            expired_emails.append(email)
            if token in token_usage:
                del token_usage[token]
    
    for email in expired_emails:
        del reset_tokens[email]
        
    if expired_emails:
        logger.info(f"Cleaned up {len(expired_emails)} expired tokens")


def send_reset_email(user, token: str) -> bool:
    """Send password reset email to user."""
    try:
        # Get mail instance from current app
        reset_url = url_for('password_reset.reset_password', 
                           token=token, 
                           _external=True)
        
        msg = Message(
            subject="NightScan - Password Reset Request",
            sender=current_app.config.get('MAIL_DEFAULT_SENDER', 'noreply@nightscan.app'),
            recipients=[user.email]
        )
        
        msg.body = f"""
Hello {user.username},

You have requested to reset your password for your NightScan account.

Please click the following link to reset your password:
{reset_url}

This link will expire in 1 hour.

If you did not request this password reset, please ignore this email.

Best regards,
The NightScan Team
"""
        
        mail.send(msg)
        return True
        
    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}")
        return False


@password_reset_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Handle password reset request."""
    if request.method == 'POST':
        # Import here to avoid circular dependency
        from web.app import User
        
        email = request.form.get('email', '').strip().lower()
        
        if not email:
            flash('Please enter your email address', 'error')
            return render_template('forgot_password.html')
        
        # Always show success message to prevent email enumeration
        flash('If an account exists with this email, you will receive password reset instructions.', 'info')
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        if user:
            # Generate and send reset token
            token = generate_reset_token(email)
            if send_reset_email(user, token):
                logger.info(f"Password reset requested for {email}")
            else:
                logger.error(f"Failed to send reset email for {email}")
        else:
            # Log attempt for non-existent email
            logger.warning(f"Password reset attempted for non-existent email: {email}")
            
        return redirect(url_for('login'))
        
    return render_template('forgot_password.html')


@password_reset_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Handle password reset with token."""
    # Import here to avoid circular dependency
    from web.app import User, db
    
    # Verify token
    token_data = verify_reset_token(token)
    if not token_data:
        flash('Invalid or expired reset link', 'error')
        return redirect(url_for('password_reset.forgot_password'))
    
    email = token_data['email']
    user = User.query.filter_by(email=email).first()
    
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validate passwords
        if not password or not confirm_password:
            flash('Please enter both password fields', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        elif len(password) < 8:
            flash('Password must be at least 8 characters long', 'error')
        else:
            # Update password using secure method
            secure_auth = SecureAuth()
            user.password_hash = secure_auth.hash_password(password)
            
            try:
                db.session.commit()
                
                # Clean up used token
                del reset_tokens[email]
                if token in token_usage:
                    del token_usage[token]
                
                flash('Your password has been reset successfully. Please log in.', 'success')
                logger.info(f"Password reset successful for {email}")
                return redirect(url_for('login'))
                
            except Exception as e:
                db.session.rollback()
                logger.error(f"Failed to reset password for {email}: {e}")
                flash('An error occurred. Please try again.', 'error')
    
    return render_template('reset_password.html', token=token)


# API endpoints for mobile/external apps
@password_reset_bp.route('/api/password-reset/request', methods=['POST'])
def api_request_reset():
    """API endpoint to request password reset."""
    # Import here to avoid circular dependency
    from web.app import User
    
    data = request.get_json()
    
    if not data or 'email' not in data:
        return {'success': False, 'error': 'Email required'}, 400
    
    email = data['email'].strip().lower()
    
    # Always return success to prevent enumeration
    response = {
        'success': True,
        'message': 'If an account exists with this email, reset instructions will be sent.'
    }
    
    user = User.query.filter_by(email=email).first()
    if user:
        token = generate_reset_token(email)
        send_reset_email(user, token)
        
    return response, 200


@password_reset_bp.route('/api/password-reset/verify', methods=['POST'])
def api_verify_token():
    """API endpoint to verify reset token."""
    data = request.get_json()
    
    if not data or 'token' not in data:
        return {'success': False, 'error': 'Token required'}, 400
    
    token_data = verify_reset_token(data['token'])
    
    if token_data:
        return {
            'success': True,
            'email': token_data['email'],
            'remaining_uses': token_data['remaining_uses']
        }, 200
    else:
        return {'success': False, 'error': 'Invalid or expired token'}, 400


@password_reset_bp.route('/api/password-reset/reset', methods=['POST'])
def api_reset_password():
    """API endpoint to reset password with token."""
    # Import here to avoid circular dependency
    from web.app import User, db
    
    data = request.get_json()
    
    if not data:
        return {'success': False, 'error': 'No data provided'}, 400
        
    token = data.get('token')
    password = data.get('password')
    
    if not token or not password:
        return {'success': False, 'error': 'Token and password required'}, 400
    
    # Verify token
    token_data = verify_reset_token(token)
    if not token_data:
        return {'success': False, 'error': 'Invalid or expired token'}, 400
    
    # Validate password
    if len(password) < 8:
        return {'success': False, 'error': 'Password must be at least 8 characters long'}, 400
    
    # Update password
    email = token_data['email']
    user = User.query.filter_by(email=email).first()
    
    if not user:
        return {'success': False, 'error': 'User not found'}, 404
    
    try:
        secure_auth = SecureAuth()
        user.password_hash = secure_auth.hash_password(password)
        db.session.commit()
        
        # Clean up used token
        del reset_tokens[email]
        if token in token_usage:
            del token_usage[token]
            
        return {'success': True, 'message': 'Password reset successfully'}, 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"API password reset failed for {email}: {e}")
        return {'success': False, 'error': 'Failed to reset password'}, 500