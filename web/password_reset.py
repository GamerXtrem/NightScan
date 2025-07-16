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

from web.app import db, User, limiter, get_remote_address
from web.app_extensions import mail
from secure_auth import SecureAuth

logger = logging.getLogger(__name__)

# Create blueprint
password_reset_bp = Blueprint('password_reset', __name__)

# Token storage (in production, use Redis)
reset_tokens = {}  # email -> (token, timestamp)
token_usage = {}  # token -> usage_count


def generate_reset_token(email: str) -> str:
    """Generate secure password reset token."""
    # Clean old tokens periodically
    current_time = time.time()
    for stored_email, (token, timestamp) in list(reset_tokens.items()):
        if current_time - timestamp > 3600:  # 1 hour expiry
            del reset_tokens[stored_email]
            if token in token_usage:
                del token_usage[token]
    
    # Generate new token
    token = secrets.token_urlsafe(32)
    reset_tokens[email] = (token, current_time)
    token_usage[token] = 0
    
    # Also create signed token for URL
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    signed_token = serializer.dumps({'email': email, 'token': token}, salt='password-reset-salt')
    
    return signed_token


def verify_reset_token(signed_token: str, max_age: int = 3600) -> Optional[Dict]:
    """Verify password reset token."""
    serializer = URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    
    try:
        data = serializer.loads(signed_token, salt='password-reset-salt', max_age=max_age)
        email = data.get('email')
        token = data.get('token')
        
        # Verify token exists and hasn't been used
        if email in reset_tokens:
            stored_token, timestamp = reset_tokens[email]
            if stored_token == token and token_usage.get(token, 0) < 1:
                return {'email': email, 'token': token}
                
    except SignatureExpired:
        logger.warning(f"Expired reset token attempted")
    except BadTimeSignature:
        logger.warning(f"Invalid reset token attempted")
    except Exception as e:
        logger.error(f"Error verifying reset token: {e}")
        
    return None


def send_reset_email(user: User, token: str):
    """Send password reset email."""
    reset_url = url_for('password_reset.reset_password', token=token, _external=True)
    
    msg = Message(
        'NightScan - Password Reset Request',
        sender=current_app.config.get('MAIL_DEFAULT_SENDER', 'noreply@nightscan.app'),
        recipients=[user.email]
    )
    
    msg.body = f'''Hello {user.username},

You have requested to reset your password for your NightScan account.

To reset your password, click the following link:
{reset_url}

This link will expire in 1 hour.

If you did not request this password reset, please ignore this email and your password will remain unchanged.

For security reasons, this link can only be used once.

Best regards,
The NightScan Team
'''
    
    msg.html = f'''
<h2>Password Reset Request</h2>
<p>Hello {user.username},</p>
<p>You have requested to reset your password for your NightScan account.</p>
<p>To reset your password, click the button below:</p>
<p style="margin: 30px 0;">
    <a href="{reset_url}" style="background-color: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 4px;">
        Reset Password
    </a>
</p>
<p>Or copy and paste this link: {reset_url}</p>
<p><strong>This link will expire in 1 hour.</strong></p>
<p>If you did not request this password reset, please ignore this email and your password will remain unchanged.</p>
<p>For security reasons, this link can only be used once.</p>
<p>Best regards,<br>The NightScan Team</p>
'''
    
    try:
        mail.send(msg)
        logger.info(f"Password reset email sent to {user.email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}")
        return False


@password_reset_bp.route('/forgot-password', methods=['GET', 'POST'])
@limiter.limit("3 per 5 minutes")
def forgot_password():
    """Handle password reset request."""
    if request.method == 'POST':
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
@limiter.limit("5 per 10 minutes")
def reset_password(token):
    """Handle password reset with token."""
    # Verify token
    token_data = verify_reset_token(token)
    if not token_data:
        flash('Invalid or expired reset link', 'error')
        return redirect(url_for('password_reset.forgot_password'))
    
    email = token_data['email']
    user = User.query.filter_by(email=email).first()
    
    if not user:
        flash('Invalid reset link', 'error')
        return redirect(url_for('password_reset.forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validate passwords
        if not password or not confirm_password:
            flash('Please enter and confirm your new password', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        elif len(password) < 10:
            flash('Password must be at least 10 characters long', 'error')
        elif not any(c.isupper() for c in password):
            flash('Password must contain at least one uppercase letter', 'error')
        elif not any(c.islower() for c in password):
            flash('Password must contain at least one lowercase letter', 'error')
        elif not any(c.isdigit() for c in password):
            flash('Password must contain at least one number', 'error')
        elif not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
            flash('Password must contain at least one special character', 'error')
        else:
            # Update password
            secure_auth = SecureAuth(current_app.config['SECRET_KEY'])
            user.password_hash = secure_auth.hash_password(password)
            db.session.commit()
            
            # Mark token as used
            raw_token = token_data['token']
            token_usage[raw_token] = token_usage.get(raw_token, 0) + 1
            
            # Clear reset token
            if email in reset_tokens:
                del reset_tokens[email]
            
            logger.info(f"Password successfully reset for {email}")
            flash('Your password has been reset successfully. Please log in with your new password.', 'success')
            return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)


@password_reset_bp.route('/api/password-reset/request', methods=['POST'])
@limiter.limit("3 per 5 minutes")
def api_request_reset():
    """API endpoint for password reset request."""
    data = request.get_json()
    if not data or 'email' not in data:
        return {'error': 'Email required'}, 400
    
    email = data['email'].strip().lower()
    
    # Always return success to prevent email enumeration
    response = {
        'message': 'If an account exists with this email, reset instructions will be sent.',
        'status': 'success'
    }
    
    user = User.query.filter_by(email=email).first()
    if user:
        token = generate_reset_token(email)
        send_reset_email(user, token)
    
    return response, 200


@password_reset_bp.route('/api/password-reset/verify', methods=['POST'])
@limiter.limit("10 per 10 minutes")
def api_verify_token():
    """API endpoint to verify reset token."""
    data = request.get_json()
    if not data or 'token' not in data:
        return {'error': 'Token required'}, 400
    
    token_data = verify_reset_token(data['token'])
    if token_data:
        return {'valid': True, 'email': token_data['email']}, 200
    
    return {'valid': False}, 200


@password_reset_bp.route('/api/password-reset/reset', methods=['POST'])
@limiter.limit("5 per 10 minutes")
def api_reset_password():
    """API endpoint to reset password with token."""
    data = request.get_json()
    
    required_fields = ['token', 'password']
    if not data or not all(field in data for field in required_fields):
        return {'error': 'Token and password required'}, 400
    
    token_data = verify_reset_token(data['token'])
    if not token_data:
        return {'error': 'Invalid or expired token'}, 400
    
    email = token_data['email']
    user = User.query.filter_by(email=email).first()
    
    if not user:
        return {'error': 'Invalid token'}, 400
    
    password = data['password']
    
    # Validate password
    if len(password) < 10:
        return {'error': 'Password must be at least 10 characters'}, 400
    
    # Update password
    secure_auth = SecureAuth(current_app.config['SECRET_KEY'])
    user.password_hash = secure_auth.hash_password(password)
    db.session.commit()
    
    # Mark token as used
    raw_token = token_data['token']
    token_usage[raw_token] = token_usage.get(raw_token, 0) + 1
    
    # Clear reset token
    if email in reset_tokens:
        del reset_tokens[email]
    
    return {'message': 'Password reset successfully', 'status': 'success'}, 200