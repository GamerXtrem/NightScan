"""
Session Management Module

Handles secure session management and storage.
"""

import secrets
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import hmac
from flask import Flask, session, request, g
from werkzeug.datastructures import CallbackDict
import redis
import pickle

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages secure sessions."""
    
    def __init__(self, config):
        self.config = config
        
        # Session configuration
        self.session_lifetime = getattr(config.security, 'session_lifetime', 3600)  # 1 hour
        self.session_cookie_secure = config.security.force_https
        self.session_cookie_httponly = True
        self.session_cookie_samesite = 'Lax'
        self.session_cookie_name = 'nightscan_session'
        
        # Session storage backend
        self.storage_backend = getattr(config.security, 'session_backend', 'filesystem')
        self.sessions = {}  # In-memory fallback
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize session storage backend."""
        if self.storage_backend == 'redis':
            try:
                self.redis_client = redis.Redis(
                    host=getattr(self.config, 'redis_host', 'localhost'),
                    port=getattr(self.config, 'redis_port', 6379),
                    db=getattr(self.config, 'redis_db', 0),
                    decode_responses=False
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis session storage initialized")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.storage_backend = 'memory'
        
        elif self.storage_backend == 'filesystem':
            self.session_dir = Path(self.config.paths.temp) / 'sessions'
            self.session_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Filesystem session storage initialized")
        
        else:
            logger.info("Memory session storage initialized")
    
    def init_app(self, app: Flask) -> None:
        """Initialize session management with Flask app."""
        # Configure Flask session
        app.config.update(
            SECRET_KEY=self.config.security.secret_key,
            SESSION_COOKIE_NAME=self.session_cookie_name,
            SESSION_COOKIE_SECURE=self.session_cookie_secure,
            SESSION_COOKIE_HTTPONLY=self.session_cookie_httponly,
            SESSION_COOKIE_SAMESITE=self.session_cookie_samesite,
            PERMANENT_SESSION_LIFETIME=timedelta(seconds=self.session_lifetime)
        )
        
        # Override session interface if needed
        if self.storage_backend != 'default':
            app.session_interface = self.SecureSessionInterface(self)
        
        # Add session lifecycle handlers
        @app.before_request
        def before_request():
            # Check session validity
            if 'session_id' in session:
                session_data = self.get_session(session['session_id'])
                if session_data:
                    # Check expiration
                    if self._is_session_expired(session_data):
                        self.destroy_session(session['session_id'])
                        session.clear()
                    else:
                        # Update last activity
                        session_data['last_activity'] = datetime.utcnow().isoformat()
                        self.update_session(session['session_id'], session_data)
                        g.session_data = session_data
                else:
                    # Invalid session
                    session.clear()
        
        logger.info("Session manager initialized")
    
    def create_session(self, user_id: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session."""
        # Generate secure session ID
        session_id = self._generate_session_id()
        
        # Create session data
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': datetime.utcnow().isoformat(),
            'last_activity': datetime.utcnow().isoformat(),
            'ip_address': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'data': data or {}
        }
        
        # Store session
        self._store_session(session_id, session_data)
        
        # Set in Flask session
        session['session_id'] = session_id
        session['user_id'] = user_id
        session.permanent = True
        
        logger.info(f"Session created for user {user_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return self._load_session(session_id)
    
    def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        existing = self.get_session(session_id)
        if not existing:
            return False
        
        # Merge data
        existing.update(data)
        existing['last_activity'] = datetime.utcnow().isoformat()
        
        # Store updated session
        self._store_session(session_id, existing)
        return True
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session."""
        return self._delete_session(session_id)
    
    def destroy_all_user_sessions(self, user_id: str) -> int:
        """Destroy all sessions for a user."""
        count = 0
        
        # This is inefficient for large scale, but works for now
        all_sessions = self._get_all_sessions()
        for sid, sdata in all_sessions.items():
            if sdata.get('user_id') == user_id:
                if self.destroy_session(sid):
                    count += 1
        
        logger.info(f"Destroyed {count} sessions for user {user_id}")
        return count
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        count = 0
        all_sessions = self._get_all_sessions()
        
        for sid, sdata in all_sessions.items():
            if self._is_session_expired(sdata):
                if self.destroy_session(sid):
                    count += 1
        
        logger.info(f"Cleaned up {count} expired sessions")
        return count
    
    def _generate_session_id(self) -> str:
        """Generate a secure session ID."""
        # Generate random bytes
        random_bytes = secrets.token_bytes(32)
        
        # Add timestamp and request info for uniqueness
        timestamp = str(datetime.utcnow().timestamp()).encode()
        ip = request.remote_addr.encode() if request.remote_addr else b'unknown'
        
        # Create hash
        data = random_bytes + timestamp + ip
        session_id = hashlib.sha256(data).hexdigest()
        
        return session_id
    
    def _is_session_expired(self, session_data: Dict[str, Any]) -> bool:
        """Check if session is expired."""
        try:
            last_activity = datetime.fromisoformat(session_data['last_activity'])
            expiry_time = last_activity + timedelta(seconds=self.session_lifetime)
            return datetime.utcnow() > expiry_time
        except:
            return True
    
    # Storage backend methods
    
    def _store_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Store session data."""
        try:
            if self.storage_backend == 'redis':
                # Serialize and store in Redis
                serialized = pickle.dumps(data)
                self.redis_client.setex(
                    f"session:{session_id}",
                    self.session_lifetime,
                    serialized
                )
            
            elif self.storage_backend == 'filesystem':
                # Store as JSON file
                session_file = self.session_dir / f"{session_id}.json"
                with open(session_file, 'w') as f:
                    json.dump(data, f)
                # Set file permissions
                session_file.chmod(0o600)
            
            else:
                # Memory storage
                self.sessions[session_id] = data
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store session: {e}")
            return False
    
    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data."""
        try:
            if self.storage_backend == 'redis':
                # Load from Redis
                data = self.redis_client.get(f"session:{session_id}")
                if data:
                    return pickle.loads(data)
            
            elif self.storage_backend == 'filesystem':
                # Load from file
                session_file = self.session_dir / f"{session_id}.json"
                if session_file.exists():
                    with open(session_file, 'r') as f:
                        return json.load(f)
            
            else:
                # Memory storage
                return self.sessions.get(session_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
    
    def _delete_session(self, session_id: str) -> bool:
        """Delete session data."""
        try:
            if self.storage_backend == 'redis':
                # Delete from Redis
                self.redis_client.delete(f"session:{session_id}")
            
            elif self.storage_backend == 'filesystem':
                # Delete file
                session_file = self.session_dir / f"{session_id}.json"
                if session_file.exists():
                    session_file.unlink()
            
            else:
                # Memory storage
                if session_id in self.sessions:
                    del self.sessions[session_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    def _get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all sessions (for cleanup)."""
        sessions = {}
        
        try:
            if self.storage_backend == 'redis':
                # Get all session keys
                for key in self.redis_client.scan_iter("session:*"):
                    session_id = key.decode().split(':')[1]
                    session_data = self._load_session(session_id)
                    if session_data:
                        sessions[session_id] = session_data
            
            elif self.storage_backend == 'filesystem':
                # Read all session files
                for session_file in self.session_dir.glob("*.json"):
                    session_id = session_file.stem
                    session_data = self._load_session(session_id)
                    if session_data:
                        sessions[session_id] = session_data
            
            else:
                # Memory storage
                sessions = self.sessions.copy()
        
        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
        
        return sessions
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions."""
        if self.storage_backend == 'redis':
            return len(list(self.redis_client.scan_iter("session:*")))
        elif self.storage_backend == 'filesystem':
            return len(list(self.session_dir.glob("*.json")))
        else:
            return len(self.sessions)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information (sanitized)."""
        session_data = self.get_session(session_id)
        if not session_data:
            return None
        
        # Return sanitized info
        return {
            'session_id': session_id,
            'user_id': session_data.get('user_id'),
            'created_at': session_data.get('created_at'),
            'last_activity': session_data.get('last_activity'),
            'ip_address': session_data.get('ip_address'),
            'is_expired': self._is_session_expired(session_data)
        }
    
    class SecureSessionInterface:
        """Custom session interface for Flask."""
        
        def __init__(self, session_manager):
            self.session_manager = session_manager
        
        def open_session(self, app, request):
            # Implementation would go here for custom session handling
            # For now, use default Flask session
            return None
        
        def save_session(self, app, session, response):
            # Implementation would go here for custom session handling
            # For now, use default Flask session
            pass