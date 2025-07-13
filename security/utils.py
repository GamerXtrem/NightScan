"""
Security Utilities Module

Provides utility functions for security operations.
"""

import os
import secrets
import hashlib
import hmac
import base64
import logging
import ipaddress
import socket
import re
from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import struct
from urllib.parse import urlparse, parse_qs
import dns.resolver
import whois

logger = logging.getLogger(__name__)


class SecurityUtils:
    """Security utility functions."""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize DNS resolver
        self.dns_resolver = dns.resolver.Resolver()
        self.dns_resolver.timeout = 2
        self.dns_resolver.lifetime = 2
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def generate_api_key(self, prefix: str = 'nsk') -> str:
        """Generate secure API key with prefix."""
        # Generate random component
        random_part = secrets.token_urlsafe(32)
        
        # Create checksum
        checksum = hashlib.sha256(random_part.encode()).hexdigest()[:6]
        
        # Combine with prefix
        return f"{prefix}_{random_part}_{checksum}"
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key format and checksum."""
        try:
            parts = api_key.split('_')
            if len(parts) != 3:
                return False
            
            prefix, random_part, checksum = parts
            
            # Verify checksum
            expected_checksum = hashlib.sha256(random_part.encode()).hexdigest()[:6]
            return hmac.compare_digest(checksum, expected_checksum)
            
        except Exception:
            return False
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """
        Hash password using PBKDF2.
        
        Returns:
            Tuple of (hash, salt) both as base64 strings
        """
        if salt is None:
            salt = os.urandom(32)
        
        # Use PBKDF2 with SHA256
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # iterations
        )
        
        # Return base64 encoded values
        return base64.b64encode(key).decode(), base64.b64encode(salt).decode()
    
    def verify_password_hash(self, password: str, hash_b64: str, salt_b64: str) -> bool:
        """Verify password against hash."""
        try:
            # Decode from base64
            salt = base64.b64decode(salt_b64)
            
            # Hash the password with the same salt
            new_hash, _ = self.hash_password(password, salt)
            
            # Compare hashes
            return hmac.compare_digest(new_hash, hash_b64)
            
        except Exception:
            return False
    
    def get_client_ip(self) -> str:
        """Get client IP address from request."""
        try:
            from flask import request
            
            # Check for proxy headers if configured
            if self.config.security.trusted_proxies:
                # X-Forwarded-For
                forwarded_for = request.headers.get('X-Forwarded-For')
                if forwarded_for:
                    # Get first IP in chain
                    ip = forwarded_for.split(',')[0].strip()
                    if self.is_valid_ip(ip):
                        return ip
                
                # X-Real-IP
                real_ip = request.headers.get('X-Real-IP')
                if real_ip and self.is_valid_ip(real_ip):
                    return real_ip
            
            # Default to remote_addr
            return request.remote_addr or 'unknown'
            
        except Exception:
            return 'unknown'
    
    def is_valid_ip(self, ip_str: str) -> bool:
        """Check if string is valid IP address."""
        try:
            ipaddress.ip_address(ip_str)
            return True
        except ValueError:
            return False
    
    def is_private_ip(self, ip_str: str) -> bool:
        """Check if IP is private/internal."""
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            return False
    
    def get_ip_info(self, ip_str: str) -> Dict[str, Any]:
        """Get information about IP address."""
        info = {
            'ip': ip_str,
            'valid': False,
            'version': None,
            'is_private': False,
            'is_loopback': False,
            'is_multicast': False,
            'reverse_dns': None,
            'asn': None
        }
        
        try:
            ip = ipaddress.ip_address(ip_str)
            info.update({
                'valid': True,
                'version': ip.version,
                'is_private': ip.is_private,
                'is_loopback': ip.is_loopback,
                'is_multicast': ip.is_multicast
            })
            
            # Try reverse DNS
            try:
                info['reverse_dns'] = socket.gethostbyaddr(ip_str)[0]
            except:
                pass
            
        except ValueError:
            pass
        
        return info
    
    def check_ip_reputation(self, ip_str: str) -> Dict[str, Any]:
        """
        Check IP reputation against blacklists.
        
        Note: This is a basic implementation. In production,
        use specialized services like IPQualityScore, AbuseIPDB, etc.
        """
        reputation = {
            'ip': ip_str,
            'is_blacklisted': False,
            'blacklists': [],
            'risk_score': 0
        }
        
        # Basic checks
        if not self.is_valid_ip(ip_str):
            reputation['risk_score'] = 100
            return reputation
        
        # Check if private IP (generally safe)
        if self.is_private_ip(ip_str):
            reputation['risk_score'] = 0
            return reputation
        
        # Check common spam blacklists (RBL)
        # Note: This is simplified - real implementation would check multiple RBLs
        rbl_domains = [
            'zen.spamhaus.org',
            'b.barracudacentral.org',
            'cbl.abuseat.org'
        ]
        
        # Reverse IP for DNS lookup
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.version == 4:
                reversed_ip = '.'.join(reversed(ip_str.split('.')))
                
                for rbl in rbl_domains:
                    try:
                        query = f"{reversed_ip}.{rbl}"
                        self.dns_resolver.query(query, 'A')
                        # If query succeeds, IP is listed
                        reputation['is_blacklisted'] = True
                        reputation['blacklists'].append(rbl)
                        reputation['risk_score'] += 30
                    except:
                        # Not listed in this RBL
                        pass
        except:
            pass
        
        return reputation
    
    def sanitize_filename(self, filename: str, max_length: int = 255) -> str:
        """Sanitize filename for safe storage."""
        # Remove path components
        filename = os.path.basename(filename)
        
        # Replace dangerous characters
        filename = re.sub(r'[^\w\s.-]', '_', filename)
        
        # Remove multiple dots (except for extension)
        parts = filename.rsplit('.', 1)
        if len(parts) == 2:
            name, ext = parts
            name = re.sub(r'\.+', '_', name)
            filename = f"{name}.{ext}"
        else:
            filename = re.sub(r'\.+', '_', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > max_length:
            name, ext = os.path.splitext(filename)
            max_name_length = max_length - len(ext) - 1
            filename = name[:max_name_length] + ext
        
        # Ensure filename is not empty
        if not filename:
            filename = f"file_{uuid.uuid4().hex[:8]}"
        
        return filename
    
    def validate_url(self, url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
        """Validate URL for safety."""
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in allowed_schemes:
                return False
            
            # Check for basic structure
            if not parsed.netloc:
                return False
            
            # Check for dangerous patterns
            dangerous_patterns = [
                'javascript:', 'data:', 'vbscript:', 'file:',
                '..', '//', '\\\\', '%00', '\x00'
            ]
            
            url_lower = url.lower()
            for pattern in dangerous_patterns:
                if pattern in url_lower:
                    return False
            
            # Check domain
            if not self._is_valid_domain(parsed.netloc):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Check if domain is valid."""
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Check format
        if not re.match(r'^[a-zA-Z0-9.-]+$', domain):
            return False
        
        # Check TLD
        parts = domain.split('.')
        if len(parts) < 2:
            return False
        
        # Additional checks could include:
        # - DNS resolution
        # - Domain reputation
        # - Age verification
        
        return True
    
    def generate_csrf_token(self) -> str:
        """Generate CSRF token."""
        timestamp = struct.pack('>Q', int(datetime.utcnow().timestamp()))
        random_bytes = os.urandom(24)
        
        token_data = timestamp + random_bytes
        
        # Sign with secret key
        signature = hmac.new(
            self.config.security.secret_key.encode(),
            token_data,
            hashlib.sha256
        ).digest()
        
        # Combine and encode
        token = base64.urlsafe_b64encode(token_data + signature).decode()
        
        return token
    
    def verify_csrf_token(self, token: str, max_age_seconds: int = 3600) -> bool:
        """Verify CSRF token."""
        try:
            # Decode token
            decoded = base64.urlsafe_b64decode(token.encode())
            
            # Extract parts
            timestamp = decoded[:8]
            random_bytes = decoded[8:32]
            signature = decoded[32:]
            
            # Verify signature
            token_data = timestamp + random_bytes
            expected_signature = hmac.new(
                self.config.security.secret_key.encode(),
                token_data,
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return False
            
            # Check timestamp
            token_timestamp = struct.unpack('>Q', timestamp)[0]
            current_timestamp = int(datetime.utcnow().timestamp())
            
            if current_timestamp - token_timestamp > max_age_seconds:
                return False
            
            return True
            
        except Exception:
            return False
    
    def encode_jwt_payload(self, payload: Dict[str, Any], expiry_hours: int = 24) -> str:
        """Encode JWT payload (simplified version)."""
        # Add standard claims
        payload['iat'] = int(datetime.utcnow().timestamp())
        payload['exp'] = int((datetime.utcnow() + timedelta(hours=expiry_hours)).timestamp())
        payload['jti'] = str(uuid.uuid4())
        
        # Encode payload
        header = {'alg': 'HS256', 'typ': 'JWT'}
        
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).decode().rstrip('=')
        
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip('=')
        
        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.config.security.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        return f"{message}.{signature_b64}"
    
    def check_domain_age(self, domain: str) -> Optional[int]:
        """
        Check domain age in days.
        
        Returns:
            Age in days or None if unable to determine
        """
        try:
            w = whois.whois(domain)
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation_date = w.creation_date[0]
                else:
                    creation_date = w.creation_date
                
                age = (datetime.now() - creation_date).days
                return age
        except:
            pass
        
        return None
    
    def calculate_password_strength(self, password: str) -> Dict[str, Any]:
        """Calculate password strength score."""
        score = 0
        feedback = []
        
        # Length
        length = len(password)
        if length >= 12:
            score += 20
        elif length >= 8:
            score += 10
        else:
            feedback.append("Password too short")
        
        # Character variety
        has_lower = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digit = bool(re.search(r'[0-9]', password))
        has_special = bool(re.search(r'[^a-zA-Z0-9]', password))
        
        variety = sum([has_lower, has_upper, has_digit, has_special])
        score += variety * 20
        
        if not has_lower:
            feedback.append("Add lowercase letters")
        if not has_upper:
            feedback.append("Add uppercase letters")
        if not has_digit:
            feedback.append("Add numbers")
        if not has_special:
            feedback.append("Add special characters")
        
        # Pattern checks
        if re.search(r'(.)\1{2,}', password):
            score -= 10
            feedback.append("Avoid repeated characters")
        
        if re.search(r'(012|123|234|345|456|567|678|789|890|abc|bcd|cde|def)', password.lower()):
            score -= 10
            feedback.append("Avoid sequential characters")
        
        # Common patterns
        common_patterns = ['password', 'qwerty', 'admin', 'letmein', '123456']
        for pattern in common_patterns:
            if pattern in password.lower():
                score -= 20
                feedback.append(f"Avoid common words like '{pattern}'")
                break
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        # Determine strength level
        if score >= 80:
            strength = 'strong'
        elif score >= 60:
            strength = 'moderate'
        elif score >= 40:
            strength = 'weak'
        else:
            strength = 'very_weak'
        
        return {
            'score': score,
            'strength': strength,
            'feedback': feedback
        }
    
    def generate_otp(self, length: int = 6) -> str:
        """Generate one-time password."""
        return ''.join(secrets.choice('0123456789') for _ in range(length))