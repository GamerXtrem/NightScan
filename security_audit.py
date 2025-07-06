#!/usr/bin/env python3
"""
NightScan Security Audit Tool
Comprehensive security analysis for vulnerabilities and compliance.
"""

import os
import re
import json
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class SecurityAuditor:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path)
        self.vulnerabilities = defaultdict(list)
        self.security_score = 100
        
    def run_full_audit(self):
        """Run comprehensive security audit."""
        print("🔒 NightScan Security Audit - Comprehensive Analysis")
        print("=" * 60)
        
        self.check_hardcoded_secrets()
        self.check_sql_injection_vulnerabilities()
        self.check_xss_vulnerabilities()
        self.check_csrf_protection()
        self.check_authentication_security()
        self.check_authorization_flaws()
        self.check_input_validation()
        self.check_file_upload_security()
        self.check_dependency_vulnerabilities()
        self.check_docker_security()
        self.check_kubernetes_security()
        self.check_logging_security()
        self.check_encryption_standards()
        self.check_session_management()
        self.check_rate_limiting()
        self.check_cors_configuration()
        self.check_security_headers()
        self.check_ssl_tls_configuration()
        self.check_environment_exposure()
        self.check_privilege_escalation()
        
        self.generate_security_report()
        
    def check_hardcoded_secrets(self):
        """Check for hardcoded secrets and credentials."""
        print("  🔑 Checking for hardcoded secrets...")
        
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded Password'),
            (r'secret_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded Secret Key'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API Key'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded Token'),
            (r'["\'][A-Za-z0-9+/]{32,}={0,2}["\']', 'Potential Base64 Secret'),
            (r'sk_[a-zA-Z0-9]{20,}', 'Stripe Secret Key'),
            (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
            (r'AIza[0-9A-Za-z\\-_]{35}', 'Google API Key'),
            (r'postgres://[^:]+:[^@]+@', 'Database URL with credentials'),
            (r'mysql://[^:]+:[^@]+@', 'MySQL URL with credentials'),
        ]
        
        for file_path in self.root_path.rglob("*"):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern, description in secret_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Skip obvious test/example values
                            if any(test_val in str(match).lower() for test_val in 
                                   ['test', 'example', 'dummy', 'placeholder', 'your-key-here']):
                                continue
                                
                            self.vulnerabilities['hardcoded_secrets'].append({
                                'type': 'CRITICAL',
                                'description': description,
                                'file': str(file_path),
                                'match': str(match)[:50] + "...",
                                'line': self._find_line_number(content, str(match))
                            })
                            self.security_score -= 15
                            
                except Exception:
                    continue
                    
    def check_sql_injection_vulnerabilities(self):
        """Check for SQL injection vulnerabilities."""
        print("  💉 Checking for SQL injection vulnerabilities...")
        
        dangerous_patterns = [
            (r'execute\(["\'][^"\']*%s[^"\']*["\']', 'String formatting in SQL'),
            (r'execute\(["\'][^"\']*\+[^"\']*["\']', 'String concatenation in SQL'),
            (r'query\(["\'][^"\']*%s[^"\']*["\']', 'String formatting in query'),
            (r'\.format\([^)]*\)[^;]*WHERE', 'Format string in WHERE clause'),
            (r'f["\'][^"\']*SELECT[^"\']*{[^}]*}', 'F-string in SELECT'),
        ]
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in dangerous_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        self.vulnerabilities['sql_injection'].append({
                            'type': 'HIGH',
                            'description': description,
                            'file': str(file_path),
                            'pattern': match[:100] + "...",
                            'line': self._find_line_number(content, match)
                        })
                        self.security_score -= 10
                        
            except Exception:
                continue
                
    def check_xss_vulnerabilities(self):
        """Check for Cross-Site Scripting vulnerabilities."""
        print("  🕸️  Checking for XSS vulnerabilities...")
        
        xss_patterns = [
            (r'innerHTML\s*=\s*[^;]*request\.', 'Direct HTML injection from request'),
            (r'\.html\([^)]*request\.', 'HTML rendering with user input'),
            (r'render_template_string\([^)]*request\.', 'Template string with user input'),
            (r'\|safe\s*}}', 'Jinja2 safe filter usage'),
            (r'Markup\([^)]*request\.', 'Direct markup creation from request'),
        ]
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in xss_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        self.vulnerabilities['xss'].append({
                            'type': 'HIGH',
                            'description': description,
                            'file': str(file_path),
                            'pattern': match,
                            'line': self._find_line_number(content, match)
                        })
                        self.security_score -= 8
                        
            except Exception:
                continue
                
    def check_csrf_protection(self):
        """Check CSRF protection implementation."""
        print("  🛡️  Checking CSRF protection...")
        
        # Check if CSRF protection is enabled
        csrf_enabled = False
        csrf_exclusions = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'CSRFProtect' in content or 'csrf' in content.lower():
                    csrf_enabled = True
                    
                # Check for CSRF exemptions
                if '@csrf.exempt' in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if '@csrf.exempt' in line:
                            csrf_exclusions.append({
                                'file': str(file_path),
                                'line': i + 1,
                                'context': lines[min(i+1, len(lines)-1)] if i+1 < len(lines) else ''
                            })
                            
            except Exception:
                continue
                
        if not csrf_enabled:
            self.vulnerabilities['csrf'].append({
                'type': 'HIGH',
                'description': 'CSRF protection not implemented',
                'recommendation': 'Implement Flask-WTF CSRFProtect'
            })
            self.security_score -= 12
            
        for exclusion in csrf_exclusions:
            self.vulnerabilities['csrf'].append({
                'type': 'MEDIUM',
                'description': 'CSRF protection exemption found',
                'file': exclusion['file'],
                'line': exclusion['line'],
                'context': exclusion['context']
            })
            self.security_score -= 5
            
    def check_authentication_security(self):
        """Check authentication implementation security."""
        print("  🔐 Checking authentication security...")
        
        auth_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for weak password hashing
                if 'md5(' in content.lower() or 'sha1(' in content.lower():
                    auth_issues.append({
                        'type': 'HIGH',
                        'description': 'Weak password hashing algorithm',
                        'file': str(file_path),
                        'recommendation': 'Use bcrypt, scrypt, or Argon2'
                    })
                    self.security_score -= 10
                    
                # Check for session fixation vulnerabilities
                if 'session[' in content and 'login' in content.lower():
                    if 'session.regenerate' not in content and 'new_session' not in content:
                        auth_issues.append({
                            'type': 'MEDIUM',
                            'description': 'Potential session fixation vulnerability',
                            'file': str(file_path),
                            'recommendation': 'Regenerate session ID after login'
                        })
                        self.security_score -= 6
                        
                # Check for hardcoded authentication bypasses
                if re.search(r'if.*==.*["\']admin["\'].*or.*True', content, re.IGNORECASE):
                    auth_issues.append({
                        'type': 'CRITICAL',
                        'description': 'Authentication bypass detected',
                        'file': str(file_path)
                    })
                    self.security_score -= 20
                    
            except Exception:
                continue
                
        self.vulnerabilities['authentication'] = auth_issues
        
    def check_authorization_flaws(self):
        """Check for authorization and access control flaws."""
        print("  👤 Checking authorization controls...")
        
        authz_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for missing authorization checks
                route_pattern = r'@app\.route\([^)]+\)'
                routes = re.findall(route_pattern, content)
                
                for route in routes:
                    route_start = content.find(route)
                    # Look for the function definition after the route
                    func_match = re.search(r'def\s+(\w+)', content[route_start:route_start+500])
                    if func_match:
                        func_name = func_match.group(1)
                        func_content = content[route_start:route_start+1000]
                        
                        # Check if route has authorization
                        if ('@login_required' not in func_content and 
                            'current_user' not in func_content and
                            'authenticate' not in func_content.lower()):
                            authz_issues.append({
                                'type': 'MEDIUM',
                                'description': 'Route without authorization check',
                                'file': str(file_path),
                                'route': route,
                                'function': func_name
                            })
                            self.security_score -= 4
                            
                # Check for privilege escalation risks
                if 'sudo' in content or 'setuid' in content:
                    authz_issues.append({
                        'type': 'HIGH',
                        'description': 'Potential privilege escalation',
                        'file': str(file_path)
                    })
                    self.security_score -= 8
                    
            except Exception:
                continue
                
        self.vulnerabilities['authorization'] = authz_issues
        
    def check_input_validation(self):
        """Check input validation and sanitization."""
        print("  ✅ Checking input validation...")
        
        validation_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for direct request parameter usage
                dangerous_uses = [
                    (r'request\.args\.get\([^)]+\)[^;]*execute', 'Direct request parameter in SQL'),
                    (r'request\.form\.get\([^)]+\)[^;]*execute', 'Direct form input in SQL'),
                    (r'request\.json\[[^]]+\][^;]*execute', 'Direct JSON input in SQL'),
                    (r'eval\(request\.', 'eval() with user input'),
                    (r'exec\(request\.', 'exec() with user input'),
                    (r'os\.system\([^)]*request\.', 'os.system() with user input'),
                    (r'subprocess\.[^(]*\([^)]*request\.', 'subprocess with user input'),
                ]
                
                for pattern, description in dangerous_uses:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        validation_issues.append({
                            'type': 'HIGH',
                            'description': description,
                            'file': str(file_path),
                            'pattern': match[:100] + "..."
                        })
                        self.security_score -= 10
                        
            except Exception:
                continue
                
        self.vulnerabilities['input_validation'] = validation_issues
        
    def check_file_upload_security(self):
        """Check file upload security."""
        print("  📁 Checking file upload security...")
        
        upload_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'request.files' in content:
                    # Check for file type validation
                    if 'filename' in content:
                        if not re.search(r'secure_filename|allowed_file|validate.*extension', content, re.IGNORECASE):
                            upload_issues.append({
                                'type': 'HIGH',
                                'description': 'File upload without proper validation',
                                'file': str(file_path),
                                'recommendation': 'Implement file type and size validation'
                            })
                            self.security_score -= 10
                            
                    # Check for path traversal protection
                    if 'save(' in content or 'write(' in content:
                        if 'secure_filename' not in content:
                            upload_issues.append({
                                'type': 'HIGH',
                                'description': 'File upload vulnerable to path traversal',
                                'file': str(file_path),
                                'recommendation': 'Use secure_filename() and validate paths'
                            })
                            self.security_score -= 12
                            
            except Exception:
                continue
                
        self.vulnerabilities['file_upload'] = upload_issues
        
    def check_dependency_vulnerabilities(self):
        """Check for vulnerable dependencies."""
        print("  📦 Checking dependency vulnerabilities...")
        
        # Known vulnerable versions (simplified check)
        vulnerable_packages = {
            'django': ['<3.2.13', '<4.0.4'],
            'flask': ['<2.0.3'],
            'requests': ['<2.27.1'],
            'pillow': ['<9.0.1'],
            'pyyaml': ['<6.0'],
            'jinja2': ['<2.11.3'],
            'werkzeug': ['<2.0.3'],
        }
        
        dependency_issues = []
        
        # Check requirements.txt
        req_file = self.root_path / 'requirements.txt'
        if req_file.exists():
            with open(req_file, 'r') as f:
                requirements = f.read()
                
            for package, vulnerable_versions in vulnerable_packages.items():
                pattern = rf'{package}==([0-9.]+)'
                match = re.search(pattern, requirements, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    # Simplified version check
                    for vuln_version in vulnerable_versions:
                        if '<' in vuln_version and version < vuln_version.replace('<', ''):
                            dependency_issues.append({
                                'type': 'HIGH',
                                'description': f'Vulnerable {package} version',
                                'version': version,
                                'recommendation': f'Update {package} to latest version'
                            })
                            self.security_score -= 8
                            
        self.vulnerabilities['dependencies'] = dependency_issues
        
    def check_docker_security(self):
        """Check Docker security configuration."""
        print("  🐳 Checking Docker security...")
        
        docker_issues = []
        
        dockerfile_path = self.root_path / 'Dockerfile'
        if dockerfile_path.exists():
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
                
            # Check for root user
            if 'USER root' in dockerfile_content or 'USER 0' in dockerfile_content:
                docker_issues.append({
                    'type': 'HIGH',
                    'description': 'Running as root user in Docker',
                    'recommendation': 'Use non-root user'
                })
                self.security_score -= 10
                
            # Check for security flags
            if '--privileged' in dockerfile_content:
                docker_issues.append({
                    'type': 'CRITICAL',
                    'description': 'Privileged Docker container',
                    'recommendation': 'Remove --privileged flag'
                })
                self.security_score -= 15
                
            # Check for secrets in build
            if any(secret in dockerfile_content.upper() for secret in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']):
                docker_issues.append({
                    'type': 'HIGH',
                    'description': 'Potential secrets in Dockerfile',
                    'recommendation': 'Use Docker secrets or build args'
                })
                self.security_score -= 8
                
        self.vulnerabilities['docker'] = docker_issues
        
    def check_kubernetes_security(self):
        """Check Kubernetes security configuration."""
        print("  ☸️  Checking Kubernetes security...")
        
        k8s_issues = []
        
        k8s_dir = self.root_path / 'k8s'
        if k8s_dir.exists():
            for yaml_file in k8s_dir.rglob("*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        yaml_content = f.read()
                        
                    # Check for privileged containers
                    if 'privileged: true' in yaml_content:
                        k8s_issues.append({
                            'type': 'CRITICAL',
                            'description': 'Privileged container in Kubernetes',
                            'file': str(yaml_file)
                        })
                        self.security_score -= 15
                        
                    # Check for missing security context
                    if 'containers:' in yaml_content and 'securityContext:' not in yaml_content:
                        k8s_issues.append({
                            'type': 'MEDIUM',
                            'description': 'Missing security context',
                            'file': str(yaml_file)
                        })
                        self.security_score -= 5
                        
                    # Check for root user
                    if 'runAsUser: 0' in yaml_content:
                        k8s_issues.append({
                            'type': 'HIGH',
                            'description': 'Running as root in Kubernetes',
                            'file': str(yaml_file)
                        })
                        self.security_score -= 10
                        
                except Exception:
                    continue
                    
        self.vulnerabilities['kubernetes'] = k8s_issues
        
    def check_logging_security(self):
        """Check logging security practices."""
        print("  📝 Checking logging security...")
        
        logging_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for logging sensitive data
                sensitive_patterns = [
                    (r'log.*password', 'Password in logs'),
                    (r'print.*password', 'Password in print statements'),
                    (r'log.*secret', 'Secret in logs'),
                    (r'log.*token', 'Token in logs'),
                    (r'log.*key', 'Key in logs'),
                ]
                
                for pattern, description in sensitive_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        logging_issues.append({
                            'type': 'MEDIUM',
                            'description': description,
                            'file': str(file_path)
                        })
                        self.security_score -= 4
                        
            except Exception:
                continue
                
        self.vulnerabilities['logging'] = logging_issues
        
    def check_encryption_standards(self):
        """Check encryption implementation."""
        print("  🔒 Checking encryption standards...")
        
        crypto_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for weak encryption
                weak_crypto = [
                    (r'DES\(', 'DES encryption (deprecated)'),
                    (r'MD5\(', 'MD5 hashing (weak)'),
                    (r'SHA1\(', 'SHA1 hashing (weak)'),
                    (r'RC4\(', 'RC4 encryption (broken)'),
                ]
                
                for pattern, description in weak_crypto:
                    if re.search(pattern, content, re.IGNORECASE):
                        crypto_issues.append({
                            'type': 'HIGH',
                            'description': description,
                            'file': str(file_path)
                        })
                        self.security_score -= 10
                        
            except Exception:
                continue
                
        self.vulnerabilities['encryption'] = crypto_issues
        
    def check_session_management(self):
        """Check session management security."""
        print("  🍪 Checking session management...")
        
        session_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for insecure session configuration
                if 'SESSION_COOKIE_SECURE' in content:
                    if 'SESSION_COOKIE_SECURE = False' in content:
                        session_issues.append({
                            'type': 'HIGH',
                            'description': 'Insecure session cookies',
                            'file': str(file_path)
                        })
                        self.security_score -= 8
                        
                # Check for missing HttpOnly
                if 'session' in content.lower() and 'httponly' not in content.lower():
                    session_issues.append({
                        'type': 'MEDIUM',
                        'description': 'Missing HttpOnly flag on cookies',
                        'file': str(file_path)
                    })
                    self.security_score -= 5
                    
            except Exception:
                continue
                
        self.vulnerabilities['session_management'] = session_issues
        
    def check_rate_limiting(self):
        """Check rate limiting implementation."""
        print("  🚦 Checking rate limiting...")
        
        rate_limit_found = False
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'limiter' in content.lower() or '@limit' in content:
                    rate_limit_found = True
                    break
                    
            except Exception:
                continue
                
        if not rate_limit_found:
            self.vulnerabilities['rate_limiting'].append({
                'type': 'MEDIUM',
                'description': 'No rate limiting implementation found',
                'recommendation': 'Implement Flask-Limiter'
            })
            self.security_score -= 6
            
    def check_cors_configuration(self):
        """Check CORS configuration."""
        print("  🌐 Checking CORS configuration...")
        
        cors_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'CORS' in content:
                    # Check for wildcard origins
                    if "origins='*'" in content or 'origins=["*"]' in content:
                        cors_issues.append({
                            'type': 'HIGH',
                            'description': 'CORS wildcard origin',
                            'file': str(file_path)
                        })
                        self.security_score -= 8
                        
            except Exception:
                continue
                
        self.vulnerabilities['cors'] = cors_issues
        
    def check_security_headers(self):
        """Check security headers implementation."""
        print("  📋 Checking security headers...")
        
        headers_found = {
            'csp': False,
            'hsts': False,
            'x_frame_options': False,
            'x_content_type_options': False
        }
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'Content-Security-Policy' in content:
                    headers_found['csp'] = True
                if 'Strict-Transport-Security' in content:
                    headers_found['hsts'] = True
                if 'X-Frame-Options' in content:
                    headers_found['x_frame_options'] = True
                if 'X-Content-Type-Options' in content:
                    headers_found['x_content_type_options'] = True
                    
            except Exception:
                continue
                
        for header, found in headers_found.items():
            if not found:
                self.vulnerabilities['security_headers'].append({
                    'type': 'MEDIUM',
                    'description': f'Missing {header.replace("_", " ").title()} header',
                    'recommendation': 'Implement security headers'
                })
                self.security_score -= 3
                
    def check_ssl_tls_configuration(self):
        """Check SSL/TLS configuration."""
        print("  🔐 Checking SSL/TLS configuration...")
        
        ssl_issues = []
        
        # Check for SSL enforcement
        ssl_enforced = False
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if 'force_https' in content or 'FORCE_HTTPS' in content:
                    ssl_enforced = True
                    break
                    
            except Exception:
                continue
                
        if not ssl_enforced:
            ssl_issues.append({
                'type': 'HIGH',
                'description': 'SSL/HTTPS not enforced',
                'recommendation': 'Implement HTTPS enforcement'
            })
            self.security_score -= 10
            
        self.vulnerabilities['ssl_tls'] = ssl_issues
        
    def check_environment_exposure(self):
        """Check for environment variable exposure."""
        print("  🌍 Checking environment exposure...")
        
        env_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for environment variable exposure in logs
                if 'os.environ' in content and ('print' in content or 'log' in content):
                    env_issues.append({
                        'type': 'MEDIUM',
                        'description': 'Potential environment variable exposure',
                        'file': str(file_path)
                    })
                    self.security_score -= 4
                    
            except Exception:
                continue
                
        self.vulnerabilities['environment'] = env_issues
        
    def check_privilege_escalation(self):
        """Check for privilege escalation vulnerabilities."""
        print("  ⬆️  Checking privilege escalation risks...")
        
        privesc_issues = []
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                dangerous_calls = [
                    (r'subprocess\.call\([^)]*shell=True', 'Shell injection risk'),
                    (r'os\.system\(', 'OS command execution'),
                    (r'eval\(', 'Code evaluation'),
                    (r'exec\(', 'Code execution'),
                ]
                
                for pattern, description in dangerous_calls:
                    if re.search(pattern, content, re.IGNORECASE):
                        privesc_issues.append({
                            'type': 'HIGH',
                            'description': description,
                            'file': str(file_path)
                        })
                        self.security_score -= 12
                        
            except Exception:
                continue
                
        self.vulnerabilities['privilege_escalation'] = privesc_issues
        
    def _find_line_number(self, content, search_string):
        """Find line number of a string in content."""
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if search_string in line:
                return i + 1
        return 0
        
    def generate_security_report(self):
        """Generate comprehensive security report."""
        print("\n" + "="*60)
        print("🔒 NIGHTSCAN SECURITY AUDIT REPORT")
        print("="*60)
        
        total_vulnerabilities = sum(len(vulns) for vulns in self.vulnerabilities.values())
        
        # Security score calculation
        final_score = max(0, self.security_score)
        
        if final_score >= 90:
            score_status = "🟢 EXCELLENT"
        elif final_score >= 75:
            score_status = "🟡 GOOD"
        elif final_score >= 60:
            score_status = "🟠 FAIR"
        else:
            score_status = "🔴 POOR"
            
        print(f"\n📊 SECURITY SCORE: {final_score}/100 ({score_status})")
        print(f"🚨 TOTAL VULNERABILITIES: {total_vulnerabilities}")
        
        # Count by severity
        critical_count = sum(len([v for v in vulns if v.get('type') == 'CRITICAL']) 
                           for vulns in self.vulnerabilities.values())
        high_count = sum(len([v for v in vulns if v.get('type') == 'HIGH']) 
                        for vulns in self.vulnerabilities.values())
        medium_count = sum(len([v for v in vulns if v.get('type') == 'MEDIUM']) 
                          for vulns in self.vulnerabilities.values())
        
        print(f"   🔴 CRITICAL: {critical_count}")
        print(f"   🟠 HIGH: {high_count}")
        print(f"   🟡 MEDIUM: {medium_count}")
        
        # Detailed vulnerabilities by category
        for category, vulns in self.vulnerabilities.items():
            if vulns:
                print(f"\n🔍 {category.upper().replace('_', ' ')} ({len(vulns)} issues)")
                print("-" * 40)
                
                for i, vuln in enumerate(vulns, 1):
                    severity_emoji = {
                        'CRITICAL': '🔴',
                        'HIGH': '🟠',
                        'MEDIUM': '🟡'
                    }.get(vuln.get('type', 'MEDIUM'), '🟡')
                    
                    print(f"{i}. {severity_emoji} {vuln['description']}")
                    if 'file' in vuln:
                        print(f"   📁 File: {vuln['file']}")
                    if 'line' in vuln:
                        print(f"   📍 Line: {vuln['line']}")
                    if 'recommendation' in vuln:
                        print(f"   💡 Fix: {vuln['recommendation']}")
                    print()
                    
        # Security recommendations
        print(f"\n🛡️  SECURITY RECOMMENDATIONS")
        print("-" * 30)
        
        if critical_count > 0:
            print("🔴 IMMEDIATE ACTIONS (Critical):")
            print("   1. Remove any hardcoded secrets immediately")
            print("   2. Fix authentication bypasses")
            print("   3. Remove privileged container configurations")
            print("   4. Address SQL injection vulnerabilities")
            print()
            
        if high_count > 0:
            print("🟠 HIGH PRIORITY ACTIONS:")
            print("   1. Implement proper input validation")
            print("   2. Add file upload security")
            print("   3. Enforce HTTPS/SSL")
            print("   4. Fix authorization flaws")
            print("   5. Update vulnerable dependencies")
            print()
            
        if medium_count > 0:
            print("🟡 MEDIUM PRIORITY ACTIONS:")
            print("   1. Implement security headers")
            print("   2. Add rate limiting")
            print("   3. Improve session management")
            print("   4. Enhance logging security")
            print("   5. Review CORS configuration")
            print()
            
        # Compliance assessment
        print(f"\n📋 COMPLIANCE ASSESSMENT")
        print("-" * 25)
        
        compliance_frameworks = {
            'OWASP Top 10': self._assess_owasp_compliance(),
            'CIS Controls': self._assess_cis_compliance(),
            'NIST Framework': self._assess_nist_compliance(),
            'GDPR': self._assess_gdpr_compliance()
        }
        
        for framework, compliance in compliance_frameworks.items():
            status = "✅ COMPLIANT" if compliance['score'] >= 80 else "❌ NON-COMPLIANT"
            print(f"   {framework}: {compliance['score']}/100 ({status})")
            
        # Save detailed report
        report_data = {
            'security_score': final_score,
            'total_vulnerabilities': total_vulnerabilities,
            'severity_breakdown': {
                'critical': critical_count,
                'high': high_count,
                'medium': medium_count
            },
            'vulnerabilities': dict(self.vulnerabilities),
            'compliance': compliance_frameworks,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('security_audit_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\n💾 Detailed security report saved to: security_audit_report.json")
        
    def _assess_owasp_compliance(self):
        """Assess OWASP Top 10 compliance."""
        owasp_score = 100
        
        # A01:2021 – Broken Access Control
        if self.vulnerabilities.get('authorization'):
            owasp_score -= 15
            
        # A02:2021 – Cryptographic Failures
        if self.vulnerabilities.get('encryption') or self.vulnerabilities.get('hardcoded_secrets'):
            owasp_score -= 15
            
        # A03:2021 – Injection
        if self.vulnerabilities.get('sql_injection') or self.vulnerabilities.get('input_validation'):
            owasp_score -= 15
            
        # A04:2021 – Insecure Design
        if not self.vulnerabilities.get('csrf'):
            owasp_score -= 10
            
        # A05:2021 – Security Misconfiguration
        if self.vulnerabilities.get('security_headers') or self.vulnerabilities.get('docker'):
            owasp_score -= 10
            
        # A06:2021 – Vulnerable Components
        if self.vulnerabilities.get('dependencies'):
            owasp_score -= 10
            
        # A07:2021 – Authentication Failures
        if self.vulnerabilities.get('authentication'):
            owasp_score -= 10
            
        # A08:2021 – Software Integrity Failures
        if self.vulnerabilities.get('docker') or self.vulnerabilities.get('kubernetes'):
            owasp_score -= 5
            
        # A09:2021 – Logging Failures
        if self.vulnerabilities.get('logging'):
            owasp_score -= 5
            
        # A10:2021 – Server-Side Request Forgery
        if self.vulnerabilities.get('input_validation'):
            owasp_score -= 5
            
        return {'score': max(0, owasp_score), 'framework': 'OWASP Top 10 2021'}
        
    def _assess_cis_compliance(self):
        """Assess CIS Controls compliance."""
        cis_score = 100
        
        if self.vulnerabilities.get('hardcoded_secrets'):
            cis_score -= 20
        if self.vulnerabilities.get('dependencies'):
            cis_score -= 15
        if self.vulnerabilities.get('docker') or self.vulnerabilities.get('kubernetes'):
            cis_score -= 15
        if not self.vulnerabilities.get('rate_limiting'):
            cis_score -= 10
            
        return {'score': max(0, cis_score), 'framework': 'CIS Controls v8'}
        
    def _assess_nist_compliance(self):
        """Assess NIST Framework compliance."""
        nist_score = 100
        
        if self.vulnerabilities.get('authentication'):
            nist_score -= 15
        if self.vulnerabilities.get('encryption'):
            nist_score -= 15
        if self.vulnerabilities.get('logging'):
            nist_score -= 10
        if self.vulnerabilities.get('session_management'):
            nist_score -= 10
            
        return {'score': max(0, nist_score), 'framework': 'NIST Cybersecurity Framework'}
        
    def _assess_gdpr_compliance(self):
        """Assess GDPR compliance."""
        gdpr_score = 100
        
        if self.vulnerabilities.get('logging'):
            gdpr_score -= 20  # Data protection
        if self.vulnerabilities.get('encryption'):
            gdpr_score -= 15  # Data encryption
        if self.vulnerabilities.get('authentication'):
            gdpr_score -= 10  # Access controls
            
        return {'score': max(0, gdpr_score), 'framework': 'GDPR'}


if __name__ == "__main__":
    auditor = SecurityAuditor()
    auditor.run_full_audit()