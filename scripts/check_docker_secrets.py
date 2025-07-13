#!/usr/bin/env python3
"""
Check Docker Compose files for hardcoded secrets.

This script validates that docker-compose files don't contain
hardcoded secrets or dangerous default values.
"""

import sys
import yaml
import re
from typing import List, Dict, Tuple


# Patterns that indicate potential hardcoded secrets
DANGEROUS_PATTERNS = [
    # Default passwords
    (r':-[a-zA-Z0-9_]{6,}', 'Default password in environment variable'),
    (r'password:\s*["\']?[a-zA-Z0-9_]{6,}', 'Hardcoded password'),
    (r'secret[_-]?key:\s*["\']?[a-zA-Z0-9_]{6,}', 'Hardcoded secret key'),
    
    # Common weak values
    (r'(password|secret|key).*[:=]\s*["\']?(admin|password|123456|default|test)', 'Weak credential'),
    (r'your-.*-here', 'Placeholder value not replaced'),
    (r'change[_-]?me', 'Placeholder value not replaced'),
    (r'todo|fixme', 'TODO/FIXME marker found'),
    
    # Specific service defaults
    (r'POSTGRES_PASSWORD.*:-.*', 'PostgreSQL default password'),
    (r'MYSQL_ROOT_PASSWORD.*:-.*', 'MySQL default root password'),
    (r'REDIS_PASSWORD.*:-.*', 'Redis default password'),
    (r'MONGO_INITDB_ROOT_PASSWORD.*:-.*', 'MongoDB default password'),
]

# Environment variables that must NOT have defaults
REQUIRED_SECRETS = [
    'DB_PASSWORD',
    'DATABASE_PASSWORD',
    'POSTGRES_PASSWORD',
    'MYSQL_ROOT_PASSWORD',
    'REDIS_PASSWORD',
    'SECRET_KEY',
    'FLASK_SECRET_KEY',
    'DJANGO_SECRET_KEY',
    'CSRF_SECRET_KEY',
    'JWT_SECRET_KEY',
    'GRAFANA_PASSWORD',
    'ADMIN_PASSWORD',
]


def check_environment_value(value: str, var_name: str = None) -> List[str]:
    """Check an environment variable value for issues."""
    issues = []
    
    if not isinstance(value, str):
        return issues
        
    # Check for default values in ${VAR:-default} format
    if ':-' in value:
        default_match = re.search(r'\$\{([^:]+):-([^}]+)\}', value)
        if default_match:
            var, default = default_match.groups()
            
            # Check if this is a required secret with a default
            if any(req in var.upper() for req in REQUIRED_SECRETS):
                issues.append(f"Required secret '{var}' has default value: {default}")
                
            # Check if default looks like a real credential
            if len(default) > 6 and not default.startswith('/'):
                if any(word in default.lower() for word in ['password', 'secret', 'key']):
                    issues.append(f"Potential hardcoded secret in default: {default}")
                    
    # Check for dangerous patterns
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            issues.append(f"{description}: {value[:50]}...")
            
    return issues


def check_docker_compose(file_path: str) -> Tuple[bool, List[str]]:
    """
    Check a docker-compose file for hardcoded secrets.
    
    Returns:
        Tuple of (has_issues, list_of_issues)
    """
    issues = []
    
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return True, [f"Failed to parse YAML: {e}"]
        
    if not isinstance(data, dict):
        return True, ["Invalid docker-compose format"]
        
    # Check services
    services = data.get('services', {})
    for service_name, service_config in services.items():
        if not isinstance(service_config, dict):
            continue
            
        # Check environment variables
        env_vars = service_config.get('environment', {})
        
        # Handle list format
        if isinstance(env_vars, list):
            for env_var in env_vars:
                if '=' in env_var:
                    key, value = env_var.split('=', 1)
                    env_issues = check_environment_value(value, key)
                    for issue in env_issues:
                        issues.append(f"Service '{service_name}': {issue}")
                        
        # Handle dict format
        elif isinstance(env_vars, dict):
            for key, value in env_vars.items():
                if value is not None:
                    env_issues = check_environment_value(str(value), key)
                    for issue in env_issues:
                        issues.append(f"Service '{service_name}', env '{key}': {issue}")
                        
        # Check command for passwords
        command = service_config.get('command')
        if command:
            if isinstance(command, list):
                command = ' '.join(command)
            if isinstance(command, str):
                # Check for passwords in command line
                if re.search(r'--password[= ][^ ]+', command):
                    issues.append(f"Service '{service_name}': Password in command line")
                if re.search(r'--requirepass [^ ]+', command):
                    issues.append(f"Service '{service_name}': Redis password in command line")
                    
    # Check for .env file reference without proper gitignore
    if 'env_file' in str(data):
        issues.append("Uses env_file - ensure .env files are in .gitignore")
        
    return len(issues) > 0, issues


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: check_docker_secrets.py <docker-compose.yml>")
        sys.exit(1)
        
    has_issues = False
    all_issues = []
    
    for file_path in sys.argv[1:]:
        file_has_issues, issues = check_docker_compose(file_path)
        if file_has_issues:
            has_issues = True
            all_issues.append((file_path, issues))
            
    if has_issues:
        print("❌ Docker Compose security issues found:\n")
        for file_path, issues in all_issues:
            print(f"File: {file_path}")
            for issue in issues:
                print(f"  - {issue}")
            print()
            
        print("Recommendations:")
        print("  1. Remove all default values for secrets")
        print("  2. Use ${VAR:?VAR is required} syntax to enforce required variables")
        print("  3. Store actual values in .env file (not in git)")
        print("  4. Use a secrets management system in production")
        sys.exit(1)
    else:
        print("✅ No hardcoded secrets found in Docker Compose files")
        

if __name__ == "__main__":
    main()