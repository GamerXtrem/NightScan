#!/usr/bin/env python3
"""
Validate required environment variables for NightScan deployment.

This script ensures all required secrets and configuration are properly set
before starting the application.
"""

import os
import sys
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation severity levels."""
    CRITICAL = "critical"  # Must be set, no defaults allowed
    REQUIRED = "required"  # Must be set, but can have secure defaults
    RECOMMENDED = "recommended"  # Should be set for production
    OPTIONAL = "optional"  # Nice to have


@dataclass
class EnvVariable:
    """Environment variable definition."""
    name: str
    description: str
    level: ValidationLevel
    pattern: Optional[str] = None  # Regex pattern for validation
    min_length: Optional[int] = None
    example: Optional[str] = None
    
    def validate(self, value: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Validate the variable value."""
        if not value:
            return False, f"{self.name} is not set"
            
        # Check minimum length
        if self.min_length and len(value) < self.min_length:
            return False, f"{self.name} must be at least {self.min_length} characters"
            
        # Check pattern
        if self.pattern and not re.match(self.pattern, value):
            return False, f"{self.name} does not match required pattern"
            
        # Check for dangerous defaults
        dangerous_values = [
            "your-secret-key-here",
            "your-jwt-secret-here",
            "your-csrf-secret-key",
            "nightscan_secret",
            "redis_secret",
            "admin",
            "password",
            "123456",
            "default",
            "example",
            "test"
        ]
        
        if value.lower() in dangerous_values:
            return False, f"{self.name} contains a dangerous default value"
            
        # Check for placeholder patterns
        if re.search(r'(TODO|FIXME|CHANGE_ME|REPLACE_ME)', value, re.IGNORECASE):
            return False, f"{self.name} contains a placeholder value"
            
        return True, None


# Define all environment variables
ENVIRONMENT_VARIABLES = [
    # Critical Security Variables
    EnvVariable(
        "SECRET_KEY",
        "Flask application secret key",
        ValidationLevel.CRITICAL,
        min_length=32,
        example="Generated with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
    ),
    EnvVariable(
        "CSRF_SECRET_KEY",
        "CSRF protection secret key",
        ValidationLevel.CRITICAL,
        min_length=32,
        example="Generated with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
    ),
    EnvVariable(
        "DB_PASSWORD",
        "PostgreSQL database password",
        ValidationLevel.CRITICAL,
        min_length=16,
        pattern=r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$',
        example="Complex password with uppercase, lowercase, numbers, and symbols"
    ),
    EnvVariable(
        "REDIS_PASSWORD",
        "Redis cache password",
        ValidationLevel.CRITICAL,
        min_length=16,
        pattern=r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$',
        example="Complex password with uppercase, lowercase, numbers, and symbols"
    ),
    EnvVariable(
        "GRAFANA_PASSWORD",
        "Grafana admin password",
        ValidationLevel.CRITICAL,
        min_length=12,
        example="Strong password for Grafana admin access"
    ),
    
    # Required Configuration
    EnvVariable(
        "NIGHTSCAN_ENV",
        "Deployment environment",
        ValidationLevel.REQUIRED,
        pattern=r'^(development|staging|production)$',
        example="production"
    ),
    
    # API Keys (if using external services)
    EnvVariable(
        "JWT_SECRET_KEY",
        "JWT token signing key",
        ValidationLevel.RECOMMENDED,
        min_length=32,
        example="Generated with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
    ),
    EnvVariable(
        "AWS_ACCESS_KEY_ID",
        "AWS access key for backups",
        ValidationLevel.OPTIONAL,
        pattern=r'^AKIA[0-9A-Z]{16}$',
        example="AKIAIOSFODNN7EXAMPLE"
    ),
    EnvVariable(
        "AWS_SECRET_ACCESS_KEY",
        "AWS secret access key",
        ValidationLevel.OPTIONAL,
        min_length=40,
        example="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    ),
    
    # Port Configuration
    EnvVariable(
        "WEB_PORT",
        "Web application port",
        ValidationLevel.OPTIONAL,
        pattern=r'^\d{1,5}$',
        example="8000"
    ),
    EnvVariable(
        "POSTGRES_PORT",
        "PostgreSQL port",
        ValidationLevel.OPTIONAL,
        pattern=r'^\d{1,5}$',
        example="5432"
    ),
    EnvVariable(
        "REDIS_PORT",
        "Redis port",
        ValidationLevel.OPTIONAL,
        pattern=r'^\d{1,5}$',
        example="6379"
    ),
]


def validate_environment() -> Tuple[bool, List[str], List[str]]:
    """
    Validate all environment variables.
    
    Returns:
        Tuple of (success, errors, warnings)
    """
    errors = []
    warnings = []
    
    for var in ENVIRONMENT_VARIABLES:
        value = os.environ.get(var.name)
        
        if not value:
            if var.level == ValidationLevel.CRITICAL:
                errors.append(f"CRITICAL: {var.name} - {var.description} (MUST be set)")
                if var.example:
                    errors.append(f"  Example: {var.example}")
            elif var.level == ValidationLevel.REQUIRED:
                errors.append(f"REQUIRED: {var.name} - {var.description}")
                if var.example:
                    errors.append(f"  Example: {var.example}")
            elif var.level == ValidationLevel.RECOMMENDED:
                warnings.append(f"RECOMMENDED: {var.name} - {var.description}")
        else:
            # Validate the value
            valid, error_msg = var.validate(value)
            if not valid:
                if var.level in [ValidationLevel.CRITICAL, ValidationLevel.REQUIRED]:
                    errors.append(f"ERROR: {error_msg}")
                    if var.example:
                        errors.append(f"  Example: {var.example}")
                else:
                    warnings.append(f"WARNING: {error_msg}")
    
    return len(errors) == 0, errors, warnings


def generate_env_template() -> str:
    """Generate a .env template file content."""
    lines = ["# NightScan Environment Configuration Template"]
    lines.append("# Generated by validate_env.py")
    lines.append("# Copy this to .env and fill in your values")
    lines.append("")
    
    # Group by level
    levels = {}
    for var in ENVIRONMENT_VARIABLES:
        if var.level not in levels:
            levels[var.level] = []
        levels[var.level].append(var)
    
    # Write each level
    for level in [ValidationLevel.CRITICAL, ValidationLevel.REQUIRED, 
                  ValidationLevel.RECOMMENDED, ValidationLevel.OPTIONAL]:
        if level in levels:
            lines.append(f"\n# {level.value.upper()} Variables")
            lines.append("#" + "=" * 50)
            
            for var in levels[level]:
                lines.append(f"\n# {var.description}")
                if var.example:
                    lines.append(f"# Example: {var.example}")
                if var.pattern:
                    lines.append(f"# Pattern: {var.pattern}")
                if var.min_length:
                    lines.append(f"# Minimum length: {var.min_length}")
                lines.append(f"{var.name}=")
    
    return "\n".join(lines)


def main():
    """Main validation function."""
    print("NightScan Environment Validation")
    print("=" * 50)
    
    # Check if generating template
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-template":
        template = generate_env_template()
        with open(".env.template", "w") as f:
            f.write(template)
        print("Generated .env.template file")
        return
    
    # Validate environment
    success, errors, warnings = validate_environment()
    
    # Display results
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  {error}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if success and not warnings:
        print("\n✅ All environment variables are properly configured!")
    elif success:
        print("\n✅ All required variables are set, but check warnings above.")
    else:
        print("\n❌ Environment validation FAILED!")
        print("\nTo generate a template .env file, run:")
        print("  python scripts/validate_env.py --generate-template")
        sys.exit(1)
    
    # Additional security checks
    print("\n🔒 Security Checks:")
    
    # Check for .env file permissions
    if os.path.exists(".env"):
        stat_info = os.stat(".env")
        mode = oct(stat_info.st_mode)[-3:]
        if mode != "600":
            print(f"  ⚠️  .env file permissions are {mode}, should be 600")
            print("     Run: chmod 600 .env")
        else:
            print("  ✅ .env file permissions are secure (600)")
    
    # Check for secrets in environment
    env_str = str(os.environ)
    if any(secret in env_str.lower() for secret in ["password", "secret", "key", "token"]):
        visible_secrets = [k for k in os.environ.keys() 
                         if any(s in k.lower() for s in ["password", "secret", "key", "token"])]
        if visible_secrets:
            print(f"  ⚠️  Sensitive variables in environment: {', '.join(visible_secrets[:3])}...")
    
    print("\n" + "=" * 50)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()