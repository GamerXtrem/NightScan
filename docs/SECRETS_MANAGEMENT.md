# NightScan Secrets Management Guide

## Overview

This guide describes how to securely manage secrets and sensitive configuration in NightScan. Following these practices is **mandatory** for production deployments.

## Critical Security Issues Identified

The following hardcoded secrets have been identified and must be removed:

1. **Docker Compose Default Values**
   - PostgreSQL password: `nightscan_secret`
   - Redis password: `redis_secret`
   - Secret keys: `your-secret-key-here`, `your-csrf-secret-key`
   - Grafana admin password: `admin`

2. **Configuration Files**
   - Database URLs with embedded credentials
   - Placeholder values in examples

3. **Code Issues**
   - Secrets displayed in console output
   - Weak validation of secret values

## Required Actions

### 1. Use Secure Docker Compose

Replace `docker-compose.yml` with `docker-compose.secure.yml`:

```bash
mv docker-compose.yml docker-compose.insecure.yml
cp docker-compose.secure.yml docker-compose.yml
```

### 2. Generate Strong Secrets

Use the provided script to generate secure secrets:

```bash
# Generate all required secrets
python3 -c "
import secrets
print(f'DB_PASSWORD={secrets.token_urlsafe(32)}')
print(f'REDIS_PASSWORD={secrets.token_urlsafe(32)}')
print(f'SECRET_KEY={secrets.token_urlsafe(64)}')
print(f'CSRF_SECRET_KEY={secrets.token_urlsafe(64)}')
print(f'JWT_SECRET_KEY={secrets.token_urlsafe(64)}')
print(f'GRAFANA_PASSWORD={secrets.token_urlsafe(24)}')
" > .env
```

### 3. Validate Environment

Before starting the application:

```bash
# Validate all required secrets are set
python scripts/validate_env.py

# Generate template if needed
python scripts/validate_env.py --generate-template
```

### 4. Set File Permissions

```bash
# Secure the .env file
chmod 600 .env

# Ensure it's in .gitignore
echo ".env" >> .gitignore
echo ".env.*" >> .gitignore
echo "!.env.example" >> .gitignore
echo "!.env.template" >> .gitignore
```

## Development Setup

### Local Development

1. Copy the example environment:
   ```bash
   cp .env.example .env.development
   ```

2. Generate development secrets:
   ```bash
   python scripts/generate_dev_secrets.py > .env.development
   ```

3. Use development compose file:
   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

### Testing

For automated testing, use separate test secrets:

```bash
# Generate test secrets
export TEST_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export TEST_DB_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Run tests
pytest
```

## Production Deployment

### Option 1: Environment Variables

Set secrets as environment variables on the host:

```bash
# On production server
export DB_PASSWORD='<strong-password>'
export REDIS_PASSWORD='<strong-password>'
export SECRET_KEY='<generated-secret>'
export CSRF_SECRET_KEY='<generated-secret>'
```

### Option 2: Docker Secrets (Swarm Mode)

```yaml
# docker-compose.prod.yml
version: '3.8'

secrets:
  db_password:
    external: true
  redis_password:
    external: true

services:
  web:
    secrets:
      - db_password
      - redis_password
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password
      REDIS_PASSWORD_FILE: /run/secrets/redis_password
```

### Option 3: External Secret Managers

#### AWS Secrets Manager

```python
# Configure in production
export USE_AWS_SECRETS_MANAGER=true
export AWS_REGION=us-east-1
```

#### HashiCorp Vault

```bash
export VAULT_ADDR='https://vault.example.com'
export VAULT_TOKEN='<vault-token>'
```

#### Azure Key Vault

```bash
export AZURE_KEY_VAULT_NAME='nightscan-keyvault'
```

## Secret Rotation

### Manual Rotation

```python
from security.secrets_manager import rotate_secret

# Rotate a specific secret
new_secret = rotate_secret('DB_PASSWORD')
print(f"New DB password: {new_secret}")

# Check which secrets need rotation
from security.secrets_manager import get_secrets_manager
manager = get_secrets_manager()
needs_rotation = manager.check_rotation_needed()
```

### Automated Rotation

Set up a cron job for automatic rotation:

```cron
# Rotate secrets monthly
0 0 1 * * /usr/bin/python3 /app/scripts/rotate_secrets.py
```

## Pre-commit Hooks

Enable secret detection before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Create baseline for existing files
detect-secrets scan > .secrets.baseline

# Run manually
pre-commit run --all-files
```

## Security Checklist

- [ ] No default passwords in docker-compose.yml
- [ ] All secrets use strong, unique values
- [ ] .env file has 600 permissions
- [ ] .env is in .gitignore
- [ ] Pre-commit hooks are installed
- [ ] Secret rotation is configured
- [ ] Production uses external secret manager
- [ ] No secrets in application logs
- [ ] Regular security audits scheduled

## Emergency Procedures

### Suspected Secret Exposure

1. **Immediately rotate all secrets**:
   ```bash
   python scripts/emergency_rotate.py
   ```

2. **Update all running services**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

3. **Audit access logs**:
   ```bash
   python scripts/audit_access.py --since="1 hour ago"
   ```

4. **Notify security team**

### Recovery from Backup

If secrets are lost:

1. Use the encrypted backup:
   ```bash
   python scripts/restore_secrets.py --backup=/secure/backup/secrets.enc
   ```

2. Verify all services:
   ```bash
   python scripts/validate_env.py
   docker-compose ps
   ```

## Best Practices

1. **Never commit secrets** - Use environment variables or secret managers
2. **Use strong secrets** - Minimum 32 characters, randomly generated
3. **Rotate regularly** - Monthly for production, quarterly for staging
4. **Limit access** - Use principle of least privilege
5. **Audit usage** - Log secret access and changes
6. **Encrypt at rest** - Use encrypted storage for secret files
7. **Use HTTPS** - Always transmit secrets over encrypted connections
8. **Separate environments** - Never share secrets between environments

## Troubleshooting

### "Required secret not found" error

```bash
# Check which secrets are missing
python scripts/validate_env.py

# Verify environment variables
env | grep -E "(PASSWORD|SECRET|KEY)" | cut -d'=' -f1
```

### Docker Compose validation fails

```bash
# Test the configuration
docker-compose config

# Check for hardcoded secrets
python scripts/check_docker_secrets.py docker-compose.yml
```

### Secret rotation fails

```bash
# Check permissions
ls -la /app/secrets/

# Verify secret manager connection
python -c "from security.secrets_manager import get_secrets_manager; m = get_secrets_manager(); print(m.get_secret('TEST_KEY', required=False))"
```

## References

- [OWASP Secret Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Docker Secrets Documentation](https://docs.docker.com/engine/swarm/secrets/)
- [12 Factor App - Config](https://12factor.net/config)