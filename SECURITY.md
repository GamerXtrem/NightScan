# NightScan Security Guide

This document outlines the security measures implemented in NightScan and provides guidance for secure deployment and operations.

## 🔒 Security Architecture

NightScan implements a defense-in-depth security strategy across multiple layers:

### 1. Secrets Management
- **External Secrets Operator** for Kubernetes secret management
- **HashiCorp Vault** or **AWS Secrets Manager** for secure secret storage
- **No hardcoded secrets** in source code or configuration files
- **Automatic secret rotation** capabilities

### 2. Application Security
- **Input validation** and sanitization for all user inputs
- **SQL injection prevention** using parameterized queries
- **XSS protection** with proper output encoding
- **CSRF protection** with Flask-WTF tokens
- **Content Security Policy** (CSP) headers
- **Rate limiting** to prevent abuse

### 3. Authentication & Authorization
- **Strong password policies** (minimum 10 characters, complexity requirements)
- **Session management** with secure cookies
- **Failed login tracking** with IP-based lockouts
- **Role-based access control** (RBAC)

### 4. Network Security
- **TLS/HTTPS enforcement** for all communications
- **Security headers** (HSTS, X-Frame-Options, etc.)
- **Network policies** in Kubernetes for pod-to-pod communication
- **Ingress security** with proper SSL termination

### 5. Container Security
- **Multi-stage Docker builds** with minimal attack surface
- **Non-root containers** with dedicated user accounts
- **Security scanning** with Trivy or similar tools
- **Resource limits** to prevent resource exhaustion

## 🚨 Critical Security Fixes Implemented

### ✅ Fixed: Hardcoded Secrets in Kubernetes

**Problem:** Secrets were base64 encoded directly in Kubernetes manifests.

**Solution:** Implemented External Secrets Operator with secure backends.

**Before:**
```yaml
apiVersion: v1
kind: Secret
data:
  password: bmlnaHRzY2FuX3NlY3JldA==  # INSECURE!
```

**After:**
```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
spec:
  secretStoreRef:
    name: vault-backend
  data:
  - secretKey: password
    remoteRef:
      key: nightscan/postgres
      property: password
```

## 🛡️ Security Configuration

### Prerequisites

1. **Install External Secrets Operator:**
```bash
kubectl apply -f k8s/secrets-management.yaml
```

2. **Setup Secret Backend:**

#### Option A: HashiCorp Vault
```bash
# Install Vault (production setup)
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault --set server.dev.enabled=true

# Setup secrets
export VAULT_ADDR="http://vault.company.com:8200"
export VAULT_TOKEN="your-vault-token"
./scripts/setup-secrets.sh
```

#### Option B: AWS Secrets Manager
```bash
# Configure AWS credentials
aws configure

# Setup secrets
export SECRET_BACKEND="aws-secrets-manager"
export AWS_REGION="us-east-1"
./scripts/setup-secrets.sh
```

### Environment-Specific Security

#### Development
```bash
# Use local Vault for development
docker run --cap-add=IPC_LOCK -d --name=dev-vault -p 8200:8200 vault:latest

# Initialize with development secrets
export VAULT_ADDR="http://localhost:8200"
export VAULT_TOKEN="dev-token"
./scripts/setup-secrets.sh
```

#### Production
```bash
# Use managed Vault or AWS Secrets Manager
export VAULT_ADDR="https://vault.company.com"
export VAULT_TOKEN="$(cat /etc/vault/token)"
./scripts/setup-secrets.sh

# Verify security configuration
./scripts/security-check.sh
```

## 🔍 Security Monitoring

### Metrics and Alerts

Monitor these security-related metrics:

```yaml
# Prometheus alerts for security events
groups:
- name: security_alerts
  rules:
  - alert: HighFailedLoginRate
    expr: rate(failed_logins_total[5m]) > 0.1
    for: 2m
    annotations:
      summary: "High rate of failed login attempts"
      
  - alert: SecretAccessFailure
    expr: external_secrets_sync_calls_error > 0
    for: 1m
    annotations:
      summary: "Failed to sync secrets from external store"
      
  - alert: UnauthorizedAPIAccess
    expr: rate(http_requests_total{status=~"401|403"}[5m]) > 0.05
    for: 5m
    annotations:
      summary: "High rate of unauthorized API access attempts"
```

### Audit Logging

Enable comprehensive audit logging:

```python
# In web/app.py
import logging
security_logger = logging.getLogger('security')

@app.before_request
def log_security_events():
    if request.endpoint in ['login', 'register', 'api.predict']:
        security_logger.info(
            f"Security event: {request.method} {request.endpoint} "
            f"from {request.remote_addr} user={current_user.get_id()}"
        )
```

### Security Scanning

Regular security scans:

```bash
# Container security scanning
trivy image nightscan/web-app:latest

# Dependency vulnerability scanning
pip-audit -r requirements.txt

# Code security scanning
bandit -r . -f json -o security-report.json
```

## 🔐 Secret Management Best Practices

### Secret Types and Storage

| Secret Type | Storage Location | Rotation Frequency |
|-------------|------------------|-------------------|
| Database passwords | Vault/AWS SM | 90 days |
| API keys | Vault/AWS SM | 180 days |
| JWT signing keys | Vault/AWS SM | 30 days |
| Encryption keys | Vault/AWS SM | 365 days |
| TLS certificates | Cert-Manager | Auto (Let's Encrypt) |

### Secret Rotation

Automated secret rotation:

```bash
# Create rotation script
cat > rotate-secrets.sh << 'EOF'
#!/bin/bash
# Rotate database password
NEW_PASSWORD=$(openssl rand -base64 32)
vault kv put secret/nightscan/postgres password="$NEW_PASSWORD"

# Update database password
kubectl rollout restart deployment/postgres
kubectl rollout restart deployment/web-app
EOF

# Schedule with cron (every 90 days)
0 2 1 */3 * /opt/nightscan/rotate-secrets.sh
```

### Access Control

#### Vault Policies

```hcl
# nightscan-app policy
path "secret/data/nightscan/*" {
  capabilities = ["read"]
}

path "secret/metadata/nightscan/*" {
  capabilities = ["list", "read"]
}
```

#### AWS IAM Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:nightscan/*"
    }
  ]
}
```

## 🛠️ Security Hardening Checklist

### Application Level
- [ ] Input validation on all user inputs
- [ ] Output encoding to prevent XSS
- [ ] Parameterized queries for SQL injection prevention
- [ ] Strong password policies enforced
- [ ] Session timeout configured
- [ ] Rate limiting implemented
- [ ] CSRF protection enabled
- [ ] Security headers configured

### Infrastructure Level
- [ ] TLS/HTTPS enforced everywhere
- [ ] Network policies defined
- [ ] Pod security policies/standards applied
- [ ] Resource limits configured
- [ ] Non-root containers
- [ ] Read-only root filesystems where possible
- [ ] Security contexts properly set

### Operational Level
- [ ] Secrets stored in external secret manager
- [ ] Regular security scans scheduled
- [ ] Audit logging enabled
- [ ] Monitoring and alerting configured
- [ ] Incident response plan documented
- [ ] Regular security reviews scheduled
- [ ] Staff security training completed

## 🚨 Security Incident Response

### Detection
Monitor for these security indicators:
- Unusual login patterns
- High error rates
- Unauthorized API access
- Resource exhaustion
- Secret access failures

### Response Procedure

1. **Immediate Actions:**
   ```bash
   # Disable compromised user
   kubectl patch user $USER_ID -p '{"spec":{"disabled":true}}'
   
   # Rotate compromised secrets
   ./scripts/rotate-secrets.sh $SECRET_NAME
   
   # Scale down affected services if needed
   kubectl scale deployment/web-app --replicas=0
   ```

2. **Investigation:**
   - Review audit logs
   - Check access patterns
   - Analyze compromise scope
   - Document findings

3. **Recovery:**
   - Apply security patches
   - Update compromised credentials
   - Restore services
   - Monitor for continued threats

4. **Post-Incident:**
   - Conduct post-mortem
   - Update security measures
   - Improve detection capabilities
   - Update incident response procedures

## 📞 Security Contacts

- **Security Team:** security@nightscan.com
- **Incident Response:** incident@nightscan.com
- **Vulnerability Reports:** security-reports@nightscan.com

## 🔄 Security Updates

This security guide should be reviewed and updated:
- After any security incident
- Quarterly during security reviews
- When new features are added
- When infrastructure changes are made

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [AWS Secrets Manager Best Practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)