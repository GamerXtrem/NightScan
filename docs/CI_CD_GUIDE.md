# NightScan CI/CD Guide

This guide explains the continuous integration and deployment (CI/CD) pipeline implemented for NightScan, replacing the previous manual deployment scripts.

## Overview

The NightScan CI/CD pipeline is built using GitHub Actions and provides:

- Automated testing and code quality checks
- Docker image building and registry management
- Multi-environment deployments (staging, production, VPS-lite)
- Raspberry Pi image building
- Release management and versioning

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Commit    │────▶│   CI Tests   │────▶│   Build     │
│   to Git    │     │   & Linting  │     │   Images    │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            │                            │
                    ▼                            ▼                            ▼
            ┌──────────────┐           ┌──────────────┐            ┌──────────────┐
            │   Staging    │           │  Production  │            │   VPS Lite   │
            │  Deployment  │           │  Deployment  │            │  Deployment  │
            └──────────────┘           └──────────────┘            └──────────────┘
```

## Workflows

### 1. CI - Build and Test (`.github/workflows/ci.yml`)

**Triggers**: Push to main/develop, Pull requests

**Jobs**:
- **lint-python**: Runs flake8, black, isort, mypy
- **test-python**: Runs pytest with coverage for Python 3.13
- **security-scan**: Checks for vulnerabilities with safety and bandit
- **build-docker**: Builds all Docker images to verify they compile
- **integration-test**: Runs integration tests with docker-compose

**Usage**:
```bash
# Automatically triggered on push/PR
# Manual trigger:
gh workflow run ci.yml
```

### 2. Docker Build and Push (`.github/workflows/docker-build.yml`)

**Triggers**: Push to main/develop, Version tags, Manual

**Features**:
- Builds multi-architecture images (amd64, arm64, armv7)
- Pushes to GitHub Container Registry (ghcr.io)
- Generates SBOM (Software Bill of Materials)
- Scans for vulnerabilities with Trivy
- Updates deployment manifests with new image tags

**Images Built**:
- `nightscan-web`: Web application
- `nightscan-prediction`: ML prediction API
- `nightscan-worker`: Background worker
- `nightscan-edge`: Raspberry Pi edge computing

### 3. Deploy to Staging (`.github/workflows/deploy-staging.yml`)

**Triggers**: Push to develop branch, Manual

**Process**:
1. SSH to staging server
2. Pull latest images
3. Run database migrations
4. Deploy with zero-downtime rolling update
5. Run health checks
6. Execute smoke tests

**Environment Variables**:
- `STAGING_HOST`: Staging server hostname
- `STAGING_USER`: SSH user
- `STAGING_SSH_KEY`: SSH private key
- `STAGING_DATABASE_URL`: Database connection string

### 4. Deploy to Production (`.github/workflows/deploy-production.yml`)

**Triggers**: Version tags (v*.*.*), Manual with version input

**Safety Features**:
- Manual approval required (can be bypassed for emergencies)
- Pre-deployment health checks
- Automatic database backup before deployment
- Blue-green deployment with health verification
- Automatic rollback on failure

**Process**:
1. Pre-deployment checks
2. Manual approval gate
3. Backup production data
4. Deploy new version
5. Verify deployment
6. Update CDN and monitoring
7. Create GitHub release

### 5. Deploy VPS Lite (`.github/workflows/deploy-vps-lite.yml`)

**Triggers**: Manual only

**Features**:
- Resource-constrained deployment (2-8GB RAM)
- Automatic resource detection
- SQLite database option
- Minimal monitoring stack
- Configurable resource modes: minimal, balanced, performance

**Usage**:
```bash
gh workflow run deploy-vps-lite.yml \
  -f target_host=192.168.1.100 \
  -f resource_mode=balanced
```

### 6. Build Raspberry Pi Image (`.github/workflows/build-pi-image.yml`)

**Triggers**: Changes to NightScanPi/, Manual

**Features**:
- Builds complete SD card images
- Multi-architecture support (arm64, armv7)
- Includes all dependencies and ML models
- First-boot automatic configuration
- Generates checksums

**Outputs**:
- `nightscan-pi-{version}.zip`: Compressed SD card image
- Docker images for containerized deployment

### 7. Release Management (`.github/workflows/release.yml`)

**Triggers**: Manual with version input

**Process**:
1. Determine version (major/minor/patch)
2. Update version files
3. Generate changelog from commits
4. Create git tag
5. Build release artifacts
6. Create GitHub release
7. Trigger deployments
8. Send notifications

## Configuration

### Required Secrets

```yaml
# GitHub Settings → Secrets and variables → Actions

# Container Registry
GITHUB_TOKEN         # Automatically provided
DOCKER_HUB_USERNAME  # Optional: Docker Hub username
DOCKER_HUB_TOKEN     # Optional: Docker Hub access token

# Staging Deployment
STAGING_HOST         # Staging server hostname/IP
STAGING_USER         # SSH username
STAGING_SSH_KEY      # SSH private key (ed25519 recommended)
STAGING_DATABASE_URL # PostgreSQL connection string
STAGING_SECRET_KEY   # Django secret key

# Production Deployment
PRODUCTION_HOST      # Production server hostname/IP
PRODUCTION_USER      # SSH username
PRODUCTION_SSH_KEY   # SSH private key

# VPS Deployment
VPS_SSH_KEY          # SSH key for VPS access
VPS_USER             # SSH username (usually 'root' or 'ubuntu')
VPS_SECRET_KEY       # Django secret key for VPS

# Optional Services
CODECOV_TOKEN        # Codecov.io token for coverage reports
CLOUDFLARE_API_TOKEN # CloudFlare API for cache purging
CLOUDFLARE_ZONE_ID   # CloudFlare zone ID
DISCORD_WEBHOOK      # Discord webhook for notifications
```

### Environment Configuration

Create environments in GitHub:

1. **staging**: Auto-deploys from develop branch
2. **production-approval**: Manual approval for production
3. **production**: Production deployment environment
4. **vps-lite**: For VPS deployments

## Usage Examples

### Deploy to Staging
```bash
# Automatic on push to develop
git push origin develop

# Manual deployment
gh workflow run deploy-staging.yml
```

### Deploy to Production
```bash
# Create a release tag
git tag v1.2.3
git push origin v1.2.3

# Or use release workflow
gh workflow run release.yml -f version=1.2.3 -f release_type=minor
```

### Deploy to VPS
```bash
# Deploy to a specific VPS
gh workflow run deploy-vps-lite.yml \
  -f target_host=vps.example.com \
  -f deploy_branch=main \
  -f resource_mode=minimal
```

### Build Pi Image
```bash
# Build for Pi Zero 2W
gh workflow run build-pi-image.yml \
  -f pi_version=pi-zero-2w \
  -f include_models=true
```

## Monitoring

### Workflow Status
```bash
# List recent workflow runs
gh run list

# Watch a specific run
gh run watch

# View workflow run details
gh run view <run-id>
```

### Deployment Status
- Check GitHub Actions tab for workflow status
- Monitor deployment environments in Settings → Environments
- View deployment history in the Deployments tab

## Rollback Procedures

### Staging Rollback
```bash
# SSH to staging
ssh user@staging.example.com

# Rollback to previous version
cd /opt/nightscan/staging
git checkout <previous-tag>
docker-compose pull
docker-compose up -d
```

### Production Rollback
- Automatic rollback triggers on deployment failure
- Manual rollback:
  ```bash
  gh workflow run deploy-production.yml \
    -f version=<previous-version> \
    -f skip_approval=true
  ```

## Troubleshooting

### Common Issues

1. **SSH Connection Failed**
   - Verify SSH key is correctly added to secrets
   - Check server firewall allows GitHub Actions IPs
   - Ensure SSH key has correct permissions (600)

2. **Docker Pull Failed**
   - Check registry authentication
   - Verify image tags exist
   - Check disk space on target server

3. **Health Check Failed**
   - Check application logs: `docker-compose logs web`
   - Verify database connectivity
   - Check for missing environment variables

4. **Resource Constraints**
   - Use VPS-lite deployment for limited resources
   - Monitor memory usage during deployment
   - Consider staged rollout for large updates

### Logs and Debugging

```bash
# View workflow logs
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>

# SSH to server and check logs
docker-compose logs -f
journalctl -u docker -f
```

## Best Practices

1. **Always test in staging first**
2. **Use semantic versioning for releases**
3. **Monitor resource usage during deployments**
4. **Keep secrets rotated regularly**
5. **Review security scan results**
6. **Document any manual steps in PR descriptions**
7. **Use environments for access control**

## Migration from Old Scripts

The new CI/CD pipeline replaces these scripts:
- `scripts/deploy.sh` → Use production deployment workflow
- `scripts/deploy-vps-lite.sh` → Use VPS-lite workflow
- `scripts/deploy-production.sh` → Use production workflow
- `scripts/deploy-enhanced.sh` → Integrated into workflows
- Individual setup scripts → Automated in workflows

## Support

For issues or questions:
1. Check workflow run logs
2. Review this documentation
3. Check GitHub Actions status: https://www.githubstatus.com/
4. Open an issue with the `ci/cd` label