# GitHub Actions Workflows

This directory contains all CI/CD workflows for NightScan.

## Workflows Overview

### Core Workflows

- **`ci.yml`** - Runs on every push/PR: linting, tests, security scans
- **`docker-build.yml`** - Builds and pushes Docker images to registry
- **`release.yml`** - Creates new releases with proper versioning

### Deployment Workflows

- **`deploy-staging.yml`** - Deploys to staging environment (auto on develop branch)
- **`deploy-production.yml`** - Deploys to production with manual approval
- **`deploy-vps-lite.yml`** - Deploys to resource-constrained VPS servers
- **`build-pi-image.yml`** - Builds Raspberry Pi SD card images

### Reusable Components

- **`reusable-tests.yml`** - Shared test workflow for consistency
- **`../actions/`** - Composite actions for common tasks

## Quick Start

```bash
# Install GitHub CLI
brew install gh  # macOS
# or see: https://cli.github.com/

# Login to GitHub
gh auth login

# List workflows
gh workflow list

# Run a workflow manually
gh workflow run deploy-staging.yml

# View workflow runs
gh run list

# Watch a running workflow
gh run watch
```

## Required Secrets

Configure these in Settings → Secrets and variables → Actions:

### Deployment Secrets
- `STAGING_HOST`, `STAGING_USER`, `STAGING_SSH_KEY`
- `PRODUCTION_HOST`, `PRODUCTION_USER`, `PRODUCTION_SSH_KEY`
- `VPS_SSH_KEY`, `VPS_USER`

### Service Secrets
- `STAGING_DATABASE_URL`, `STAGING_SECRET_KEY`
- `VPS_SECRET_KEY`
- `CODECOV_TOKEN` (optional)
- `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ZONE_ID` (optional)

## Documentation

- [Full CI/CD Guide](../../docs/CI_CD_GUIDE.md)
- [Migration from Scripts](../../docs/MIGRATION_TO_CICD.md)

## Workflow Status Badges

Add these to your README:

```markdown
![CI](https://github.com/USER/REPO/workflows/CI%20-%20Build%20and%20Test/badge.svg)
![Docker](https://github.com/USER/REPO/workflows/Docker%20-%20Build%20and%20Push/badge.svg)
```