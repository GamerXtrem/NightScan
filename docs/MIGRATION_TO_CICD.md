# Migration Guide: From Deployment Scripts to CI/CD

This guide helps you migrate from the old bash deployment scripts to the new GitHub Actions CI/CD pipeline.

## Quick Comparison

| Old Method | New Method | Benefits |
|------------|------------|----------|
| `./scripts/deploy.sh` | GitHub Actions workflow | Automated, consistent, tracked |
| Manual SSH and commands | Automated SSH deployment | No manual errors, repeatable |
| No testing before deploy | Automatic tests run first | Catch issues before production |
| Manual version management | Automatic versioning | Consistent version tags |
| No rollback procedure | Automatic rollback on failure | Safer deployments |

## Before You Start

### 1. Set Up GitHub Secrets

You need to add your deployment credentials to GitHub:

1. Go to your repository on GitHub
2. Click Settings → Secrets and variables → Actions
3. Add the following secrets:

```bash
# For staging deployment
STAGING_HOST=your-staging-server.com
STAGING_USER=deploy_user
STAGING_SSH_KEY=<paste your private SSH key>
STAGING_DATABASE_URL=postgresql://user:pass@localhost/db
STAGING_SECRET_KEY=<your-django-secret-key>

# For production deployment
PRODUCTION_HOST=your-production-server.com
PRODUCTION_USER=deploy_user
PRODUCTION_SSH_KEY=<paste your private SSH key>

# For VPS deployment
VPS_SSH_KEY=<paste your VPS SSH key>
VPS_USER=root
VPS_SECRET_KEY=<your-django-secret-key>
```

### 2. Update Your Servers

The servers need to be ready for automated deployment:

```bash
# On each server, ensure docker and docker-compose are installed
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## Migration Steps

### Step 1: Stop Using Old Scripts

⚠️ **Important**: Do not use the old scripts once you start using CI/CD to avoid conflicts.

```bash
# Mark old scripts as deprecated
cd scripts/
for script in deploy*.sh; do
  mv "$script" "${script}.deprecated"
done
```

### Step 2: First CI/CD Deployment

#### For Staging:
```bash
# Push to develop branch
git checkout develop
git push origin develop
# This automatically triggers staging deployment
```

#### For Production:
```bash
# Create a version tag
git tag v1.0.0
git push origin v1.0.0
# This triggers production deployment with approval
```

#### For VPS:
```bash
# Manually trigger VPS deployment
gh workflow run deploy-vps-lite.yml \
  -f target_host=YOUR_VPS_IP \
  -f deploy_branch=main \
  -f resource_mode=balanced
```

### Step 3: Monitor Your First Deployment

1. Go to GitHub Actions tab in your repository
2. Click on the running workflow
3. Watch the real-time logs
4. Check for any errors

## Mapping Old Scripts to New Workflows

### `deploy.sh` → Production Workflow

**Old way**:
```bash
./scripts/deploy.sh --env production --backup
```

**New way**:
```bash
# Automatic with version tags
git tag v1.2.3
git push origin v1.2.3

# Or manual
gh workflow run deploy-production.yml -f version=v1.2.3
```

### `deploy-vps-lite.sh` → VPS Lite Workflow

**Old way**:
```bash
./scripts/deploy-vps-lite.sh 192.168.1.100
```

**New way**:
```bash
gh workflow run deploy-vps-lite.yml \
  -f target_host=192.168.1.100 \
  -f resource_mode=minimal
```

### `setup-ssl.sh` → Automated in Workflows

SSL setup is now automated in the deployment workflows. No manual steps needed.

### `backup-database.sh` → Automatic Before Production Deploy

Backups are automatically created before each production deployment.

## Common Scenarios

### Deploying a Hotfix

**Old way**:
```bash
ssh production-server
cd /opt/nightscan
git pull
docker-compose restart
```

**New way**:
```bash
# Create hotfix branch
git checkout -b hotfix/critical-fix main
# Make fix and commit
git add .
git commit -m "Fix critical issue"
# Push and create PR
git push origin hotfix/critical-fix
# After PR merge, tag and deploy
git checkout main
git pull
git tag v1.2.4
git push origin v1.2.4
```

### Rolling Back

**Old way**:
```bash
ssh production-server
# Manually restore backup
# Manually checkout previous version
# Hope nothing breaks
```

**New way**:
```bash
# Automatic rollback on failure
# Or manual rollback to specific version
gh workflow run deploy-production.yml \
  -f version=v1.2.3 \
  -f skip_approval=true
```

### Deploying to Multiple VPS Instances

**Old way**:
```bash
for server in server1 server2 server3; do
  ./scripts/deploy-vps-lite.sh $server
done
```

**New way**:
```bash
# Deploy to each VPS
for server in server1 server2 server3; do
  gh workflow run deploy-vps-lite.yml \
    -f target_host=$server \
    -f resource_mode=balanced
done
```

## Troubleshooting Migration Issues

### Issue: Deployment fails with "Permission denied"

**Solution**: Check that your SSH key is correctly added to GitHub secrets:
```bash
# Test SSH key locally
ssh -i ~/.ssh/deploy_key user@server

# Copy the PRIVATE key content
cat ~/.ssh/deploy_key
# Paste into GitHub secret (including BEGIN/END lines)
```

### Issue: "Docker command not found"

**Solution**: Docker needs to be installed on the target server:
```bash
# SSH to server and install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Logout and login again
```

### Issue: Old configuration conflicts

**Solution**: Clean up old deployment artifacts:
```bash
# On the server
cd /opt/nightscan
docker-compose down
docker system prune -af
rm -rf .env docker-compose.override.yml
# Let CI/CD recreate everything
```

## Benefits You'll Notice

1. **Consistency**: Every deployment follows the same steps
2. **Visibility**: See exactly what's happening in real-time
3. **Safety**: Automatic rollbacks protect production
4. **Speed**: Parallel builds and caching make deploys faster
5. **Audit Trail**: Every deployment is logged with who triggered it
6. **Testing**: Code is tested before it reaches any environment

## Tips for Success

1. **Start with staging**: Test the CI/CD with staging first
2. **Use version tags**: Always tag releases for production
3. **Monitor the first few deployments**: Watch the logs to understand the process
4. **Keep secrets secure**: Rotate SSH keys and passwords regularly
5. **Document custom steps**: If you have special deployment needs, document them

## Getting Help

- **View logs**: Click on any workflow run in GitHub Actions
- **Download artifacts**: Use `gh run download <run-id>`
- **Check status**: `gh run list` shows recent runs
- **Re-run failed jobs**: Click "Re-run jobs" in the GitHub UI

## Next Steps

1. Read the full [CI/CD Guide](CI_CD_GUIDE.md)
2. Set up monitoring for your deployments
3. Configure alerts for deployment failures
4. Train your team on the new process

Remember: The old scripts are deprecated. Embrace the automation! 🚀