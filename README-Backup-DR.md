# NightScan Backup and Disaster Recovery

This document covers the comprehensive backup and disaster recovery (DR) system for NightScan.

## Overview

NightScan includes a robust backup and disaster recovery system that provides:

- **Automated backups** with multiple schedules and types
- **Cloud storage integration** for off-site backup storage
- **Real-time health monitoring** and automatic failover
- **Point-in-time recovery** capabilities
- **Disaster recovery orchestration** with automated recovery procedures

## Backup System

### Backup Types

1. **Full Backup**
   - Database, uploads, models, and configuration
   - Recommended frequency: Daily
   - Retention: 30 days

2. **Quick Backup**
   - Database, uploads, and configuration (excludes models)
   - Recommended frequency: Every 4-6 hours
   - Retention: 7 days

3. **Database-Only Backup**
   - Database only for rapid recovery
   - Recommended frequency: Every 1-2 hours
   - Retention: 24 hours

### Backup Components

- **Database**: PostgreSQL dump with compression
- **Upload Files**: User-uploaded audio files and processed results
- **Model Files**: ML models and training data
- **Configuration**: Application settings and environment variables

### Storage Locations

- **Local Storage**: `/var/backups/nightscan/` (default)
- **Cloud Storage**: AWS S3 with server-side encryption
- **Archive Storage**: Long-term retention with lifecycle policies

## Quick Start

### Manual Backup

```bash
# Create a full backup
python backup_system.py backup --type=full

# Create a quick backup (no models, no cloud)
python backup_system.py backup --type=full --no-models --no-cloud

# List all backups
python backup_system.py list

# Verify a backup
python backup_system.py verify backup_id_here
```

### Manual Restore

```bash
# Restore complete backup
python backup_system.py restore backup_id_here

# Restore specific components
python backup_system.py restore backup_id_here --components database uploads

# Restore from cloud
python backup_system.py restore backup_id_here
```

### Automated Backups

1. **Install cron schedule:**
```bash
# Copy example crontab
cp crontab.example /tmp/nightscan-cron
crontab /tmp/nightscan-cron
```

2. **Verify cron installation:**
```bash
crontab -l | grep nightscan
```

## Configuration

### Environment Variables

```bash
# Backup configuration
export BACKUP_DIR="/var/backups/nightscan"
export BACKUP_S3_BUCKET="your-backup-bucket"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key" 
export AWS_REGION="us-east-1"

# Database configuration
export DATABASE_URL="postgresql://user:pass@localhost:5432/nightscan"

# Notification configuration (optional)
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export NOTIFICATION_EMAIL="admin@yourdomain.com"
```

### Backup Policies

Edit `backup_system.py` to customize:

```python
# Retention policies
RETENTION_DAYS = {
    'full': 30,
    'quick': 7,
    'database_only': 1
}

# Compression settings
COMPRESSION_ENABLED = True
COMPRESSION_LEVEL = 9

# Cloud upload settings
CLOUD_UPLOAD_ENABLED = True
STORAGE_CLASS = 'STANDARD_IA'
```

## Disaster Recovery

### Health Monitoring

The DR system continuously monitors:

- **Web Application**: HTTP health checks
- **Prediction API**: ML service availability 
- **Database**: Connection and query performance
- **Cache**: Redis connectivity and performance
- **Storage**: Disk space and I/O performance

### Automatic Failover

Failover triggers:
- 3 consecutive health check failures
- Response time exceeding 5 seconds
- Service unavailability for 90 seconds

Failover targets:
- Secondary web servers
- Backup prediction API instances
- Read replicas for database
- Alternative cache instances

### Recovery Procedures

Automated recovery actions:
1. **Service Restart**: Attempt to restart failed services
2. **Resource Cleanup**: Clear temporary files and reset connections
3. **Configuration Reload**: Apply updated configurations
4. **Data Validation**: Verify data integrity after recovery

## Operations Guide

### Daily Operations

1. **Monitor backup status:**
```bash
# Check recent backups
python backup_system.py list | head -5

# Verify latest backup
python backup_system.py verify $(python backup_system.py list | jq -r '.[0].backup_id')
```

2. **Check system health:**
```bash
# Get DR status
python disaster_recovery.py status

# Manual health check
python disaster_recovery.py monitor --once
```

3. **Review logs:**
```bash
# Backup logs
tail -f /var/log/nightscan/backup.log

# DR logs
tail -f /var/log/nightscan/disaster_recovery.log
```

### Weekly Operations

1. **Backup verification:**
```bash
# Test restore of recent backup
python backup_system.py restore latest_backup_id --dry-run

# Verify cloud backups
aws s3 ls s3://your-backup-bucket/nightscan/
```

2. **Cleanup old backups:**
```bash
python backup_system.py cleanup
```

3. **DR testing:**
```bash
# Test manual failover
python disaster_recovery.py failover web_app --reason="DR test"

# Test failback
python disaster_recovery.py failback web_app --reason="DR test complete"
```

### Emergency Procedures

#### Complete System Failure

1. **Assessment:**
```bash
# Check system status
python disaster_recovery.py status

# Identify failed components
systemctl status nightscan-*
docker ps -a
```

2. **Recovery:**
```bash
# Find latest good backup
python backup_system.py list | head -1

# Restore from backup
python backup_system.py restore backup_id_here

# Restart services
systemctl restart nightscan-web
systemctl restart nightscan-prediction
```

#### Database Corruption

1. **Stop application:**
```bash
systemctl stop nightscan-web
systemctl stop nightscan-prediction
```

2. **Restore database:**
```bash
# Find latest database backup
BACKUP_ID=$(python backup_system.py list | jq -r '.[0].backup_id')

# Restore database only
python backup_system.py restore $BACKUP_ID --components database
```

3. **Verify and restart:**
```bash
# Test database connectivity
pg_isready -h localhost -p 5432

# Restart services
systemctl start nightscan-web
systemctl start nightscan-prediction
```

#### Data Loss Recovery

1. **Identify scope:**
```bash
# Check available backups
python backup_system.py list

# Identify backup before data loss
RECOVERY_BACKUP="backup_20240301_120000"
```

2. **Point-in-time recovery:**
```bash
# Restore specific components
python backup_system.py restore $RECOVERY_BACKUP --components uploads

# Verify data integrity
python verify_data_integrity.py
```

## Monitoring and Alerting

### Metrics

Key metrics monitored:
- Backup success rate
- Backup duration and size
- Recovery time objective (RTO)
- Recovery point objective (RPO)
- Service availability
- Failover frequency

### Alerts

Configure alerts for:
- Backup failures
- Extended backup duration
- Service downtime
- Failover events
- Storage capacity issues

Example Prometheus alerts:
```yaml
groups:
- name: nightscan_backup
  rules:
  - alert: BackupFailed
    expr: nightscan_backup_success == 0
    for: 5m
    annotations:
      summary: "NightScan backup failed"
      
  - alert: ServiceDown
    expr: up{job="nightscan"} == 0
    for: 2m
    annotations:
      summary: "NightScan service is down"
```

### Integration

Integrate with:
- **PagerDuty**: Critical alert escalation
- **Slack**: Real-time notifications
- **Email**: Backup reports and summaries
- **SIEM**: Security event correlation

## Security

### Backup Security

- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- Access control (IAM policies)
- Audit logging (CloudTrail)

### Access Control

```bash
# Backup operator role
sudo groupadd nightscan-backup
sudo usermod -a -G nightscan-backup backup-user

# Permissions
chmod 750 /var/backups/nightscan
chgrp nightscan-backup /var/backups/nightscan
```

### Secrets Management

Never store secrets in backup metadata:
- Use environment variables
- Implement secret rotation
- Audit secret access

## Testing

### Backup Testing

```bash
# Test backup creation
./scripts/test-backup.sh

# Test restore functionality  
./scripts/test-restore.sh

# Performance testing
./scripts/backup-performance-test.sh
```

### DR Testing

```bash
# Monthly DR drill
./scripts/dr-drill.sh

# Failover testing
./scripts/test-failover.sh

# Recovery validation
./scripts/validate-recovery.sh
```

## Troubleshooting

### Common Issues

1. **Backup Failures:**
   - Check disk space
   - Verify database connectivity
   - Review permissions
   - Check network connectivity for cloud uploads

2. **Slow Backups:**
   - Monitor I/O utilization
   - Check compression settings
   - Verify network bandwidth
   - Consider incremental backups

3. **Restore Issues:**
   - Verify backup integrity
   - Check target system resources
   - Validate permissions
   - Review dependency order

### Log Analysis

```bash
# Backup logs
grep ERROR /var/log/nightscan/backup.log

# DR logs
grep FAILOVER /var/log/nightscan/disaster_recovery.log

# System logs
journalctl -u nightscan-* --since "1 hour ago"
```

### Performance Tuning

```bash
# Database backup optimization
export PGDUMP_JOBS=4  # Parallel jobs
export PGDUMP_COMPRESS=9  # Max compression

# File backup optimization
export TAR_OPTIONS="--use-compress-program=pigz"  # Parallel compression

# Network optimization
export AWS_MAX_BANDWIDTH="100MB/s"  # Limit upload speed
```

## Best Practices

1. **Regular Testing**: Test backups and recovery procedures monthly
2. **Documentation**: Keep runbooks updated and accessible
3. **Monitoring**: Implement comprehensive monitoring and alerting
4. **Security**: Follow security best practices for backup storage
5. **Automation**: Minimize manual intervention in recovery procedures
6. **Communication**: Establish clear communication protocols for incidents

## Support

For issues with backup and disaster recovery:

1. Check the logs in `/var/log/nightscan/`
2. Verify system resources and connectivity
3. Review configuration and permissions
4. Test with manual backup/restore commands
5. Consult monitoring dashboards for system health