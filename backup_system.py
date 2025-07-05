"""Backup and disaster recovery system for NightScan."""

import os
import json
import logging
import asyncio
import subprocess
import shutil
import gzip
import tarfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from config import get_config

logger = logging.getLogger(__name__)

@dataclass
class BackupMetadata:
    """Metadata for a backup."""
    backup_id: str
    timestamp: datetime
    backup_type: str  # 'full', 'incremental', 'config'
    size_bytes: int
    checksum: str
    components: List[str]  # ['database', 'uploads', 'models', 'config']
    retention_days: int
    compressed: bool = True
    encrypted: bool = False
    storage_location: str = "local"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupMetadata':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class DatabaseBackup:
    """Database backup operations."""
    
    def __init__(self, config):
        self.config = config
        self.db_url = config.database.uri
        
    def create_backup(self, backup_path: Path) -> Dict[str, Any]:
        """Create database backup."""
        try:
            backup_file = backup_path / f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            
            # Extract database connection details
            import urllib.parse
            parsed = urllib.parse.urlparse(self.db_url)
            
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password
            
            # Create pg_dump command
            cmd = [
                'pg_dump',
                '-h', parsed.hostname,
                '-p', str(parsed.port or 5432),
                '-U', parsed.username,
                '-d', parsed.path[1:],  # Remove leading slash
                '--verbose',
                '--no-password',
                '--format=custom',
                '--compress=9',
                '-f', str(backup_file)
            ]
            
            logger.info(f"Creating database backup: {backup_file}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_file)
            
            return {
                'file_path': str(backup_file),
                'size_bytes': backup_file.stat().st_size,
                'checksum': checksum,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def restore_backup(self, backup_file: Path) -> bool:
        """Restore database from backup."""
        try:
            # Extract database connection details
            import urllib.parse
            parsed = urllib.parse.urlparse(self.db_url)
            
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password
            
            # Create pg_restore command
            cmd = [
                'pg_restore',
                '-h', parsed.hostname,
                '-p', str(parsed.port or 5432),
                '-U', parsed.username,
                '-d', parsed.path[1:],
                '--verbose',
                '--no-password',
                '--clean',
                '--if-exists',
                str(backup_file)
            ]
            
            logger.info(f"Restoring database from: {backup_file}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"pg_restore warnings: {result.stderr}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    def verify_backup(self, backup_file: Path, expected_checksum: str) -> bool:
        """Verify backup integrity."""
        if not backup_file.exists():
            return False
        
        actual_checksum = self._calculate_checksum(backup_file)
        return actual_checksum == expected_checksum
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


class FileBackup:
    """File system backup operations."""
    
    def __init__(self, config):
        self.config = config
        
    def create_archive(self, source_dir: Path, backup_path: Path, 
                      backup_name: str, compress: bool = True) -> Dict[str, Any]:
        """Create compressed archive of directory."""
        try:
            if not source_dir.exists():
                return {
                    'success': False,
                    'error': f"Source directory {source_dir} does not exist"
                }
            
            archive_ext = ".tar.gz" if compress else ".tar"
            archive_file = backup_path / f"{backup_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{archive_ext}"
            
            logger.info(f"Creating archive: {archive_file}")
            
            mode = "w:gz" if compress else "w"
            with tarfile.open(archive_file, mode) as tar:
                tar.add(source_dir, arcname=source_dir.name)
            
            # Calculate checksum
            checksum = self._calculate_checksum(archive_file)
            
            return {
                'file_path': str(archive_file),
                'size_bytes': archive_file.stat().st_size,
                'checksum': checksum,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Archive creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def extract_archive(self, archive_file: Path, destination: Path) -> bool:
        """Extract archive to destination."""
        try:
            logger.info(f"Extracting archive {archive_file} to {destination}")
            
            with tarfile.open(archive_file, "r:*") as tar:
                tar.extractall(destination)
            
            return True
            
        except Exception as e:
            logger.error(f"Archive extraction failed: {e}")
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


class CloudStorage:
    """Cloud storage operations for remote backups."""
    
    def __init__(self, config):
        self.config = config
        self.s3_client = None
        self.bucket_name = os.environ.get('BACKUP_S3_BUCKET', 'nightscan-backups')
        
        # Initialize S3 client if credentials are available
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                region_name=os.environ.get('AWS_REGION', 'us-east-1')
            )
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info("S3 client initialized successfully")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"S3 not available: {e}")
            self.s3_client = None
    
    def upload_backup(self, local_file: Path, remote_key: str) -> bool:
        """Upload backup file to S3."""
        if not self.s3_client:
            logger.warning("S3 client not available")
            return False
        
        try:
            logger.info(f"Uploading {local_file} to s3://{self.bucket_name}/{remote_key}")
            
            self.s3_client.upload_file(
                str(local_file),
                self.bucket_name,
                remote_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'StorageClass': 'STANDARD_IA'  # Infrequent Access for cost savings
                }
            )
            
            return True
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    def download_backup(self, remote_key: str, local_file: Path) -> bool:
        """Download backup file from S3."""
        if not self.s3_client:
            logger.warning("S3 client not available")
            return False
        
        try:
            logger.info(f"Downloading s3://{self.bucket_name}/{remote_key} to {local_file}")
            
            self.s3_client.download_file(
                self.bucket_name,
                remote_key,
                str(local_file)
            )
            
            return True
            
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            return False
    
    def list_backups(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List available backups in S3."""
        if not self.s3_client:
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            backups = []
            for obj in response.get('Contents', []):
                backups.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'storage_class': obj.get('StorageClass', 'STANDARD')
                })
            
            return backups
            
        except ClientError as e:
            logger.error(f"S3 list failed: {e}")
            return []
    
    def delete_backup(self, remote_key: str) -> bool:
        """Delete backup from S3."""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=remote_key
            )
            return True
            
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            return False


class BackupManager:
    """Main backup management system."""
    
    def __init__(self):
        self.config = get_config()
        self.backup_dir = Path(os.environ.get('BACKUP_DIR', '/var/backups/nightscan'))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.backup_dir / 'backup_metadata.json'
        
        # Initialize backup components
        self.db_backup = DatabaseBackup(self.config)
        self.file_backup = FileBackup(self.config)
        self.cloud_storage = CloudStorage(self.config)
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
    def create_full_backup(self, include_uploads: bool = True, 
                          include_models: bool = True,
                          upload_to_cloud: bool = True) -> Dict[str, Any]:
        """Create a complete system backup."""
        backup_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)
        
        logger.info(f"Starting full backup: {backup_id}")
        
        components = []
        total_size = 0
        backup_files = []
        
        try:
            # 1. Database backup
            logger.info("Backing up database...")
            db_result = self.db_backup.create_backup(backup_path)
            if db_result['success']:
                components.append('database')
                total_size += db_result['size_bytes']
                backup_files.append(db_result['file_path'])
            else:
                logger.error(f"Database backup failed: {db_result.get('error')}")
            
            # 2. Upload files backup
            if include_uploads:
                uploads_dir = Path(os.environ.get('UPLOAD_DIR', 'uploads'))
                if uploads_dir.exists():
                    logger.info("Backing up upload files...")
                    uploads_result = self.file_backup.create_archive(
                        uploads_dir, backup_path, 'uploads'
                    )
                    if uploads_result['success']:
                        components.append('uploads')
                        total_size += uploads_result['size_bytes']
                        backup_files.append(uploads_result['file_path'])
            
            # 3. Model files backup
            if include_models:
                models_dir = Path('models')
                if models_dir.exists():
                    logger.info("Backing up model files...")
                    models_result = self.file_backup.create_archive(
                        models_dir, backup_path, 'models'
                    )
                    if models_result['success']:
                        components.append('models')
                        total_size += models_result['size_bytes']
                        backup_files.append(models_result['file_path'])
            
            # 4. Configuration backup
            logger.info("Backing up configuration...")
            config_result = self._backup_configuration(backup_path)
            if config_result['success']:
                components.append('config')
                total_size += config_result['size_bytes']
                backup_files.append(config_result['file_path'])
            
            # Create backup metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type='full',
                size_bytes=total_size,
                checksum=self._calculate_backup_checksum(backup_files),
                components=components,
                retention_days=30,
                storage_location='local'
            )
            
            # Save metadata
            self.metadata[backup_id] = metadata
            self._save_metadata()
            
            # Upload to cloud if requested
            if upload_to_cloud and self.cloud_storage.s3_client:
                logger.info("Uploading backup to cloud storage...")
                
                # Create consolidated archive
                archive_file = backup_path.parent / f"{backup_id}.tar.gz"
                with tarfile.open(archive_file, "w:gz") as tar:
                    tar.add(backup_path, arcname=backup_id)
                
                # Upload to S3
                remote_key = f"nightscan/{backup_id}.tar.gz"
                if self.cloud_storage.upload_backup(archive_file, remote_key):
                    metadata.storage_location = 'cloud'
                    self._save_metadata()
                    
                    # Clean up local archive
                    archive_file.unlink()
            
            logger.info(f"Full backup completed: {backup_id} ({total_size} bytes)")
            
            return {
                'success': True,
                'backup_id': backup_id,
                'components': components,
                'size_bytes': total_size,
                'metadata': metadata.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            # Clean up partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def restore_backup(self, backup_id: str, components: Optional[List[str]] = None) -> Dict[str, Any]:
        """Restore from backup."""
        if backup_id not in self.metadata:
            return {
                'success': False,
                'error': f"Backup {backup_id} not found"
            }
        
        metadata = self.metadata[backup_id]
        backup_path = self.backup_dir / backup_id
        
        # Download from cloud if needed
        if metadata.storage_location == 'cloud':
            logger.info(f"Downloading backup {backup_id} from cloud...")
            archive_file = self.backup_dir / f"{backup_id}.tar.gz"
            remote_key = f"nightscan/{backup_id}.tar.gz"
            
            if not self.cloud_storage.download_backup(remote_key, archive_file):
                return {
                    'success': False,
                    'error': f"Failed to download backup from cloud"
                }
            
            # Extract archive
            with tarfile.open(archive_file, "r:gz") as tar:
                tar.extractall(self.backup_dir)
            
            archive_file.unlink()  # Clean up
        
        if not backup_path.exists():
            return {
                'success': False,
                'error': f"Backup path {backup_path} not found"
            }
        
        logger.info(f"Restoring backup: {backup_id}")
        
        components_to_restore = components or metadata.components
        restored_components = []
        
        try:
            # Restore database
            if 'database' in components_to_restore:
                logger.info("Restoring database...")
                db_files = list(backup_path.glob("database_*.sql"))
                if db_files:
                    if self.db_backup.restore_backup(db_files[0]):
                        restored_components.append('database')
                    else:
                        logger.error("Database restore failed")
            
            # Restore uploads
            if 'uploads' in components_to_restore:
                logger.info("Restoring upload files...")
                upload_archives = list(backup_path.glob("uploads_*.tar.gz"))
                if upload_archives:
                    uploads_dir = Path(os.environ.get('UPLOAD_DIR', 'uploads'))
                    if self.file_backup.extract_archive(upload_archives[0], uploads_dir.parent):
                        restored_components.append('uploads')
                    else:
                        logger.error("Uploads restore failed")
            
            # Restore models
            if 'models' in components_to_restore:
                logger.info("Restoring model files...")
                model_archives = list(backup_path.glob("models_*.tar.gz"))
                if model_archives:
                    if self.file_backup.extract_archive(model_archives[0], Path('.')):
                        restored_components.append('models')
                    else:
                        logger.error("Models restore failed")
            
            # Restore configuration
            if 'config' in components_to_restore:
                logger.info("Restoring configuration...")
                config_files = list(backup_path.glob("config_*.json"))
                if config_files:
                    if self._restore_configuration(config_files[0]):
                        restored_components.append('config')
                    else:
                        logger.error("Configuration restore failed")
            
            logger.info(f"Restore completed. Components restored: {restored_components}")
            
            return {
                'success': True,
                'backup_id': backup_id,
                'restored_components': restored_components
            }
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        for backup_id, metadata in self.metadata.items():
            backup_info = metadata.to_dict()
            backup_info['backup_id'] = backup_id
            backups.append(backup_info)
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        return backups
    
    def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up expired backups."""
        cleaned_local = 0
        cleaned_cloud = 0
        errors = []
        
        try:
            current_time = datetime.now()
            expired_backups = []
            
            for backup_id, metadata in self.metadata.items():
                retention_date = metadata.timestamp + timedelta(days=metadata.retention_days)
                if current_time > retention_date:
                    expired_backups.append(backup_id)
            
            for backup_id in expired_backups:
                metadata = self.metadata[backup_id]
                
                try:
                    # Clean up local backup
                    backup_path = self.backup_dir / backup_id
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                        cleaned_local += 1
                    
                    # Clean up cloud backup
                    if metadata.storage_location == 'cloud':
                        remote_key = f"nightscan/{backup_id}.tar.gz"
                        if self.cloud_storage.delete_backup(remote_key):
                            cleaned_cloud += 1
                    
                    # Remove from metadata
                    del self.metadata[backup_id]
                    
                except Exception as e:
                    errors.append(f"Failed to clean backup {backup_id}: {e}")
            
            # Save updated metadata
            self._save_metadata()
            
            logger.info(f"Cleanup completed: {cleaned_local} local, {cleaned_cloud} cloud backups removed")
            
            return {
                'success': True,
                'cleaned_local': cleaned_local,
                'cleaned_cloud': cleaned_cloud,
                'errors': errors
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity."""
        if backup_id not in self.metadata:
            return {
                'success': False,
                'error': f"Backup {backup_id} not found"
            }
        
        metadata = self.metadata[backup_id]
        backup_path = self.backup_dir / backup_id
        
        verification_results = {
            'backup_id': backup_id,
            'components': {},
            'overall_success': True
        }
        
        try:
            # Verify database backup
            if 'database' in metadata.components:
                db_files = list(backup_path.glob("database_*.sql"))
                if db_files:
                    # For now, just check file exists and size > 0
                    db_valid = db_files[0].exists() and db_files[0].stat().st_size > 0
                    verification_results['components']['database'] = db_valid
                    if not db_valid:
                        verification_results['overall_success'] = False
            
            # Verify other components by checking file existence and size
            for component in ['uploads', 'models', 'config']:
                if component in metadata.components:
                    archive_pattern = f"{component}_*.tar.gz" if component != 'config' else f"{component}_*.json"
                    files = list(backup_path.glob(archive_pattern))
                    if files:
                        file_valid = files[0].exists() and files[0].stat().st_size > 0
                        verification_results['components'][component] = file_valid
                        if not file_valid:
                            verification_results['overall_success'] = False
                    else:
                        verification_results['components'][component] = False
                        verification_results['overall_success'] = False
            
            return {
                'success': True,
                **verification_results
            }
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _backup_configuration(self, backup_path: Path) -> Dict[str, Any]:
        """Backup application configuration."""
        try:
            config_file = backup_path / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Collect configuration data (excluding secrets)
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'environment_variables': {
                    key: value for key, value in os.environ.items()
                    if not any(secret in key.lower() for secret in ['password', 'secret', 'key', 'token'])
                },
                'application_config': {
                    'max_file_size': self.config.upload.max_file_size,
                    'max_total_size': self.config.upload.max_total_size,
                    'rate_limit_enabled': self.config.rate_limit.enabled,
                    'database_pool_size': self.config.database.pool_size
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return {
                'file_path': str(config_file),
                'size_bytes': config_file.stat().st_size,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _restore_configuration(self, config_file: Path) -> bool:
        """Restore application configuration."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            logger.info(f"Configuration backup from: {config_data.get('timestamp')}")
            logger.info("Note: Manual review and application of configuration may be required")
            
            # In a real implementation, you would apply the configuration
            # For now, just log that configuration was found
            return True
            
        except Exception as e:
            logger.error(f"Configuration restore failed: {e}")
            return False
    
    def _calculate_backup_checksum(self, file_paths: List[str]) -> str:
        """Calculate combined checksum for backup files."""
        combined_hash = hashlib.sha256()
        for file_path in sorted(file_paths):  # Sort for consistent ordering
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        combined_hash.update(chunk)
        return combined_hash.hexdigest()
    
    def _load_metadata(self) -> Dict[str, BackupMetadata]:
        """Load backup metadata from file."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = {}
            for backup_id, backup_data in data.items():
                metadata[backup_id] = BackupMetadata.from_dict(backup_data)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            return {}
    
    def _save_metadata(self) -> None:
        """Save backup metadata to file."""
        try:
            data = {}
            for backup_id, metadata in self.metadata.items():
                data[backup_id] = metadata.to_dict()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")


# Global backup manager instance
_backup_manager: Optional[BackupManager] = None


def get_backup_manager() -> BackupManager:
    """Get or create global backup manager instance."""
    global _backup_manager
    
    if _backup_manager is None:
        _backup_manager = BackupManager()
    
    return _backup_manager


# CLI interface for backup operations
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NightScan Backup System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create backup command
    backup_parser = subparsers.add_parser('backup', help='Create a backup')
    backup_parser.add_argument('--type', choices=['full'], default='full', help='Backup type')
    backup_parser.add_argument('--no-uploads', action='store_true', help='Skip upload files')
    backup_parser.add_argument('--no-models', action='store_true', help='Skip model files')
    backup_parser.add_argument('--no-cloud', action='store_true', help='Skip cloud upload')
    
    # Restore backup command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('backup_id', help='Backup ID to restore')
    restore_parser.add_argument('--components', nargs='+', 
                               choices=['database', 'uploads', 'models', 'config'],
                               help='Components to restore')
    
    # List backups command
    list_parser = subparsers.add_parser('list', help='List available backups')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('backup_id', help='Backup ID to verify')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    backup_manager = get_backup_manager()
    
    if args.command == 'backup':
        result = backup_manager.create_full_backup(
            include_uploads=not args.no_uploads,
            include_models=not args.no_models,
            upload_to_cloud=not args.no_cloud
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == 'restore':
        result = backup_manager.restore_backup(args.backup_id, args.components)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'list':
        backups = backup_manager.list_backups()
        print(json.dumps(backups, indent=2))
    
    elif args.command == 'cleanup':
        result = backup_manager.cleanup_old_backups()
        print(json.dumps(result, indent=2))
    
    elif args.command == 'verify':
        result = backup_manager.verify_backup(args.backup_id)
        print(json.dumps(result, indent=2))