#!/usr/bin/env python3
"""
Log Rotation Cron Setup Script for NightScan

This script helps set up automated log maintenance using cron jobs.
It provides templates and validation for different deployment scenarios.

Usage:
    # Show recommended cron jobs
    python scripts/setup_log_cron.py --show

    # Generate crontab entries
    python scripts/setup_log_cron.py --generate > /tmp/nightscan_cron.txt

    # Install cron jobs (requires sudo)
    python scripts/setup_log_cron.py --install

    # Validate existing cron jobs
    python scripts/setup_log_cron.py --validate

    # Remove cron jobs
    python scripts/setup_log_cron.py --remove

    # Custom configuration
    python scripts/setup_log_cron.py --generate --environment production --user nightscan
"""

import os
import sys
import pwd
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class LogCronSetup:
    """Automated cron job setup for log rotation."""
    
    def __init__(self, environment: str = None, user: str = None, project_path: str = None):
        """Initialize cron setup.
        
        Args:
            environment: Target environment (dev/staging/production/raspberry_pi)
            user: User to run cron jobs as
            project_path: Path to NightScan project directory
        """
        self.environment = environment or os.environ.get('NIGHTSCAN_ENV', 'development')
        self.user = user or os.getenv('USER', 'root')
        self.project_path = project_path or str(project_root)
        
        # Python executable
        self.python_exec = sys.executable
        
        # Script paths
        self.maintenance_script = os.path.join(self.project_path, 'scripts', 'log_maintenance.py')
        self.monitor_script = os.path.join(self.project_path, 'scripts', 'log_monitor_dashboard.py')
        
        # Log paths for cron jobs
        self.cron_log_dir = self._get_cron_log_dir()
        
    def _get_cron_log_dir(self) -> str:
        """Get directory for cron job logs."""
        if self.environment == 'production':
            return '/var/log/nightscan/cron'
        elif self.environment == 'staging':
            return '/var/log/nightscan/cron'
        elif self.environment == 'raspberry_pi':
            return '/home/pi/nightscan/logs/cron'
        else:
            return os.path.join(self.project_path, 'logs', 'cron')
    
    def get_cron_jobs(self) -> List[Dict[str, str]]:
        """Get recommended cron jobs for the environment.
        
        Returns:
            List of cron job dictionaries with schedule, command, and description
        """
        jobs = []
        
        # Ensure cron log directory exists
        os.makedirs(self.cron_log_dir, exist_ok=True)
        
        # Base command template
        base_cmd = f"{self.python_exec} {self.maintenance_script}"
        base_env = f"NIGHTSCAN_ENV={self.environment}"
        
        if self.environment == 'production':
            jobs = [
                {
                    'schedule': '0 2 * * *',  # Daily at 2 AM
                    'command': f'{base_env} {base_cmd} --daily --quiet --log-file {self.cron_log_dir}/daily.log',
                    'description': 'Daily log maintenance and cleanup'
                },
                {
                    'schedule': '0 */6 * * *',  # Every 6 hours
                    'command': f'{base_env} {base_cmd} --status --quiet --log-file {self.cron_log_dir}/status.log',
                    'description': 'Regular health checks'
                },
                {
                    'schedule': '*/15 * * * *',  # Every 15 minutes
                    'command': f'{base_env} {base_cmd} --emergency-check --quiet --log-file {self.cron_log_dir}/emergency.log',
                    'description': 'Emergency disk space checks'
                },
                {
                    'schedule': '0 0 1 * *',  # Monthly on 1st
                    'command': f'{base_env} {base_cmd} --cleanup --retention-days 90 --quiet --log-file {self.cron_log_dir}/monthly.log',
                    'description': 'Monthly deep cleanup (90 day retention)'
                }
            ]
            
        elif self.environment == 'staging':
            jobs = [
                {
                    'schedule': '0 3 * * *',  # Daily at 3 AM
                    'command': f'{base_env} {base_cmd} --daily --quiet --log-file {self.cron_log_dir}/daily.log',
                    'description': 'Daily log maintenance and cleanup'
                },
                {
                    'schedule': '0 */8 * * *',  # Every 8 hours
                    'command': f'{base_env} {base_cmd} --status --quiet --log-file {self.cron_log_dir}/status.log',
                    'description': 'Regular health checks'
                },
                {
                    'schedule': '*/30 * * * *',  # Every 30 minutes
                    'command': f'{base_env} {base_cmd} --emergency-check --quiet --log-file {self.cron_log_dir}/emergency.log',
                    'description': 'Emergency disk space checks'
                }
            ]
            
        elif self.environment == 'raspberry_pi':
            jobs = [
                {
                    'schedule': '0 4 * * *',  # Daily at 4 AM
                    'command': f'{base_env} {base_cmd} --daily --quiet --log-file {self.cron_log_dir}/daily.log',
                    'description': 'Daily log maintenance and cleanup'
                },
                {
                    'schedule': '0 */12 * * *',  # Every 12 hours
                    'command': f'{base_env} {base_cmd} --status --quiet --log-file {self.cron_log_dir}/status.log',
                    'description': 'Regular health checks'
                },
                {
                    'schedule': '*/20 * * * *',  # Every 20 minutes
                    'command': f'{base_env} {base_cmd} --emergency-check --quiet --log-file {self.cron_log_dir}/emergency.log',
                    'description': 'Emergency disk space checks (critical for Pi)'
                },
                {
                    'schedule': '0 0 * * 0',  # Weekly on Sunday
                    'command': f'{base_env} {base_cmd} --cleanup --retention-days 14 --quiet --log-file {self.cron_log_dir}/weekly.log',
                    'description': 'Weekly cleanup (14 day retention for limited storage)'
                }
            ]
            
        else:  # development
            jobs = [
                {
                    'schedule': '0 5 * * *',  # Daily at 5 AM
                    'command': f'{base_env} {base_cmd} --daily --log-file {self.cron_log_dir}/daily.log',
                    'description': 'Daily log maintenance (development)'
                },
                {
                    'schedule': '0 */4 * * *',  # Every 4 hours
                    'command': f'{base_env} {base_cmd} --status --log-file {self.cron_log_dir}/status.log',
                    'description': 'Health checks (development)'
                }
            ]
        
        return jobs
    
    def show_cron_jobs(self):
        """Display recommended cron jobs."""
        jobs = self.get_cron_jobs()
        
        print(f"Recommended Cron Jobs for NightScan ({self.environment} environment)")
        print("=" * 70)
        print(f"User: {self.user}")
        print(f"Project Path: {self.project_path}")
        print(f"Python: {self.python_exec}")
        print(f"Cron Logs: {self.cron_log_dir}")
        print()
        
        for i, job in enumerate(jobs, 1):
            print(f"{i}. {job['description']}")
            print(f"   Schedule: {job['schedule']}")
            print(f"   Command:  {job['command']}")
            print()
        
        print("Notes:")
        print("- All jobs include error handling and logging")
        print("- Emergency checks will automatically perform cleanup if needed")
        print("- Adjust retention periods based on your storage requirements")
        print("- Monitor cron logs regularly for any issues")
    
    def generate_crontab(self) -> str:
        """Generate crontab entries.
        
        Returns:
            Crontab content as string
        """
        jobs = self.get_cron_jobs()
        
        lines = [
            f"# NightScan Log Rotation Cron Jobs",
            f"# Environment: {self.environment}",
            f"# Generated: {os.popen('date').read().strip()}",
            f"# User: {self.user}",
            f"",
            f"# Set PATH to include Python",
            f"PATH=/usr/local/bin:/usr/bin:/bin",
            f"",
            f"# Set environment variables",
            f"NIGHTSCAN_ENV={self.environment}",
            f"PYTHONPATH={self.project_path}",
            f""
        ]
        
        for job in jobs:
            lines.extend([
                f"# {job['description']}",
                f"{job['schedule']} {job['command']}",
                ""
            ])
        
        return "\n".join(lines)
    
    def install_cron_jobs(self) -> bool:
        """Install cron jobs for the specified user.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate crontab content
            crontab_content = self.generate_crontab()
            
            # Get current crontab
            try:
                current_crontab = subprocess.check_output(
                    ['crontab', '-l'],
                    stderr=subprocess.DEVNULL
                ).decode('utf-8')
            except subprocess.CalledProcessError:
                current_crontab = ""
            
            # Check if NightScan jobs already exist
            if "NightScan Log Rotation Cron Jobs" in current_crontab:
                print("⚠️  NightScan cron jobs already exist. Use --remove first or edit manually.")
                return False
            
            # Combine current and new crontab
            combined_crontab = current_crontab.rstrip() + "\n\n" + crontab_content
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.crontab', delete=False) as f:
                f.write(combined_crontab)
                temp_file = f.name
            
            try:
                # Install crontab
                subprocess.run(['crontab', temp_file], check=True)
                print("✅ Cron jobs installed successfully")
                
                # Verify installation
                print("\nInstalled cron jobs:")
                subprocess.run(['crontab', '-l'])
                
                return True
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file)
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install cron jobs: {e}")
            return False
        except Exception as e:
            print(f"❌ Error installing cron jobs: {e}")
            return False
    
    def remove_cron_jobs(self) -> bool:
        """Remove NightScan cron jobs.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current crontab
            try:
                current_crontab = subprocess.check_output(['crontab', '-l']).decode('utf-8')
            except subprocess.CalledProcessError:
                print("ℹ️  No crontab found for current user")
                return True
            
            # Remove NightScan section
            lines = current_crontab.split('\n')
            new_lines = []
            skip_section = False
            
            for line in lines:
                if "NightScan Log Rotation Cron Jobs" in line:
                    skip_section = True
                    continue
                elif skip_section and line.strip() == "" and len(new_lines) > 0 and new_lines[-1].strip() == "":
                    skip_section = False
                    continue
                elif skip_section and line.startswith("#"):
                    continue
                elif skip_section and any(cmd in line for cmd in ['log_maintenance.py', 'log_monitor_dashboard.py']):
                    continue
                else:
                    skip_section = False
                    new_lines.append(line)
            
            # Remove trailing empty lines
            while new_lines and new_lines[-1].strip() == "":
                new_lines.pop()
            
            new_crontab = '\n'.join(new_lines)
            
            if new_crontab.strip():
                # Write updated crontab
                with tempfile.NamedTemporaryFile(mode='w', suffix='.crontab', delete=False) as f:
                    f.write(new_crontab + '\n')
                    temp_file = f.name
                
                try:
                    subprocess.run(['crontab', temp_file], check=True)
                    print("✅ NightScan cron jobs removed successfully")
                    return True
                finally:
                    os.unlink(temp_file)
            else:
                # Remove entire crontab if empty
                subprocess.run(['crontab', '-r'], check=True)
                print("✅ Crontab cleared (was only NightScan jobs)")
                return True
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to remove cron jobs: {e}")
            return False
        except Exception as e:
            print(f"❌ Error removing cron jobs: {e}")
            return False
    
    def validate_cron_jobs(self) -> Dict[str, Any]:
        """Validate existing cron jobs.
        
        Returns:
            Validation results
        """
        validation = {
            'has_crontab': False,
            'has_nightscan_jobs': False,
            'job_count': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check if crontab exists
            current_crontab = subprocess.check_output(['crontab', '-l']).decode('utf-8')
            validation['has_crontab'] = True
            
            # Check for NightScan jobs
            nightscan_lines = [line for line in current_crontab.split('\n') 
                             if 'log_maintenance.py' in line or 'log_monitor_dashboard.py' in line]
            
            validation['has_nightscan_jobs'] = len(nightscan_lines) > 0
            validation['job_count'] = len(nightscan_lines)
            
            if validation['has_nightscan_jobs']:
                # Validate job paths
                for line in nightscan_lines:
                    if self.maintenance_script not in line and self.monitor_script not in line:
                        validation['issues'].append(f"Incorrect script path in: {line}")
                    
                    if f"NIGHTSCAN_ENV={self.environment}" not in line:
                        validation['issues'].append(f"Missing or incorrect environment in: {line}")
                
                # Check for recommended jobs
                recommended_jobs = self.get_cron_jobs()
                
                for job in recommended_jobs:
                    if job['schedule'] not in current_crontab:
                        validation['recommendations'].append(f"Consider adding: {job['description']}")
            else:
                validation['recommendations'].append("No NightScan cron jobs found. Consider installing them.")
            
            # Check log directory
            if not os.path.exists(self.cron_log_dir):
                validation['issues'].append(f"Cron log directory does not exist: {self.cron_log_dir}")
            elif not os.access(self.cron_log_dir, os.W_OK):
                validation['issues'].append(f"Cron log directory is not writable: {self.cron_log_dir}")
            
        except subprocess.CalledProcessError:
            validation['has_crontab'] = False
            validation['recommendations'].append("No crontab found. Consider installing NightScan cron jobs.")
        
        return validation
    
    def show_validation_results(self, validation: Dict[str, Any]):
        """Display validation results."""
        print(f"Cron Job Validation Results ({self.environment} environment)")
        print("=" * 50)
        
        if validation['has_crontab']:
            print("✅ Crontab exists")
        else:
            print("❌ No crontab found")
        
        if validation['has_nightscan_jobs']:
            print(f"✅ Found {validation['job_count']} NightScan cron jobs")
        else:
            print("⚠️  No NightScan cron jobs found")
        
        if validation['issues']:
            print("\n🔍 Issues Found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        
        if validation['recommendations']:
            print("\n💡 Recommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
        
        if not validation['issues'] and validation['has_nightscan_jobs']:
            print("\n✅ All validation checks passed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup NightScan log rotation cron jobs")
    
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--show', action='store_true',
                             help='Show recommended cron jobs')
    action_group.add_argument('--generate', action='store_true',
                             help='Generate crontab entries')
    action_group.add_argument('--install', action='store_true',
                             help='Install cron jobs')
    action_group.add_argument('--remove', action='store_true',
                             help='Remove NightScan cron jobs')
    action_group.add_argument('--validate', action='store_true',
                             help='Validate existing cron jobs')
    
    parser.add_argument('--environment', 
                       choices=['development', 'staging', 'production', 'raspberry_pi'],
                       help='Target environment')
    parser.add_argument('--user', 
                       help='User to run cron jobs as (default: current user)')
    parser.add_argument('--project-path',
                       help='Path to NightScan project directory')
    
    args = parser.parse_args()
    
    # Initialize setup
    setup = LogCronSetup(
        environment=args.environment,
        user=args.user,
        project_path=args.project_path
    )
    
    try:
        if args.show:
            setup.show_cron_jobs()
            
        elif args.generate:
            print(setup.generate_crontab())
            
        elif args.install:
            success = setup.install_cron_jobs()
            sys.exit(0 if success else 1)
            
        elif args.remove:
            success = setup.remove_cron_jobs()
            sys.exit(0 if success else 1)
            
        elif args.validate:
            validation = setup.validate_cron_jobs()
            setup.show_validation_results(validation)
            
            # Exit with appropriate code
            if validation['issues']:
                sys.exit(2)  # Issues found
            elif validation['recommendations']:
                sys.exit(1)  # Recommendations available
            else:
                sys.exit(0)  # All good
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()