#!/usr/bin/env python3
"""
Secure Logging Migration Script for NightScan

This script migrates existing logging statements to use secure logging
with automatic sensitive data detection and redaction. It identifies
potential security issues and provides fixes.

Features:
- Scans codebase for unsafe logging patterns
- Identifies sensitive data exposure risks
- Provides automated fixes for common issues
- Generates migration report
- Validates existing secure logging implementations

Usage:
    # Scan for issues
    python scripts/secure_logging_migration.py --scan

    # Apply automated fixes
    python scripts/secure_logging_migration.py --fix

    # Generate comprehensive report
    python scripts/secure_logging_migration.py --report security_audit.json

    # Test specific patterns
    python scripts/secure_logging_migration.py --test "logger.info(f'User {user.password} logged in')"

    # Enable secure logging system-wide
    python scripts/secure_logging_migration.py --enable-secure-logging
"""

import os
import sys
import re
import json
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sensitive_data_sanitizer import get_sanitizer, SensitivePattern, SensitivityLevel, RedactionLevel


@dataclass
class LoggingIssue:
    """Represents a logging security issue."""
    file_path: str
    line_number: int
    line_content: str
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    suggested_fix: Optional[str] = None
    pattern_matched: Optional[str] = None


@dataclass
class MigrationSummary:
    """Summary of migration results."""
    files_scanned: int
    issues_found: int
    issues_fixed: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    patterns_detected: Dict[str, int]
    migration_time: str


class SecureLoggingMigrator:
    """Handles migration to secure logging practices."""
    
    def __init__(self, project_root: str = None):
        """Initialize the migrator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root) if project_root else project_root
        self.sanitizer = get_sanitizer()
        self.issues: List[LoggingIssue] = []
        self.patterns_detected: Dict[str, int] = {}
        
        # Define unsafe logging patterns
        self._setup_detection_patterns()
        
    def _setup_detection_patterns(self):
        """Setup patterns for detecting unsafe logging."""
        self.unsafe_patterns = [
            # Direct password logging
            {
                'pattern': r'log[^(]*\([^)]*password[^)]*\)',
                'severity': 'critical',
                'description': 'Direct password logging detected',
                'suggested_fix': 'Remove password from log message or use secure logging'
            },
            
            # Token logging
            {
                'pattern': r'log[^(]*\([^)]*token[^)]*\)',
                'severity': 'high',
                'description': 'Token logging detected',
                'suggested_fix': 'Truncate or redact token in log message'
            },
            
            # API key logging
            {
                'pattern': r'log[^(]*\([^)]*api[_-]?key[^)]*\)',
                'severity': 'critical',
                'description': 'API key logging detected',
                'suggested_fix': 'Remove API key from log message'
            },
            
            # Secret logging
            {
                'pattern': r'log[^(]*\([^)]*secret[^)]*\)',
                'severity': 'high',
                'description': 'Secret logging detected',
                'suggested_fix': 'Remove secret from log message'
            },
            
            # Email addresses in logs
            {
                'pattern': r'log[^(]*\([^)]*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}[^)]*\)',
                'severity': 'medium',
                'description': 'Email address logging detected',
                'suggested_fix': 'Consider redacting email addresses'
            },
            
            # Console.log with sensitive data (JavaScript)
            {
                'pattern': r'console\.log\([^)]*(?:password|token|key|secret)[^)]*\)',
                'severity': 'critical',
                'description': 'Console logging of sensitive data (JavaScript)',
                'suggested_fix': 'Remove sensitive data from console.log or use proper logging'
            },
            
            # Exception with full stack trace
            {
                'pattern': r'log[^(]*\([^)]*exc_info\s*=\s*True[^)]*\)',
                'severity': 'medium',
                'description': 'Full exception logging - may expose sensitive data',
                'suggested_fix': 'Consider filtering stack traces in production'
            },
            
            # F-string with variables that might be sensitive
            {
                'pattern': r'log[^(]*\(f[\'"][^\'"]*(password|token|key|secret)[^\'\"]*\{[^}]+\}[^\'\"]*[\'"][^)]*\)',
                'severity': 'high',
                'description': 'F-string logging with potentially sensitive variable',
                'suggested_fix': 'Validate that sensitive variables are not logged'
            },
            
            # Database connection strings
            {
                'pattern': r'log[^(]*\([^)]*://[^:]+:[^@]+@[^)]*\)',
                'severity': 'critical',
                'description': 'Database connection string with credentials',
                'suggested_fix': 'Remove credentials from connection string logs'
            }
        ]
    
    def scan_codebase(self, directories: List[str] = None) -> List[LoggingIssue]:
        """Scan codebase for logging security issues.
        
        Args:
            directories: List of directories to scan (defaults to common directories)
            
        Returns:
            List of logging issues found
        """
        if directories is None:
            directories = [
                'web', 'api', 'ios-app', 'scripts', 'security',
                'unified_prediction_system', '.'
            ]
        
        self.issues = []
        self.patterns_detected = {}
        
        for directory in directories:
            dir_path = self.project_root / directory if self.project_root else Path(directory)
            
            if not dir_path.exists():
                continue
                
            self._scan_directory(dir_path)
        
        return self.issues
    
    def _scan_directory(self, directory: Path):
        """Scan a single directory for issues.
        
        Args:
            directory: Directory path to scan
        """
        # File extensions to scan
        extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs'}
        
        for file_path in directory.rglob('*'):
            if file_path.suffix in extensions and file_path.is_file():
                self._scan_file(file_path)
    
    def _scan_file(self, file_path: Path):
        """Scan a single file for logging issues.
        
        Args:
            file_path: File path to scan
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                self._scan_line(file_path, line_num, line.strip())
                
        except Exception as e:
            print(f"Warning: Could not scan file {file_path}: {e}")
    
    def _scan_line(self, file_path: Path, line_num: int, line: str):
        """Scan a single line for logging issues.
        
        Args:
            file_path: File containing the line
            line_num: Line number
            line: Line content
        """
        # Skip comments and empty lines
        if not line or line.strip().startswith('#') or line.strip().startswith('//'):
            return
        
        # Check against unsafe patterns
        for pattern_def in self.unsafe_patterns:
            pattern = pattern_def['pattern']
            
            if re.search(pattern, line, re.IGNORECASE):
                issue = LoggingIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    line_content=line,
                    issue_type=pattern_def['description'],
                    severity=pattern_def['severity'],
                    description=pattern_def['description'],
                    suggested_fix=pattern_def['suggested_fix'],
                    pattern_matched=pattern
                )
                
                self.issues.append(issue)
                
                # Track pattern usage
                issue_type = pattern_def['description']
                self.patterns_detected[issue_type] = self.patterns_detected.get(issue_type, 0) + 1
        
        # Check for sensitive data using sanitizer
        sanitized_line = self.sanitizer.sanitize(line)
        if sanitized_line != line:
            issue = LoggingIssue(
                file_path=str(file_path),
                line_number=line_num,
                line_content=line,
                issue_type="Sensitive data detected by sanitizer",
                severity="high",
                description="Line contains data that would be redacted by sensitive data sanitizer",
                suggested_fix="Apply secure logging filters to prevent data exposure"
            )
            
            self.issues.append(issue)
            
            self.patterns_detected["Sanitizer detection"] = (
                self.patterns_detected.get("Sanitizer detection", 0) + 1
            )
    
    def apply_automated_fixes(self, dry_run: bool = True) -> Dict[str, Any]:
        """Apply automated fixes for common logging issues.
        
        Args:
            dry_run: If True, only simulate fixes without modifying files
            
        Returns:
            Results of fix application
        """
        fixes_applied = 0
        files_modified = set()
        fix_results = {
            'dry_run': dry_run,
            'fixes_applied': 0,
            'files_modified': [],
            'fixes_by_type': {},
            'errors': []
        }
        
        # Group issues by file
        issues_by_file = {}
        for issue in self.issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)
        
        # Apply fixes file by file
        for file_path, issues in issues_by_file.items():
            try:
                success = self._apply_fixes_to_file(file_path, issues, dry_run)
                
                if success:
                    files_modified.add(file_path)
                    fixes_applied += len(issues)
                    
                    for issue in issues:
                        fix_type = issue.issue_type
                        fix_results['fixes_by_type'][fix_type] = (
                            fix_results['fixes_by_type'].get(fix_type, 0) + 1
                        )
                
            except Exception as e:
                fix_results['errors'].append(f"Error fixing {file_path}: {str(e)}")
        
        fix_results['fixes_applied'] = fixes_applied
        fix_results['files_modified'] = list(files_modified)
        
        return fix_results
    
    def _apply_fixes_to_file(self, file_path: str, issues: List[LoggingIssue], dry_run: bool) -> bool:
        """Apply fixes to a single file.
        
        Args:
            file_path: Path to file to fix
            issues: List of issues in the file
            dry_run: Whether to only simulate fixes
            
        Returns:
            True if fixes were applied successfully
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Sort issues by line number (descending) to avoid offset issues
            issues.sort(key=lambda x: x.line_number, reverse=True)
            
            modifications_made = False
            
            for issue in issues:
                line_idx = issue.line_number - 1
                
                if line_idx < 0 or line_idx >= len(lines):
                    continue
                
                original_line = lines[line_idx]
                fixed_line = self._generate_fix_for_line(original_line, issue)
                
                if fixed_line != original_line:
                    lines[line_idx] = fixed_line
                    modifications_made = True
            
            # Write back to file if not dry run and modifications were made
            if not dry_run and modifications_made:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
            
            return modifications_made
            
        except Exception as e:
            print(f"Error applying fixes to {file_path}: {e}")
            return False
    
    def _generate_fix_for_line(self, line: str, issue: LoggingIssue) -> str:
        """Generate a fix for a specific line.
        
        Args:
            line: Original line content
            issue: Issue to fix
            
        Returns:
            Fixed line content
        """
        fixed_line = line
        
        # Apply specific fixes based on issue type
        if 'password' in issue.issue_type.lower():
            # Remove password from logging
            fixed_line = re.sub(
                r'(password["\s]*[:=]["\s]*)[^"\s,}]+',
                r'\1"***"',
                fixed_line,
                flags=re.IGNORECASE
            )
        
        elif 'token' in issue.issue_type.lower():
            # Truncate tokens
            fixed_line = re.sub(
                r'(token["\s]*[:=]["\s]*"?)([^"\s,}]{8})[^"\s,}]*',
                r'\1\2***',
                fixed_line,
                flags=re.IGNORECASE
            )
        
        elif 'api key' in issue.issue_type.lower():
            # Remove API keys
            fixed_line = re.sub(
                r'(api[_-]?key["\s]*[:=]["\s]*)[^"\s,}]+',
                r'\1"***"',
                fixed_line,
                flags=re.IGNORECASE
            )
        
        elif 'console.log' in issue.issue_type.lower():
            # Comment out problematic console.log statements
            if not line.strip().startswith('//'):
                fixed_line = line.replace(line.strip(), '// ' + line.strip() + ' // REMOVED: Sensitive data logging')
        
        elif 'connection string' in issue.issue_type.lower():
            # Redact credentials in connection strings
            fixed_line = re.sub(
                r'://([^:]+):([^@]+)@',
                r'://***:***@',
                fixed_line
            )
        
        return fixed_line
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive security report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Report dictionary
        """
        # Categorize issues by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
        
        # Generate summary
        summary = MigrationSummary(
            files_scanned=len(set(issue.file_path for issue in self.issues)),
            issues_found=len(self.issues),
            issues_fixed=0,  # Will be updated by fix application
            critical_issues=severity_counts['critical'],
            high_issues=severity_counts['high'],
            medium_issues=severity_counts['medium'],
            low_issues=severity_counts['low'],
            patterns_detected=self.patterns_detected,
            migration_time=datetime.now().isoformat()
        )
        
        # Create full report
        report = {
            'summary': asdict(summary),
            'issues': [asdict(issue) for issue in self.issues],
            'recommendations': self._generate_recommendations(),
            'secure_logging_status': self._check_secure_logging_status()
        }
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        if self.patterns_detected.get('Direct password logging detected', 0) > 0:
            recommendations.append(
                "CRITICAL: Remove all password logging immediately. "
                "Passwords should never be logged in any form."
            )
        
        if self.patterns_detected.get('API key logging detected', 0) > 0:
            recommendations.append(
                "CRITICAL: Remove API key logging. "
                "Use secure configuration management for API keys."
            )
        
        if self.patterns_detected.get('Token logging detected', 0) > 0:
            recommendations.append(
                "HIGH: Implement token truncation in logs. "
                "Log only the first 8 characters followed by '***'."
            )
        
        if len(self.issues) > 0:
            recommendations.append(
                "Enable secure logging filters system-wide to automatically "
                "detect and redact sensitive data in logs."
            )
        
        if any(issue.severity == 'critical' for issue in self.issues):
            recommendations.append(
                "Immediate action required: Critical security issues found. "
                "Review and fix all critical issues before deployment."
            )
        
        recommendations.append(
            "Implement regular security audits of logging practices. "
            "Train developers on secure logging guidelines."
        )
        
        return recommendations
    
    def _check_secure_logging_status(self) -> Dict[str, Any]:
        """Check current secure logging implementation status."""
        status = {
            'secure_sanitizer_available': False,
            'secure_filters_available': False,
            'secure_logging_enabled': False,
            'configuration_files': []
        }
        
        # Check if secure components exist
        secure_files = [
            'sensitive_data_sanitizer.py',
            'secure_logging_filters.py',
            'security/logging.py'
        ]
        
        for file_name in secure_files:
            file_path = self.project_root / file_name if self.project_root else Path(file_name)
            if file_path.exists():
                if 'sanitizer' in file_name:
                    status['secure_sanitizer_available'] = True
                elif 'filters' in file_name:
                    status['secure_filters_available'] = True
                status['configuration_files'].append(str(file_path))
        
        # Check if secure logging is enabled in main files
        main_files = ['web/app.py', 'log_utils.py', 'config.py']
        
        for file_name in main_files:
            file_path = self.project_root / file_name if self.project_root else Path(file_name)
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    if 'secure_logging' in content.lower() or 'sensitivedata' in content.lower():
                        status['secure_logging_enabled'] = True
                        break
                        
                except Exception:
                    pass
        
        return status
    
    def enable_secure_logging_systemwide(self, dry_run: bool = True) -> Dict[str, Any]:
        """Enable secure logging across the entire system.
        
        Args:
            dry_run: If True, only simulate changes
            
        Returns:
            Results of enabling secure logging
        """
        result = {
            'dry_run': dry_run,
            'files_modified': [],
            'changes_made': [],
            'errors': []
        }
        
        try:
            # Update main application files to use secure logging
            updates = [
                {
                    'file': 'web/app.py',
                    'pattern': r'from log_utils import setup_logging',
                    'replacement': 'from log_utils import setup_logging\nfrom secure_logging_filters import setup_secure_logging',
                    'description': 'Import secure logging in web app'
                },
                {
                    'file': 'config.py',
                    'pattern': r'from log_utils import setup_logging',
                    'replacement': 'from log_utils import setup_logging\nfrom secure_logging_filters import get_secure_logging_manager',
                    'description': 'Import secure logging manager in config'
                }
            ]
            
            for update in updates:
                file_path = self.project_root / update['file'] if self.project_root else Path(update['file'])
                
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        if update['pattern'] in content and update['replacement'] not in content:
                            new_content = content.replace(update['pattern'], update['replacement'])
                            
                            if not dry_run:
                                with open(file_path, 'w') as f:
                                    f.write(new_content)
                            
                            result['files_modified'].append(str(file_path))
                            result['changes_made'].append(update['description'])
                    
                    except Exception as e:
                        result['errors'].append(f"Error updating {file_path}: {str(e)}")
            
            return result
            
        except Exception as e:
            result['errors'].append(f"Error enabling secure logging: {str(e)}")
            return result
    
    def test_sanitization(self, test_string: str) -> Dict[str, Any]:
        """Test sanitization on a specific string.
        
        Args:
            test_string: String to test
            
        Returns:
            Test results
        """
        original = test_string
        sanitized = self.sanitizer.sanitize(test_string)
        
        return {
            'original': original,
            'sanitized': sanitized,
            'has_sensitive_data': original != sanitized,
            'redaction_applied': original != sanitized,
            'sanitizer_stats': self.sanitizer.get_statistics()
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="NightScan Secure Logging Migration")
    
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--scan', action='store_true',
                             help='Scan codebase for logging security issues')
    action_group.add_argument('--fix', action='store_true',
                             help='Apply automated fixes')
    action_group.add_argument('--report', 
                             help='Generate comprehensive report (specify output file)')
    action_group.add_argument('--test',
                             help='Test sanitization on a specific string')
    action_group.add_argument('--enable-secure-logging', action='store_true',
                             help='Enable secure logging system-wide')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate changes without modifying files')
    parser.add_argument('--directories', nargs='+',
                       help='Specific directories to scan')
    parser.add_argument('--project-root',
                       help='Project root directory')
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = SecureLoggingMigrator(args.project_root or str(project_root))
    
    try:
        if args.scan:
            print("🔍 Scanning codebase for logging security issues...")
            issues = migrator.scan_codebase(args.directories)
            
            print(f"\n📊 Scan Results:")
            print(f"  Issues found: {len(issues)}")
            
            severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            for issue in issues:
                severity_counts[issue.severity] += 1
            
            for severity, count in severity_counts.items():
                if count > 0:
                    emoji = "🚨" if severity == 'critical' else ("⚠️" if severity == 'high' else "ℹ️")
                    print(f"  {emoji} {severity.capitalize()}: {count}")
            
            # Show top issues
            if issues:
                print(f"\n🔍 Top Issues:")
                for issue in issues[:5]:
                    print(f"  {issue.severity.upper()}: {issue.description}")
                    print(f"    File: {issue.file_path}:{issue.line_number}")
                    print(f"    Line: {issue.line_content.strip()}")
                    if issue.suggested_fix:
                        print(f"    Fix: {issue.suggested_fix}")
                    print()
        
        elif args.fix:
            print("🔧 Applying automated fixes...")
            issues = migrator.scan_codebase(args.directories)
            
            if not issues:
                print("✅ No issues found to fix")
                return
            
            results = migrator.apply_automated_fixes(dry_run=args.dry_run)
            
            if args.dry_run:
                print("🧪 Dry run - no files were modified")
            
            print(f"\n📊 Fix Results:")
            print(f"  Fixes applied: {results['fixes_applied']}")
            print(f"  Files modified: {len(results['files_modified'])}")
            
            if results['errors']:
                print(f"  Errors: {len(results['errors'])}")
                for error in results['errors']:
                    print(f"    {error}")
        
        elif args.report:
            print("📋 Generating comprehensive security report...")
            issues = migrator.scan_codebase(args.directories)
            report = migrator.generate_report(args.report)
            
            print(f"✅ Report generated: {args.report}")
            print(f"  Total issues: {report['summary']['issues_found']}")
            print(f"  Critical issues: {report['summary']['critical_issues']}")
            print(f"  Recommendations: {len(report['recommendations'])}")
        
        elif args.test:
            print(f"🧪 Testing sanitization on: {args.test}")
            result = migrator.test_sanitization(args.test)
            
            print(f"\nOriginal: {result['original']}")
            print(f"Sanitized: {result['sanitized']}")
            print(f"Has sensitive data: {result['has_sensitive_data']}")
        
        elif args.enable_secure_logging:
            print("🔒 Enabling secure logging system-wide...")
            result = migrator.enable_secure_logging_systemwide(dry_run=args.dry_run)
            
            if args.dry_run:
                print("🧪 Dry run - no files were modified")
            
            print(f"\n📊 Results:")
            print(f"  Files modified: {len(result['files_modified'])}")
            print(f"  Changes made: {len(result['changes_made'])}")
            
            for change in result['changes_made']:
                print(f"    ✅ {change}")
            
            if result['errors']:
                print(f"  Errors: {len(result['errors'])}")
                for error in result['errors']:
                    print(f"    ❌ {error}")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()