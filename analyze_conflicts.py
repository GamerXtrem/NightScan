#!/usr/bin/env python3
"""
NightScan Repository Conflict Analysis Tool
Automatically detects duplications, conflicts, and inconsistencies in the codebase.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import subprocess

class ConflictAnalyzer:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path)
        self.issues = defaultdict(list)
        
    def analyze_all(self):
        """Run all conflict analysis checks."""
        print("🔍 Analyzing NightScan repository for conflicts...")
        
        self.check_duplicate_files()
        self.check_port_conflicts()
        self.check_api_endpoint_conflicts()
        self.check_dependency_conflicts()
        self.check_import_conflicts()
        self.check_naming_conflicts()
        self.check_configuration_conflicts()
        
        self.generate_report()
        
    def check_duplicate_files(self):
        """Find duplicate files by content hash."""
        print("  📁 Checking for duplicate files...")
        
        file_hashes = defaultdict(list)
        
        for file_path in self.root_path.rglob("*.py"):
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    if len(content) > 100:  # Skip very small files
                        file_hash = hashlib.md5(content).hexdigest()
                        file_hashes[file_hash].append(file_path)
            except Exception:
                continue
                
        # Find duplicates
        for file_hash, files in file_hashes.items():
            if len(files) > 1:
                self.issues['duplicate_files'].append({
                    'type': 'Duplicate Files',
                    'priority': 'HIGH',
                    'files': [str(f) for f in files],
                    'description': f'Files with identical content found'
                })
                
        # Check for similar function names across different files
        self.check_similar_functions()
        
    def check_similar_functions(self):
        """Find similar function names that might indicate duplication."""
        function_names = defaultdict(list)
        
        for file_path in self.root_path.rglob("*.py"):
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find function definitions
                functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
                for func in functions:
                    if not func.startswith('_'):  # Skip private functions
                        function_names[func].append(str(file_path))
            except Exception:
                continue
                
        # Check for similar training/prediction functions
        training_funcs = [name for name in function_names.keys() if 'train' in name.lower()]
        predict_funcs = [name for name in function_names.keys() if 'predict' in name.lower()]
        
        if training_funcs:
            for func in training_funcs:
                if len(function_names[func]) > 1:
                    self.issues['duplicate_functions'].append({
                        'type': 'Duplicate Training Functions',
                        'priority': 'HIGH',
                        'function': func,
                        'files': function_names[func],
                        'description': f'Training function "{func}" found in multiple files'
                    })
                    
        if predict_funcs:
            for func in predict_funcs:
                if len(function_names[func]) > 1:
                    self.issues['duplicate_functions'].append({
                        'type': 'Duplicate Prediction Functions',
                        'priority': 'HIGH',
                        'function': func,
                        'files': function_names[func],
                        'description': f'Prediction function "{func}" found in multiple files'
                    })
                    
    def check_port_conflicts(self):
        """Check for hardcoded port conflicts."""
        print("  🔌 Checking for port conflicts...")
        
        port_usage = defaultdict(list)
        
        for file_path in self.root_path.rglob("*"):
            if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Look for port assignments
                    port_patterns = [
                        r'port[=:]\s*(\d{4,5})',
                        r'PORT[=:]\s*(\d{4,5})',
                        r':(\d{4,5})',  # URL patterns
                        r'\.run\([^)]*port\s*=\s*(\d{4,5})',
                    ]
                    
                    for pattern in port_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for port in matches:
                            if 1000 <= int(port) <= 65535:  # Valid port range
                                port_usage[port].append(str(file_path))
                                
                except Exception:
                    continue
                    
        # Find conflicts
        for port, files in port_usage.items():
            if len(files) > 1:
                self.issues['port_conflicts'].append({
                    'type': 'Port Conflict',
                    'priority': 'MEDIUM',
                    'port': port,
                    'files': files,
                    'description': f'Port {port} used in multiple files'
                })
                
    def check_api_endpoint_conflicts(self):
        """Check for API endpoint conflicts."""
        print("  🌐 Checking for API endpoint conflicts...")
        
        endpoints = defaultdict(list)
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for Flask routes
                route_patterns = [
                    r'@app\.route\(["\']([^"\']+)["\']',
                    r'@api[^.]*\.route\(["\']([^"\']+)["\']',
                    r'@blueprint\.route\(["\']([^"\']+)["\']',
                ]
                
                for pattern in route_patterns:
                    matches = re.findall(pattern, content)
                    for endpoint in matches:
                        endpoints[endpoint].append(str(file_path))
                        
            except Exception:
                continue
                
        # Find conflicts
        for endpoint, files in endpoints.items():
            if len(files) > 1:
                self.issues['api_conflicts'].append({
                    'type': 'API Endpoint Conflict',
                    'priority': 'MEDIUM',
                    'endpoint': endpoint,
                    'files': files,
                    'description': f'Endpoint "{endpoint}" defined in multiple files'
                })
                
    def check_dependency_conflicts(self):
        """Check for dependency version conflicts."""
        print("  📦 Checking for dependency conflicts...")
        
        requirement_files = [
            'requirements.txt',
            'requirements-ci.txt', 
            'pyproject.toml'
        ]
        
        dependencies = {}
        
        for req_file in requirement_files:
            file_path = self.root_path / req_file
            if file_path.exists():
                deps = self.parse_requirements(file_path)
                dependencies[req_file] = deps
                
        # Check for version conflicts
        all_packages = set()
        for deps in dependencies.values():
            all_packages.update(deps.keys())
            
        for package in all_packages:
            versions = {}
            for file_name, deps in dependencies.items():
                if package in deps:
                    versions[file_name] = deps[package]
                    
            if len(set(versions.values())) > 1:
                self.issues['dependency_conflicts'].append({
                    'type': 'Dependency Version Conflict',
                    'priority': 'HIGH',
                    'package': package,
                    'versions': versions,
                    'description': f'Package "{package}" has conflicting versions'
                })
                
    def parse_requirements(self, file_path):
        """Parse requirements from different file formats."""
        deps = {}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            if file_path.name == 'pyproject.toml':
                # Simple TOML parsing for dependencies
                dep_section = re.search(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if dep_section:
                    deps_text = dep_section.group(1)
                    dep_lines = re.findall(r'"([^"]+)"', deps_text)
                    for line in dep_lines:
                        if '>=' in line:
                            package, version = line.split('>=', 1)
                            deps[package.strip()] = f'>={version.strip()}'
                        elif '==' in line:
                            package, version = line.split('==', 1)
                            deps[package.strip()] = f'=={version.strip()}'
            else:
                # Parse requirements.txt format
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '>=' in line:
                            package, version = line.split('>=', 1)
                            deps[package.strip()] = f'>={version.strip()}'
                        elif '==' in line:
                            package, version = line.split('==', 1)
                            deps[package.strip()] = f'=={version.strip()}'
                        else:
                            deps[line] = 'any'
                            
        except Exception as e:
            print(f"    ⚠️  Error parsing {file_path}: {e}")
            
        return deps
        
    def check_import_conflicts(self):
        """Check for potential import conflicts."""
        print("  📥 Checking for import conflicts...")
        
        # Check for circular import risks
        imports = defaultdict(set)
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find imports
                import_patterns = [
                    r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                    r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
                ]
                
                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    for module in matches:
                        if not module.startswith('.') and '.' not in module:
                            imports[str(file_path)].add(module)
                            
            except Exception:
                continue
                
        # Simple circular import detection
        for file_path, file_imports in imports.items():
            file_name = Path(file_path).stem
            for imported_module in file_imports:
                if imported_module in [Path(f).stem for f in imports.keys()]:
                    reverse_imports = imports.get(f"**/{imported_module}.py", set())
                    if file_name in reverse_imports:
                        self.issues['import_conflicts'].append({
                            'type': 'Potential Circular Import',
                            'priority': 'MEDIUM',
                            'files': [file_path, f"{imported_module}.py"],
                            'description': f'Potential circular import between {file_name} and {imported_module}'
                        })
                        
    def check_naming_conflicts(self):
        """Check for naming conflicts."""
        print("  🏷️  Checking for naming conflicts...")
        
        class_names = defaultdict(list)
        
        for file_path in self.root_path.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find class definitions
                classes = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
                for class_name in classes:
                    class_names[class_name].append(str(file_path))
                    
            except Exception:
                continue
                
        # Check for duplicate class names
        for class_name, files in class_names.items():
            if len(files) > 1:
                # Filter out common base classes
                if class_name not in ['TestCase', 'Config', 'BaseModel']:
                    self.issues['naming_conflicts'].append({
                        'type': 'Class Name Conflict',
                        'priority': 'LOW',
                        'class_name': class_name,
                        'files': files,
                        'description': f'Class "{class_name}" defined in multiple files'
                    })
                    
    def check_configuration_conflicts(self):
        """Check for configuration conflicts."""
        print("  ⚙️  Checking for configuration conflicts...")
        
        config_files = [
            'config.py',
            'docker-compose.yml',
            'Dockerfile',
            '.env',
            '.env.example'
        ]
        
        configs = {}
        for config_file in config_files:
            file_path = self.root_path / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        configs[config_file] = f.read()
                except Exception:
                    continue
                    
        # Check for Python version conflicts
        python_versions = {}
        for file_name, content in configs.items():
            version_patterns = [
                r'python:(\d+\.\d+)',
                r'python_requires=[\'"](>=?\d+\.\d+)',
                r'python-version:\s*[\'"]([\d.]+)',
            ]
            
            for pattern in version_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    python_versions[file_name] = matches[0]
                    
        if len(set(python_versions.values())) > 1:
            self.issues['config_conflicts'].append({
                'type': 'Python Version Conflict',
                'priority': 'HIGH',
                'versions': python_versions,
                'description': 'Inconsistent Python versions across configuration files'
            })
            
    def generate_report(self):
        """Generate a comprehensive conflict report."""
        print("\n" + "="*60)
        print("🔍 NIGHTSCAN REPOSITORY CONFLICT ANALYSIS REPORT")
        print("="*60)
        
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        if total_issues == 0:
            print("✅ No conflicts detected! Repository is clean.")
            return
            
        print(f"\n📊 SUMMARY: {total_issues} potential issues found")
        
        # Count by priority
        priority_counts = Counter()
        for category_issues in self.issues.values():
            for issue in category_issues:
                priority_counts[issue['priority']] += 1
                
        print(f"   🔴 HIGH priority: {priority_counts['HIGH']}")
        print(f"   🟡 MEDIUM priority: {priority_counts['MEDIUM']}")
        print(f"   🟢 LOW priority: {priority_counts['LOW']}")
        
        # Detailed issues by category
        for category, category_issues in self.issues.items():
            if category_issues:
                print(f"\n🔍 {category.upper().replace('_', ' ')} ({len(category_issues)} issues)")
                print("-" * 40)
                
                for i, issue in enumerate(category_issues, 1):
                    priority_emoji = {
                        'HIGH': '🔴',
                        'MEDIUM': '🟡', 
                        'LOW': '🟢'
                    }[issue['priority']]
                    
                    print(f"{i}. {priority_emoji} {issue['type']}")
                    print(f"   Description: {issue['description']}")
                    
                    if 'files' in issue:
                        print(f"   Files: {', '.join(issue['files'][:3])}")
                        if len(issue['files']) > 3:
                            print(f"          ... and {len(issue['files'])-3} more")
                    
                    if 'endpoint' in issue:
                        print(f"   Endpoint: {issue['endpoint']}")
                    if 'port' in issue:
                        print(f"   Port: {issue['port']}")
                    if 'package' in issue:
                        print(f"   Package: {issue['package']}")
                        print(f"   Versions: {issue['versions']}")
                    
                    print()
                    
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 20)
        
        high_priority = sum(len(issues) for issues in self.issues.values() 
                          if any(issue['priority'] == 'HIGH' for issue in issues))
        
        if high_priority > 0:
            print("🔴 HIGH PRIORITY ACTIONS:")
            print("   1. Resolve dependency version conflicts immediately")
            print("   2. Eliminate duplicate training/prediction code")
            print("   3. Standardize Python versions across all configs")
            print()
            
        if self.issues.get('port_conflicts'):
            print("🟡 MEDIUM PRIORITY ACTIONS:")
            print("   1. Make all ports configurable via environment variables")
            print("   2. Resolve API endpoint conflicts")
            print("   3. Review import dependencies for circular references")
            print()
            
        print("📋 NEXT STEPS:")
        print("   1. Prioritize HIGH priority issues for immediate resolution")
        print("   2. Create GitHub issues for tracking conflict resolution")
        print("   3. Implement automated conflict detection in CI/CD")
        print("   4. Establish coding standards to prevent future conflicts")
        
        # Save detailed report
        report_data = {
            'summary': {
                'total_issues': total_issues,
                'priority_breakdown': dict(priority_counts),
                'categories': list(self.issues.keys())
            },
            'issues': dict(self.issues)
        }
        
        with open('conflict_analysis_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\n💾 Detailed report saved to: conflict_analysis_report.json")


if __name__ == "__main__":
    analyzer = ConflictAnalyzer()
    analyzer.analyze_all()