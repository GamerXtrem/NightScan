#!/usr/bin/env python3
"""Script to fix naming conventions in NightScan codebase"""

import os
import re
import sys
from pathlib import Path

# Mapping of old names to new names
REPLACEMENTS = {
    # Directory imports
    'audio_training_efficientnet': 'audio_training_efficientnet',
    'audio_training': 'audio_training',
    'picture_training_enhanced': 'picture_training_enhanced',
    'picture_training': 'picture_training',
}

def fix_python_imports(file_path):
    """Fix imports in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace imports
        for old_name, new_name in REPLACEMENTS.items():
            # Handle various import patterns
            patterns = [
                f'from {old_name}',
                f'import {old_name}',
                f'"{old_name}',
                f"'{old_name}",
            ]
            
            for pattern in patterns:
                new_pattern = pattern.replace(old_name, new_name)
                content = content.replace(pattern, new_pattern)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed imports in: {file_path}")
            return True
        
        return False
    
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def fix_javascript_naming(file_path):
    """Fix snake_case variables in JavaScript files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Common snake_case to camelCase conversions
        js_replacements = {
            'event_type': 'eventType',
            'user_id': 'userId',
            'file_path': 'filePath',
            'error_message': 'errorMessage',
            'is_online': 'isOnline',
            'sync_queue': 'syncQueue',
            'last_sync': 'lastSync',
            'cache_key': 'cacheKey',
            'refresh_token': 'refreshToken',
            'access_token': 'accessToken',
        }
        
        for old_name, new_name in js_replacements.items():
            # Only replace in contexts where it's likely a variable
            # Avoid replacing in strings
            pattern = rf'\b{old_name}\b(?!["\'])'
            content = re.sub(pattern, new_name, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed naming in: {file_path}")
            return True
        
        return False
    
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix naming conventions"""
    print("🔧 Fixing naming conventions in NightScan codebase...\n")
    
    # Fix Python imports
    print("📦 Fixing Python imports...")
    python_files_fixed = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['__pycache__', '.git', 'node_modules', 'venv']):
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_python_imports(file_path):
                    python_files_fixed += 1
    
    print(f"\n✅ Fixed {python_files_fixed} Python files")
    
    # Fix JavaScript naming
    print("\n📦 Fixing JavaScript naming conventions...")
    js_files_fixed = 0
    
    js_dirs = ['ios-app', 'web/static']
    for js_dir in js_dirs:
        if not os.path.exists(js_dir):
            continue
            
        for root, dirs, files in os.walk(js_dir):
            # Skip node_modules
            if 'node_modules' in root:
                continue
            
            for file in files:
                if file.endswith(('.js', '.jsx')):
                    file_path = os.path.join(root, file)
                    if fix_javascript_naming(file_path):
                        js_files_fixed += 1
    
    print(f"\n✅ Fixed {js_files_fixed} JavaScript files")
    
    print("\n🎉 Naming convention fixes complete!")
    print("\n⚠️  Please run tests to ensure everything still works correctly.")

if __name__ == '__main__':
    main()