#!/usr/bin/env python3
"""Run performance tests for the optimized analytics module."""

import subprocess
import sys
import os

def run_performance_tests():
    """Run the performance test suite."""
    print("=" * 60)
    print("NightScan Analytics Performance Tests")
    print("=" * 60)
    print()
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is not installed")
        print("Install it with: pip install pytest")
        return 1
    
    # Check if psutil is installed (for memory tests)
    try:
        import psutil
    except ImportError:
        print("WARNING: psutil is not installed")
        print("Memory usage tests will be skipped")
        print("Install it with: pip install psutil")
        print()
    
    # Run the tests
    print("Running performance tests...")
    print("-" * 60)
    
    test_file = os.path.join(os.path.dirname(__file__), 'tests', 'test_analytics_performance.py')
    
    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        test_file, 
        '-v',
        '--tb=short',
        '-s'  # Show print statements
    ])
    
    print("-" * 60)
    
    if result.returncode == 0:
        print("\n✅ All performance tests passed!")
        print("\nKey achievements:")
        print("- No N+1 queries in analytics")
        print("- Constant memory usage for CSV export")
        print("- Fast query response times")
        print("- Proper connection pool handling")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_performance_tests())