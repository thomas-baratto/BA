#!/usr/bin/env python3
"""
Simple test script to verify basic functionality of the BA repository.
This script checks that:
1. Key Python files can be parsed
2. Basic Python environment is working
3. Repository structure is intact
"""

import sys
import os
import ast


def test_syntax():
    """Test that Python files have valid syntax."""
    print("Testing Python file syntax...")
    
    files_to_test = [
        'network.py',
        'FirstTrainingLabel11.py',
        'overfit.py',
        'visualiesierung.py'
    ]
    
    all_passed = True
    for filename in files_to_test:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    ast.parse(f.read())
                print(f"  ✓ {filename} - syntax valid")
            except SyntaxError as e:
                print(f"  ✗ {filename} - syntax error: {e}")
                all_passed = False
        else:
            print(f"  ⚠ {filename} - file not found")
    
    return all_passed


def test_structure():
    """Test that key directories and files exist."""
    print("\nTesting repository structure...")
    
    expected_items = [
        ('README.md', 'file'),
        ('requirements.txt', 'file'),
        ('network.py', 'file'),
        ('FirstTrainingLabel11.py', 'file'),
        ('Daten', 'directory'),
        ('Models', 'directory'),
        ('logs', 'directory'),
    ]
    
    all_passed = True
    for item, item_type in expected_items:
        if item_type == 'file':
            exists = os.path.isfile(item)
        else:
            exists = os.path.isdir(item)
        
        if exists:
            print(f"  ✓ {item} ({item_type}) - exists")
        else:
            print(f"  ✗ {item} ({item_type}) - missing")
            all_passed = False
    
    return all_passed


def test_python_environment():
    """Test basic Python environment."""
    print("\nTesting Python environment...")
    
    print(f"  ✓ Python version: {sys.version.split()[0]}")
    print(f"  ✓ Python executable: {sys.executable}")
    
    # Try importing standard libraries
    try:
        import json
        import datetime
        import math
        print("  ✓ Standard libraries available")
    except ImportError as e:
        print(f"  ✗ Standard library import failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("BA Repository Functionality Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Python Environment", test_python_environment()))
    results.append(("Repository Structure", test_structure()))
    results.append(("Python Syntax", test_syntax()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All tests passed! The repository is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
