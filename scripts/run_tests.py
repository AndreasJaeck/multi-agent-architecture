#!/usr/bin/env python3
"""
Enhanced test runner for multi-agent architecture.
Supports running different test categories with proper project structure.
"""

import sys
import os
import unittest
import subprocess
import argparse
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def setup_python_path():
    """Add src directory to Python path for imports."""
    project_root = get_project_root()
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def run_pytest(test_path=None, markers=None, coverage=True, verbose=True):
    """Run tests using pytest with specified options."""
    project_root = get_project_root()
    
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    if test_path:
        cmd.append(str(project_root / test_path))
    else:
        cmd.append(str(project_root / "tests"))
    
    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=multi_agent",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
        ])
    
    # Verbosity
    if verbose:
        cmd.append("-v")
    
    # Change to project root for execution
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    finally:
        os.chdir(original_cwd)


def run_unittest(test_path=None):
    """Run tests using unittest discovery (fallback method)."""
    setup_python_path()
    
    project_root = get_project_root()
    
    if test_path:
        # Run specific test path
        loader = unittest.TestLoader()
        if test_path.endswith('.py'):
            # Load specific test file
            module_name = test_path.replace('/', '.').replace('.py', '')
            suite = loader.loadTestsFromName(module_name)
        else:
            # Load from directory
            suite = loader.discover(str(project_root / test_path), pattern='test_*.py')
    else:
        # Discover all tests
        loader = unittest.TestLoader()
        suite = loader.discover(str(project_root / "tests"), pattern='test_*.py')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print(f"TESTS RUN: {result.testsRun}")
    print(f"FAILURES: {len(result.failures)}")
    print(f"ERRORS: {len(result.errors)}")
    print(f"SKIPPED: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL: {'✅ PASSED' if success else '❌ FAILED'}")
    print("="*50)
    
    return success


def main():
    """Main test runner with command line options."""
    parser = argparse.ArgumentParser(description="Run tests for multi-agent architecture")
    parser.add_argument(
        "--path", 
        help="Specific test path to run (e.g., 'unit', 'integration', 'unit/test_supervisor')"
    )
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run only integration tests"
    )
    parser.add_argument(
        "--no-cov", 
        action="store_true", 
        help="Disable coverage reporting"
    )
    parser.add_argument(
        "--pytest", 
        action="store_true", 
        help="Force use of pytest (default: auto-detect)"
    )
    parser.add_argument(
        "--unittest", 
        action="store_true", 
        help="Force use of unittest (fallback)"
    )
    
    args = parser.parse_args()
    
    # Determine test path
    test_path = None
    if args.path:
        test_path = f"tests/{args.path}"
    elif args.unit:
        test_path = "tests/unit"
    elif args.integration:
        test_path = "tests/integration"
    
    # Determine markers for pytest
    markers = []
    if args.unit:
        markers.append("unit")
    elif args.integration:
        markers.append("integration")
    
    # Choose test runner
    use_pytest = args.pytest or (not args.unittest and _has_pytest())
    
    print("🧪 Multi-Agent Architecture Test Runner")
    print("="*50)
    print(f"Test Runner: {'pytest' if use_pytest else 'unittest'}")
    print(f"Test Path: {test_path or 'all tests'}")
    print(f"Markers: {', '.join(markers) if markers else 'none'}")
    enable_coverage_display = not args.no_cov and use_pytest and not args.integration
    print(f"Coverage: {'enabled' if enable_coverage_display else 'disabled' + (' (integration tests)' if args.integration and use_pytest and not args.no_cov else '')}")
    print("="*50)
    
    if use_pytest:
        # Only enable coverage for unit tests, not integration tests
        enable_coverage = not args.no_cov and not args.integration
        success = run_pytest(
            test_path=test_path, 
            markers=markers, 
            coverage=enable_coverage
        )
    else:
        success = run_unittest(test_path=test_path)
    
    sys.exit(0 if success else 1)


def _has_pytest():
    """Check if pytest is available."""
    try:
        import pytest
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()