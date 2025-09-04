#!/usr/bin/env python3
"""
Integration test runner for multi-agent supervisor architecture.
Provides convenient commands to run different types of integration tests.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent


def run_pytest_command(args_list, description):
    """Run pytest with the given arguments."""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    # Ensure we're in the project root
    project_root = get_project_root()
    os.chdir(project_root)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"] + args_list
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run integration tests for multi-agent supervisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  run-integration-tests --health           # Quick health check
  run-integration-tests --auth            # Authentication tests  
  run-integration-tests --basic           # Basic functionality
  run-integration-tests --supervisor      # Supervisor tests
  run-integration-tests --supervisor-health # Supervisor health check
  run-integration-tests --all             # All tests
  run-integration-tests --perf            # Performance tests
        """
    )
    
    # Test category options
    parser.add_argument("--health", action="store_true", 
                       help="Run health check tests only (quick)")
    parser.add_argument("--auth", action="store_true",
                       help="Run authentication tests only")
    parser.add_argument("--basic", action="store_true",
                       help="Run basic request/response tests")
    parser.add_argument("--connectivity", action="store_true",
                       help="Run connectivity tests")
    parser.add_argument("--errors", action="store_true",
                       help="Run error handling tests")
    parser.add_argument("--perf", action="store_true",
                       help="Run performance tests (slow)")
    parser.add_argument("--supervisor", action="store_true",
                       help="Run supervisor integration tests")
    parser.add_argument("--supervisor-health", action="store_true",
                       help="Run supervisor health check (quick)")
    parser.add_argument("--supervisor-perf", action="store_true",
                       help="Run supervisor performance tests (slow)")
    parser.add_argument("--all", action="store_true",
                       help="Run all integration tests")
    
    # Test execution options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--no-capture", "-s", action="store_true",
                       help="Don't capture output (show prints)")
    parser.add_argument("--profile", default="e2-demo-field-eng",
                       help="Databricks profile to use (default: e2-demo-field-eng)")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Test timeout in seconds (default: 300)")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["DATABRICKS_PROFILE"] = args.profile
    # Also set other possible env var names for compatibility
    os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile
    
    # Base pytest arguments
    base_args = [
        "tests/integration/",
        # Skip coverage for integration tests since they don't import multi_agent modules
        "--no-cov"
    ]
    
    if args.verbose:
        base_args.append("-v")
    
    if args.no_capture:
        base_args.append("-s")
    
    # Add timeout (if pytest-timeout is available)
    try:
        import pytest_timeout
        base_args.extend(["--timeout", str(args.timeout)])
    except ImportError:
        print("⚠️  pytest-timeout not installed, skipping timeout option")
    
    # Determine which tests to run
    success = True
    
    if args.health:
        test_args = [
            "tests/integration/test_databricks_connectivity.py::TestEndpointHealthCheck"
        ] + base_args[1:]  # Skip the directory from base_args
        success &= run_pytest_command(test_args, "Health Check Tests")
    
    elif args.auth:
        test_args = [
            "tests/integration/test_databricks_connectivity.py::TestDatabricksAuthentication"
        ] + base_args[1:]  # Skip the directory from base_args
        success &= run_pytest_command(test_args, "Authentication Tests")
    
    elif args.connectivity:
        test_args = [
            "tests/integration/test_databricks_connectivity.py::TestEndpointConnectivity"
        ] + base_args[1:]
        success &= run_pytest_command(test_args, "Connectivity Tests")
    
    elif args.basic:
        test_args = [
            "tests/integration/test_databricks_connectivity.py::TestEndpointRequests"
        ] + base_args[1:]
        success &= run_pytest_command(test_args, "Basic Request/Response Tests")
    
    elif args.errors:
        # Note: Error handling tests can be added later if needed
        print("Error handling tests not implemented yet")
        success = True
    
    elif args.perf:
        test_args = [
            "tests/integration/test_databricks_connectivity.py::TestEndpointPerformance"
        ] + base_args[1:]
        success &= run_pytest_command(test_args, "Performance Tests")
    
    elif args.supervisor:
        test_args = [
            "tests/integration/test_supervisor/test_supervisor_integration.py::TestSupervisorBasicFunctionality",
            "tests/integration/test_supervisor/test_supervisor_integration.py::TestSupervisorAdvancedScenarios", 
            "tests/integration/test_supervisor/test_supervisor_integration.py::TestSupervisorErrorHandling"
        ] + base_args[1:]
        success &= run_pytest_command(test_args, "Supervisor Integration Tests")
    
    elif args.supervisor_health:
        test_args = [
            "tests/integration/test_supervisor/test_supervisor_integration.py::TestSupervisorHealthCheck"
        ] + base_args[1:]
        success &= run_pytest_command(test_args, "Supervisor Health Check")
    
    elif args.supervisor_perf:
        test_args = [
            "tests/integration/test_supervisor/test_supervisor_integration.py::TestSupervisorPerformance"
        ] + base_args[1:]
        success &= run_pytest_command(test_args, "Supervisor Performance Tests")
    
    elif args.all:
        success &= run_pytest_command(base_args, "All Integration Tests")
    
    else:
        # Default: run health check
        print("No specific test category specified. Running health check...")
        test_args = [
            "tests/integration/test_databricks_connectivity.py::TestEndpointHealthCheck"
        ] + base_args[1:]
        success &= run_pytest_command(test_args, "Health Check Tests (Default)")
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed. Check output above for details.")
        
    print("\n💡 Tip: Use --verbose for more detailed output")
    print("💡 Tip: Use --health for quick status checks")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
