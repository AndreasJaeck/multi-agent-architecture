"""
Integration test configuration and fixtures for multi-agent supervisor tests.
These fixtures provide real Databricks connectivity for integration testing.

This conftest.py is specifically for integration tests and does NOT use mocks.
It connects to real Databricks endpoints for authentic testing.
"""

import os
import sys
import pytest
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports during testing
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

class DatabricksTestConfig:
    """Configuration for Databricks integration tests."""
    
    # Default endpoints from your supervisor config
    BASF_DATA_ENDPOINT = "genie_multi_agent_basf"
    GENOMICS_TOOLS_ENDPOINT = "genie_multi_agent_basf_v2"
    LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
    
    # Test configuration
    DEFAULT_PROFILE = "e2-demo-field-eng"
    REQUEST_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    
    @classmethod
    def from_env(cls) -> 'DatabricksTestConfig':
        """Create config from environment variables."""
        config = cls()
        config.BASF_DATA_ENDPOINT = os.getenv('BASF_DATA_ENDPOINT', cls.BASF_DATA_ENDPOINT)
        config.GENOMICS_TOOLS_ENDPOINT = os.getenv('GENOMICS_TOOLS_ENDPOINT', cls.GENOMICS_TOOLS_ENDPOINT)
        config.LLM_ENDPOINT_NAME = os.getenv('LLM_ENDPOINT_NAME', cls.LLM_ENDPOINT_NAME)
        config.DEFAULT_PROFILE = os.getenv('DATABRICKS_PROFILE', cls.DEFAULT_PROFILE)
        return config


class DatabricksConnectionManager:
    """Manages Databricks connections for integration tests."""
    
    def __init__(self, config: DatabricksTestConfig):
        self.config = config
        self._authenticated = None
        self._workspace_url = None
    
    def check_authentication(self) -> bool:
        """Check if Databricks CLI authentication is working."""
        if self._authenticated is not None:
            return self._authenticated
            
        try:
            # Check if databricks CLI is available and configured
            result = subprocess.run(
                ['databricks', 'auth', 'profiles', '--output', 'json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                profiles = data.get('profiles', [])
                self._authenticated = any(
                    profile.get('name') == self.config.DEFAULT_PROFILE 
                    for profile in profiles
                )
            else:
                self._authenticated = False
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            self._authenticated = False
            
        return self._authenticated
    
    def get_workspace_url(self) -> Optional[str]:
        """Get workspace URL for the configured profile."""
        if self._workspace_url is not None:
            return self._workspace_url
            
        try:
            result = subprocess.run(
                ['databricks', 'auth', 'profiles', '--output', 'json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                profiles = data.get('profiles', [])
                for profile in profiles:
                    if profile.get('name') == self.config.DEFAULT_PROFILE:
                        self._workspace_url = profile.get('host')
                        break
                        
        except Exception:
            pass
            
        return self._workspace_url
    
    def create_chat_client(self, endpoint: str):
        """Create ChatDatabricks client for testing."""
        if not self.check_authentication():
            return None
            
        try:
            # Set profile environment variable if not already set
            if 'DATABRICKS_CONFIG_PROFILE' not in os.environ:
                os.environ['DATABRICKS_CONFIG_PROFILE'] = self.config.DEFAULT_PROFILE
            
            # Import here to get the real implementation
            from databricks_langchain import ChatDatabricks
            return ChatDatabricks(endpoint=endpoint)
        except Exception as e:
            print(f"[DEBUG] Error creating ChatDatabricks client: {e}")
            return None


# ================================
# INTEGRATION TEST FIXTURES
# ================================

@pytest.fixture(scope="session")
def integration_config():
    """Integration test configuration."""
    return DatabricksTestConfig.from_env()


@pytest.fixture(scope="session")
def databricks_manager(integration_config):
    """Databricks connection manager."""
    return DatabricksConnectionManager(integration_config)


@pytest.fixture(scope="session")
def databricks_available(databricks_manager):
    """Check if Databricks is available for testing."""
    available = databricks_manager.check_authentication()
    if not available:
        pytest.skip("Databricks CLI not authenticated or profile not found")
    return available


@pytest.fixture(scope="session")
def workspace_info(databricks_manager, databricks_available):
    """Get workspace information."""
    return {
        'url': databricks_manager.get_workspace_url(),
        'profile': databricks_manager.config.DEFAULT_PROFILE
    }


# Endpoint client fixtures
@pytest.fixture
def basf_client(databricks_manager, databricks_available, integration_config):
    """ChatDatabricks client for BASF endpoint."""
    client = databricks_manager.create_chat_client(integration_config.BASF_DATA_ENDPOINT)
    if client is None:
        pytest.skip(f"Could not create client for {integration_config.BASF_DATA_ENDPOINT}")
    return client


@pytest.fixture
def genomics_client(databricks_manager, databricks_available, integration_config):
    """ChatDatabricks client for Genomics endpoint."""
    client = databricks_manager.create_chat_client(integration_config.GENOMICS_TOOLS_ENDPOINT)
    if client is None:
        pytest.skip(f"Could not create client for {integration_config.GENOMICS_TOOLS_ENDPOINT}")
    return client


@pytest.fixture
def llm_client(databricks_manager, databricks_available, integration_config):
    """ChatDatabricks client for main LLM endpoint."""
    client = databricks_manager.create_chat_client(integration_config.LLM_ENDPOINT_NAME)
    if client is None:
        pytest.skip(f"Could not create client for {integration_config.LLM_ENDPOINT_NAME}")
    return client


# Test data fixtures
@pytest.fixture
def sample_test_queries():
    """Sample queries for testing different endpoints."""
    return {
        'basf': [{"role": "user", "content": "What is the chemical composition of polyethylene?"}],
        'genomics': [{"role": "user", "content": "Calculate the GC content of this DNA sequence: ATCGATCGATCG"}],
        'general': [{"role": "user", "content": "Hello, can you help me with a question?"}]
    }


# ================================
# PYTEST CONFIGURATION
# ================================

def pytest_configure(config):
    """Configure integration test markers."""
    config.addinivalue_line("markers", "requires_databricks: Tests requiring Databricks connectivity")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")