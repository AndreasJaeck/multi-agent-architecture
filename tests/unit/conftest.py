"""
Pytest configuration and shared fixtures for multi-agent architecture tests.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add src to path for imports during testing
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


# Mock ChatDatabricks at the session level to prevent import-time failures
# But ONLY for unit tests - integration tests should use real connections
@pytest.fixture(autouse=True, scope="session")
def mock_databricks_session(request):
    """Mock ChatDatabricks and related Databricks components for unit tests only."""
    # Skip mocking for integration tests
    # Use config to determine if this is an integration test run
    config = getattr(request, 'config', None)
    if config:
        test_args = ' '.join(getattr(config.invocation_params, 'args', []))
        if 'integration' in test_args.lower():
            yield None
            return
        
    with patch('databricks_langchain.ChatDatabricks') as mock_chat_db:
        with patch('databricks.sdk.WorkspaceClient') as mock_workspace_client:
            with patch('mlflow.langchain.autolog') as mock_mlflow_autolog:
                with patch('mlflow.models.set_model') as mock_set_model:
                    with patch('langgraph.prebuilt.create_react_agent') as mock_create_agent:
                        with patch('langchain_core.runnables.RunnableLambda') as mock_runnable_lambda:
                            # Configure ChatDatabricks mock to return MockChatDatabricks instances
                            mock_chat_db.return_value = MockChatDatabricks()
                            # Configure WorkspaceClient to not fail on init
                            mock_workspace_client.return_value = Mock()
                            # Mock create_react_agent to return a simple mock agent
                            mock_agent = Mock()
                            mock_agent.invoke = Mock(return_value={"messages": [{"content": "Mock agent response"}]})
                            mock_create_agent.return_value = mock_agent
                            
                            # Configure RunnableLambda to work with our chain mocking
                            def mock_runnable_lambda_constructor(func):
                                mock_lambda = Mock()
                                mock_lambda.__or__ = lambda self, other: MockRunnableChain(other)
                                return mock_lambda
                            mock_runnable_lambda.side_effect = mock_runnable_lambda_constructor
                            
                            # MLflow mocks don't need to do anything
                            mock_mlflow_autolog.return_value = None
                            mock_set_model.return_value = None
                            
                            yield {
                                'chat_databricks': mock_chat_db,
                                'workspace_client': mock_workspace_client,
                                'mlflow_autolog': mock_mlflow_autolog,
                                'set_model': mock_set_model,
                                'create_react_agent': mock_create_agent,
                                'runnable_lambda': mock_runnable_lambda
                            }


@pytest.fixture
def mock_llm():
    """Mock LLM for testing supervisor decisions without API calls."""
    llm = Mock()
    llm.invoke = Mock()
    llm.with_structured_output = Mock()
    return llm


@pytest.fixture
def mock_agent():
    """Mock agent for testing agent nodes without LLM calls."""
    agent = Mock()
    agent.invoke = Mock()
    return agent


@pytest.fixture
def sample_user_message():
    """Sample user message for testing."""
    return {
        "role": "user", 
        "content": "What is the molecular structure of water?"
    }


@pytest.fixture
def sample_agent_response():
    """Sample successful agent response."""
    return {
        "role": "assistant",
        "name": "BASF_Data", 
        "content": "Water has the molecular formula H2O, consisting of two hydrogen atoms bonded to one oxygen atom."
    }


@pytest.fixture
def sample_error_response():
    """Sample error response from failed agent."""
    return {
        "role": "assistant",
        "name": "BASF_Data",
        "content": "❌ **BASF Data Assistant Unavailable**\n\nI'm currently unable to access the service."
    }


@pytest.fixture
def sample_state():
    """Sample conversation state for testing."""
    return {
        "messages": [
            {"role": "user", "content": "Test query"},
        ],
        "iteration_count": 0,
    }


@pytest.fixture
def sample_multi_agent_state():
    """Sample state with multiple agent responses."""
    return {
        "messages": [
            {"role": "user", "content": "Complex query requiring multiple agents"},
            {"role": "assistant", "name": "BASF_Data", "content": "Response from BASF agent with chemical data"},
            {"role": "assistant", "name": "Genomics_Tools", "content": "Response from genomics agent with calculations"}
        ],
        "iteration_count": 2
    }


@pytest.fixture
def mock_databricks_client():
    """Mock Databricks client for testing."""
    client = Mock()
    client.serving_endpoints = Mock()
    return client


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing logging and model operations."""
    mlflow = Mock()
    mlflow.start_run = Mock()
    mlflow.log_param = Mock()
    mlflow.log_metric = Mock()
    mlflow.log_artifact = Mock()
    return mlflow


class MockRunnableChain:
    """Mock for RunnableLambda | ChatDatabricks chains."""
    
    def __init__(self, llm_mock):
        self.llm_mock = llm_mock
    
    def invoke(self, state):
        # For test purposes, we want the actual patched LLM to be called
        # so tests can verify the call was made
        if hasattr(self.llm_mock, 'invoke'):
            return self.llm_mock.invoke(state)
        return Mock(content="Mock chain response")


class MockChatDatabricks:
    """Mock ChatDatabricks for testing without API calls."""
    
    def __init__(self, endpoint=None, **kwargs):
        self.endpoint = endpoint
        self._response = None
    
    def invoke(self, messages):
        if self._response:
            return self._response
        # Return a proper message format that matches what the real ChatDatabricks returns
        return Mock(content="Mock LLM response", role="assistant")
    
    def with_structured_output(self, schema):
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value=Mock(
            next_node="FINISH", 
            reasoning="Mock reasoning"
        ))
        return mock_chain
    
    def set_response(self, response):
        """Set the response this mock should return."""
        self._response = response
    
    def stream(self, *args, **kwargs):
        """Mock streaming method."""
        return iter([])
    
    def __or__(self, other):
        """Support for | operator (pipe) used in RunnableLambda chains."""
        # Return a mock that supports invoke
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value=Mock(content="Mock chained response"))
        return mock_chain
    
    def __ror__(self, other):
        """Support for reverse | operator (pipe) when LLM is on the right side."""
        # This handles: preprocessor | llm
        def chained_invoke(input_data):
            if self._response:
                return self._response
            return Mock(content="Mock chained response")
        
        mock_chain = Mock()
        mock_chain.invoke = Mock(side_effect=chained_invoke)
        return mock_chain


@pytest.fixture
def mock_chat_databricks():
    """Mock ChatDatabricks fixture."""
    return MockChatDatabricks()


# Pytest marks for organizing tests
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.requires_llm = pytest.mark.requires_llm


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_llm: Tests requiring LLM API access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests that likely require LLM calls
        if any(keyword in item.name.lower() for keyword in ["llm", "openai", "claude", "gpt"]):
            item.add_marker(pytest.mark.requires_llm)
