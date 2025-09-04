"""
Integration tests for Databricks endpoint connectivity and functionality.
Tests the actual endpoints with real authentication and requests.
"""

import pytest
import time
import json
from typing import Dict, Any

# Import integration test configuration
from .conftest import DatabricksTestConfig

# Test markers
integration_test = pytest.mark.integration
requires_databricks = pytest.mark.requires_databricks
slow_test = pytest.mark.slow


@integration_test
@requires_databricks
class TestDatabricksAuthentication:
    """Test Databricks authentication and basic connectivity."""
    
    def test_cli_authentication(self, databricks_manager):
        """Test that Databricks CLI authentication is working."""
        assert databricks_manager.check_authentication(), (
            "Databricks CLI authentication failed. Please run: "
            f"databricks configure --profile {databricks_manager.config.DEFAULT_PROFILE}"
        )
    
    def test_workspace_access(self, workspace_info):
        """Test that we can access workspace information."""
        assert workspace_info['url'] is not None, "Could not retrieve workspace URL"
        assert workspace_info['profile'] == 'e2-demo-field-eng', "Wrong profile configured"
        print(f"Connected to workspace: {workspace_info['url']}")
    
    def test_profile_configuration(self, integration_config):
        """Test that the profile is properly configured."""
        import subprocess
        
        try:
            result = subprocess.run(
                ['databricks', 'auth', 'profiles', '--output', 'json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            assert result.returncode == 0, "Failed to list Databricks profiles"
            
            data = json.loads(result.stdout)
            profiles = data.get('profiles', [])
            profile_names = [p.get('name') for p in profiles]
            
            assert integration_config.DEFAULT_PROFILE in profile_names, (
                f"Profile {integration_config.DEFAULT_PROFILE} not found. "
                f"Available profiles: {profile_names}"
            )
            
        except Exception as e:
            pytest.fail(f"Could not verify profile configuration: {e}")


@integration_test
@requires_databricks
class TestEndpointConnectivity:
    """Test connectivity to individual Databricks serving endpoints."""
    
    def test_basf_endpoint_connection(self, basf_client, integration_config):
        """Test connection to BASF data endpoint."""
        assert basf_client is not None, (
            f"Could not create client for {integration_config.BASF_DATA_ENDPOINT}"
        )
        print(f"Successfully connected to BASF endpoint: {integration_config.BASF_DATA_ENDPOINT}")
    
    def test_genomics_endpoint_connection(self, genomics_client, integration_config):
        """Test connection to Genomics tools endpoint."""
        assert genomics_client is not None, (
            f"Could not create client for {integration_config.GENOMICS_TOOLS_ENDPOINT}"
        )
        print(f"Successfully connected to Genomics endpoint: {integration_config.GENOMICS_TOOLS_ENDPOINT}")
    
    def test_llm_endpoint_connection(self, llm_client, integration_config):
        """Test connection to main LLM endpoint."""
        assert llm_client is not None, (
            f"Could not create client for {integration_config.LLM_ENDPOINT_NAME}"
        )
        print(f"Successfully connected to LLM endpoint: {integration_config.LLM_ENDPOINT_NAME}")


@integration_test
@requires_databricks
class TestEndpointRequests:
    """Test actual request/response functionality for each endpoint."""
    
    def test_basf_endpoint_request(self, basf_client, sample_test_queries):
        """Test a request to the BASF endpoint."""
        try:
            start_time = time.time()
            response = basf_client.invoke(sample_test_queries['basf'])
            end_time = time.time()
            
            # Verify response structure
            assert response is not None, "No response received from BASF endpoint"
            assert hasattr(response, 'content'), "Response missing content attribute"
            
            # Check response content - handle different formats
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, list):
                    # Multi-agent endpoints may return list of responses
                    content = content[0].get('content', '') if content and isinstance(content[0], dict) else str(content[0])
                elif not isinstance(content, str):
                    content = str(content)
            else:
                content = str(response)
            
            assert len(content) > 0, "Empty response from BASF endpoint"
            
            # Performance check
            response_time = end_time - start_time
            print(f"BASF endpoint response time: {response_time:.2f}s")
            print(f"BASF response preview: {content[:100]}...")
            
            # Reasonable response time (multi-agent endpoints can be slower)
            assert response_time < 180, f"BASF endpoint too slow: {response_time:.2f}s"
            
        except Exception as e:
            pytest.fail(f"BASF endpoint request failed: {str(e)}")
    
    def test_genomics_endpoint_request(self, genomics_client, sample_test_queries):
        """Test a request to the Genomics endpoint."""
        try:
            start_time = time.time()
            response = genomics_client.invoke(sample_test_queries['genomics'])
            end_time = time.time()
            
            # Verify response structure
            assert response is not None, "No response received from Genomics endpoint"
            assert hasattr(response, 'content'), "Response missing content attribute"
            
            # Check response content - handle different formats
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, list):
                    # Multi-agent endpoints may return list of responses
                    content = content[0].get('content', '') if content and isinstance(content[0], dict) else str(content[0])
                elif not isinstance(content, str):
                    content = str(content)
            else:
                content = str(response)
                
            assert len(content) > 0, "Empty response from Genomics endpoint"
            
            # Performance check
            response_time = end_time - start_time
            print(f"Genomics endpoint response time: {response_time:.2f}s")
            print(f"Genomics response preview: {content[:100]}...")
            
            # Reasonable response time (multi-agent endpoints can be slower)
            assert response_time < 120, f"Genomics endpoint too slow: {response_time:.2f}s"
            
        except Exception as e:
            pytest.fail(f"Genomics endpoint request failed: {str(e)}")
    
    def test_llm_endpoint_request(self, llm_client, sample_test_queries):
        """Test a request to the main LLM endpoint."""
        try:
            start_time = time.time()
            response = llm_client.invoke(sample_test_queries['general'])
            end_time = time.time()
            
            # Verify response structure
            assert response is not None, "No response received from LLM endpoint"
            assert hasattr(response, 'content'), "Response missing content attribute"
            
            # Check response content - handle different formats  
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, list):
                    # Multi-agent endpoints may return list of responses
                    content = content[0].get('content', '') if content and isinstance(content[0], dict) else str(content[0])
                elif not isinstance(content, str):
                    content = str(content)
            else:
                content = str(response)
                
            assert len(content) > 0, "Empty response from LLM endpoint"
            
            # Performance check
            response_time = end_time - start_time
            print(f"LLM endpoint response time: {response_time:.2f}s")
            print(f"LLM response preview: {content[:100]}...")
            
            # Reasonable response time
            assert response_time < 30, f"LLM endpoint too slow: {response_time:.2f}s"
            
        except Exception as e:
            pytest.fail(f"LLM endpoint request failed: {str(e)}")


@integration_test
@requires_databricks
class TestEndpointHealthCheck:
    """Health check tests for monitoring endpoint status."""
    
    def test_endpoint_health_summary(self, basf_client, genomics_client, llm_client, 
                                   integration_config, sample_test_queries):
        """Comprehensive health check for all endpoints."""
        endpoints_status = {}
        
        # Test each endpoint
        test_cases = [
            (basf_client, integration_config.BASF_DATA_ENDPOINT, "BASF", sample_test_queries['basf']),
            (genomics_client, integration_config.GENOMICS_TOOLS_ENDPOINT, "Genomics", sample_test_queries['genomics']),
            (llm_client, integration_config.LLM_ENDPOINT_NAME, "LLM", sample_test_queries['general'])
        ]
        
        for client, endpoint_name, display_name, test_message in test_cases:
            try:
                start_time = time.time()
                response = client.invoke(test_message)
                end_time = time.time()
                
                # Handle different response formats
                if hasattr(response, 'content'):
                    content = response.content
                    if isinstance(content, list):
                        # Multi-agent endpoints may return list of responses
                        content = content[0].get('content', '') if content and isinstance(content[0], dict) else str(content[0])
                    elif not isinstance(content, str):
                        content = str(content)
                else:
                    content = str(response)
                
                endpoints_status[display_name] = {
                    'status': 'healthy',
                    'response_time': end_time - start_time,
                    'endpoint': endpoint_name,
                    'has_content': len(content) > 0,
                    'content_length': len(content)
                }
                
            except Exception as e:
                endpoints_status[display_name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'endpoint': endpoint_name
                }
        
        # Print detailed status summary
        print("\n" + "=" * 60)
        print("🏥 Endpoint Health Check Summary")
        print("=" * 60)
        
        healthy_endpoints = []
        unhealthy_endpoints = []
        
        for name, status in endpoints_status.items():
            if status['status'] == 'healthy':
                print(f"✅ {name:12} | {status['response_time']:5.2f}s | {status['content_length']:4d} chars | {status['endpoint']}")
                healthy_endpoints.append(name)
            else:
                print(f"❌ {name:12} | ERROR  | {status['endpoint']}")
                print(f"   └─ {status['error'][:80]}...")
                unhealthy_endpoints.append(name)
        
        print("-" * 60)
        print(f"📊 Status: {len(healthy_endpoints)}/{len(test_cases)} endpoints healthy")
        
        if len(healthy_endpoints) == len(test_cases):
            print("🎉 All endpoints are working perfectly!")
        elif len(healthy_endpoints) > 0:
            print(f"⚠️  {len(unhealthy_endpoints)} endpoint(s) need attention")
        else:
            print("🚨 All endpoints are down - check configuration!")
        
        print("=" * 60)
        
        # Test passes if at least one endpoint is working
        assert len(healthy_endpoints) > 0, f"No endpoints are healthy: {unhealthy_endpoints}"


@integration_test
@requires_databricks
@slow_test
class TestEndpointPerformance:
    """Performance tests for endpoint response times and reliability."""
    
    def test_response_consistency(self, basf_client, sample_test_queries):
        """Test response time consistency across multiple requests."""
        response_times = []
        num_requests = 3
        
        print(f"\n🔄 Testing response consistency with {num_requests} requests...")
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = basf_client.invoke(sample_test_queries['basf'])
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                print(f"Request {i+1}: {response_time:.2f}s")
                
                # Brief pause between requests
                time.sleep(1)
                
            except Exception as e:
                pytest.fail(f"Request {i+1} failed: {str(e)}")
        
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"📈 Performance: Avg={avg_time:.2f}s, Min={min_time:.2f}s, Max={max_time:.2f}s")
        
        # Performance assertions
        assert avg_time < 30, f"Average response time too high: {avg_time:.2f}s"
        assert max_time < 60, f"Maximum response time too high: {max_time:.2f}s"
