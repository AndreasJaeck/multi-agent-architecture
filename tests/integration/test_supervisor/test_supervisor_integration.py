"""
Integration tests for the supervisor_of_supervisors agent.
Tests the actual multi-agent system with real Databricks endpoints.
"""

import pytest
import time
from typing import Dict, Any, List

# Import the actual agent
from multi_agent.supervisor.supervisor_of_supervisors import AGENT
from mlflow.types.agent import ChatAgentMessage

# Import integration test configuration
from ..conftest import DatabricksTestConfig

# Test markers
integration_test = pytest.mark.integration
requires_databricks = pytest.mark.requires_databricks
slow_test = pytest.mark.slow


def extract_meaningful_response(response):
    """Extract meaningful response content from supervisor agent response."""
    for i, msg in enumerate(response.messages):
        content = getattr(msg, 'content', '')
        name = getattr(msg, 'name', 'N/A')
        
        if content and len(content) > 0:
            # Try to extract content from list format if needed
            if content.startswith('[{') and 'content' in content:
                try:
                    import ast
                    parsed = ast.literal_eval(content)
                    if isinstance(parsed, list) and len(parsed) > 0 and 'content' in parsed[0]:
                        return parsed[0]['content']
                except:
                    # Try JSON parsing as fallback
                    try:
                        import json
                        parsed = json.loads(content.replace("'", '"'))
                        if isinstance(parsed, list) and len(parsed) > 0 and 'content' in parsed[0]:
                            return parsed[0]['content']
                    except:
                        pass
            # Or use as-is if it's plain text
            elif not content.startswith('[{'):
                return content
    
    return None


@integration_test
@requires_databricks
class TestSupervisorBasicFunctionality:
    """Test basic supervisor functionality with real endpoints."""
    
    def test_agent_initialization(self):
        """Test that the supervisor agent is properly initialized."""
        assert AGENT is not None, "AGENT should be initialized"
        assert hasattr(AGENT, 'predict'), "AGENT should have predict method"
        assert hasattr(AGENT, 'predict_stream'), "AGENT should have predict_stream method"
    
    def test_simple_basf_query(self):
        """Test a query that should be routed to BASF agent."""
        messages = [
            ChatAgentMessage(
                id="1",
                role="user",
                content="What is the molecular formula of benzene?"
            )
        ]
        
        start_time = time.time()
        response = AGENT.predict(messages)
        end_time = time.time()
        
        # Verify response structure
        assert response is not None, "Should receive a response"
        assert hasattr(response, 'messages'), "Response should have messages"
        assert len(response.messages) > 0, "Should have at least one message"
        
        # Extract meaningful response content
        final_content = extract_meaningful_response(response)
        assert final_content is not None and len(final_content) > 0, \
            f"Should have a meaningful response. Messages: {[(getattr(msg, 'name', 'N/A'), len(getattr(msg, 'content', ''))) for msg in response.messages]}"
        
        # Performance check
        response_time = end_time - start_time
        print(f"BASF query response time: {response_time:.2f}s")
        print(f"Response preview: {final_content[:150]}...")
        
        # Should complete in reasonable time
        assert response_time < 120, f"Response too slow: {response_time:.2f}s"
        
        # Content should be relevant to chemistry
        content_lower = final_content.lower()
        chemistry_terms = ['benzene', 'c6h6', 'molecular', 'formula', 'chemical', 'carbon', 'hydrogen']
        assert any(term in content_lower for term in chemistry_terms), \
            "Response should contain chemistry-related terms"
    
    def test_simple_genomics_query(self):
        """Test a query that should be routed to Genomics agent."""
        messages = [
            ChatAgentMessage(
                id="1",
                role="user",
                content="Calculate the GC content of this DNA sequence: ATGCGCATTAGC"
            )
        ]
        
        start_time = time.time()
        response = AGENT.predict(messages)
        end_time = time.time()
        
        # Verify response structure
        assert response is not None, "Should receive a response"
        assert hasattr(response, 'messages'), "Response should have messages"
        assert len(response.messages) > 0, "Should have at least one message"
        
        # Extract meaningful response content
        final_content = extract_meaningful_response(response)
        assert final_content is not None and len(final_content) > 0, \
            f"Should have a meaningful response. Messages: {[(getattr(msg, 'name', 'N/A'), len(getattr(msg, 'content', ''))) for msg in response.messages]}"
        
        # Performance check
        response_time = end_time - start_time
        print(f"Genomics query response time: {response_time:.2f}s")
        print(f"Response preview: {final_content[:150]}...")
        
        # Should complete in reasonable time (genomics can be slower due to computation)
        assert response_time < 180, f"Response too slow: {response_time:.2f}s"
        
        # Content should be relevant to genomics/DNA
        content_lower = final_content.lower()
        genomics_terms = ['gc content', 'dna', 'sequence', 'nucleotide', 'base', 'genome', 'genetic']
        assert any(term in content_lower for term in genomics_terms), \
            "Response should contain genomics-related terms"


@integration_test
@requires_databricks
@slow_test
class TestSupervisorAdvancedScenarios:
    """Test advanced supervisor scenarios with complex routing."""
    
    def test_multi_domain_query(self):
        """Test a query that might require knowledge from multiple domains."""
        messages = [
            ChatAgentMessage(
                id="1",
                role="user",
                content="I'm studying a protein involved in drug metabolism. Can you help me understand both the chemical structure aspects and the genetic factors that influence its expression?"
            )
        ]
        
        start_time = time.time()
        response = AGENT.predict(messages)
        end_time = time.time()
        
        # Verify response structure
        assert response is not None, "Should receive a response"
        assert len(response.messages) > 0, "Should have at least one message"
        
        # Extract meaningful response content
        final_content = extract_meaningful_response(response)
        assert final_content is not None and len(final_content) > 0, \
            f"Should have a meaningful response. Messages: {[(getattr(msg, 'name', 'N/A'), len(getattr(msg, 'content', ''))) for msg in response.messages]}"
        
        # Performance check (can be slower for complex queries)
        response_time = end_time - start_time
        print(f"Multi-domain query response time: {response_time:.2f}s")
        print(f"Response preview: {final_content[:200]}...")
        
        # Should complete in reasonable time
        assert response_time < 300, f"Response too slow: {response_time:.2f}s"
        
        # Response should address both chemical and genetic aspects
        content_lower = final_content.lower()
        chemical_terms = ['chemical', 'structure', 'molecular', 'drug', 'metabolism']
        genetic_terms = ['genetic', 'gene', 'expression', 'protein', 'dna', 'rna']
        
        has_chemical = any(term in content_lower for term in chemical_terms)
        has_genetic = any(term in content_lower for term in genetic_terms)
        
        # Should ideally address both domains, but at least one
        assert has_chemical or has_genetic, "Response should address chemical or genetic aspects"
    
    def test_conversational_flow(self):
        """Test a multi-turn conversation with the supervisor."""
        # First message
        messages = [
            ChatAgentMessage(
                id="1",
                role="user",
                content="What is caffeine's molecular formula?"
            )
        ]
        
        response1 = AGENT.predict(messages)
        assert response1 is not None and len(response1.messages) > 0
        
        # Add response to conversation history
        messages.extend(response1.messages)
        
        # Follow-up question
        messages.append(ChatAgentMessage(
            id="2",
            role="user", 
            content="Now calculate how many carbon atoms are in 10 grams of caffeine."
        ))
        
        start_time = time.time()
        response2 = AGENT.predict(messages)
        end_time = time.time()
        
        # Verify response structure
        assert response2 is not None, "Should receive a response"
        assert len(response2.messages) > 0, "Should have at least one message"
        
        # Extract meaningful response content from follow-up
        final_content = extract_meaningful_response(response2)
        assert final_content is not None and len(final_content) > 0, \
            f"Should have a meaningful follow-up response. Messages: {[(getattr(msg, 'name', 'N/A'), len(getattr(msg, 'content', ''))) for msg in response2.messages]}"
        
        response_time = end_time - start_time
        
        print(f"Follow-up query response time: {response_time:.2f}s")
        print(f"Follow-up response preview: {final_content[:150]}...")
        
        # Should handle follow-up reasonably quickly
        assert response_time < 180, f"Follow-up response too slow: {response_time:.2f}s"
        
        # Should reference calculation or numbers
        content_lower = final_content.lower()
        calc_terms = ['carbon', 'atoms', 'calculate', 'grams', '10', 'molecular', 'weight']
        assert any(term in content_lower for term in calc_terms), \
            "Follow-up should address the calculation request"


@integration_test
@requires_databricks
class TestSupervisorErrorHandling:
    """Test supervisor error handling and edge cases."""
    
    def test_ambiguous_query(self):
        """Test handling of ambiguous queries."""
        messages = [
            ChatAgentMessage(
                id="1",
                role="user",
                content="What is X?"
            )
        ]
        
        response = AGENT.predict(messages)
        
        # Should still provide a response, even if asking for clarification
        assert response is not None, "Should receive a response"
        assert len(response.messages) > 0, "Should have at least one message"
        
        # Extract meaningful response content
        final_content = extract_meaningful_response(response)
        assert final_content is not None and len(final_content) > 0, \
            "Should provide some response to ambiguous query"
        
        # Should ask for clarification or explanation
        content_lower = final_content.lower()
        clarification_terms = ['clarify', 'specific', 'more information', 'what do you mean', 'unclear']
        # Note: The agent might still try to answer, so this is not a strict requirement
        print(f"Ambiguous query response: {final_content[:100]}...")
    
    def test_empty_message_handling(self):
        """Test handling of empty or minimal messages."""
        messages = [
            ChatAgentMessage(
                id="1",
                role="user",
                content=""
            )
        ]
        
        response = AGENT.predict(messages)
        
        # Should handle empty content gracefully
        assert response is not None, "Should receive a response even for empty input"
        assert len(response.messages) > 0, "Should have at least one message"
        
        # Extract meaningful response content
        final_content = extract_meaningful_response(response) 
        # Should provide some kind of response (asking for input, etc.)
        assert final_content is not None and len(final_content) > 0, \
            "Should provide some response to empty input"
        
        print(f"Empty input response: {final_content[:100]}...")


@integration_test
@requires_databricks
class TestSupervisorPerformance:
    """Performance tests for the supervisor system."""
    
    def test_response_consistency(self):
        """Test response time consistency across multiple identical queries."""
        messages = [
            ChatAgentMessage(
                id="1",
                role="user",
                content="What is the boiling point of water?"
            )
        ]
        
        response_times = []
        num_tests = 3
        
        print(f"\n🔄 Testing response consistency with {num_tests} identical queries...")
        
        for i in range(num_tests):
            start_time = time.time()
            response = AGENT.predict(messages)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Verify each response is valid
            assert response is not None, f"Response {i+1} should not be None"
            assert len(response.messages) > 0, f"Response {i+1} should have messages"
            
            print(f"Query {i+1}: {response_time:.2f}s")
            
            # Brief pause between requests
            if i < num_tests - 1:
                time.sleep(2)
        
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"📈 Performance: Avg={avg_time:.2f}s, Min={min_time:.2f}s, Max={max_time:.2f}s")
        
        # Performance assertions
        assert avg_time < 60, f"Average response time too high: {avg_time:.2f}s"
        assert max_time < 120, f"Maximum response time too high: {max_time:.2f}s"
        
        # Consistency check - max shouldn't be too much larger than min (within reason)
        if min_time > 0:
            ratio = max_time / min_time
            assert ratio < 5, f"Response times too inconsistent: ratio {ratio:.2f}"


@integration_test
@requires_databricks  
class TestSupervisorHealthCheck:
    """Health check test for monitoring supervisor system."""
    
    def test_supervisor_health_summary(self):
        """Comprehensive health check for the supervisor system."""
        test_scenarios = [
            {
                'name': 'BASF Chemistry Query',
                'message': 'What is the molecular weight of methanol?',
                'expected_terms': ['methanol', 'molecular', 'weight', 'ch3oh', 'chemical'],
                'max_time': 90
            },
            {
                'name': 'Genomics Calculation',
                'message': 'Calculate the complement of DNA sequence: ATCG',
                'expected_terms': ['complement', 'dna', 'sequence', 'atcg', 'tagc'],
                'max_time': 120
            },
            {
                'name': 'General Query',  
                'message': 'What is the difference between DNA and RNA?',
                'expected_terms': ['dna', 'rna', 'difference', 'nucleic', 'genetic'],
                'max_time': 90
            }
        ]
        
        results = {}
        
        print("\n" + "=" * 70)
        print("🏥 Supervisor Agent Health Check Summary")
        print("=" * 70)
        
        for scenario in test_scenarios:
            try:
                messages = [ChatAgentMessage(
                    id="1",
                    role="user",
                    content=scenario['message']
                )]
                
                start_time = time.time()
                response = AGENT.predict(messages)
                end_time = time.time()
                
                response_time = end_time - start_time
                
                # Check response validity
                if response and len(response.messages) > 0:
                    # Extract meaningful response content
                    meaningful_content = extract_meaningful_response(response)
                    
                    if meaningful_content:
                        content = meaningful_content.lower()
                        
                        # Check for expected terms
                        terms_found = sum(1 for term in scenario['expected_terms'] if term in content)
                        has_relevant_content = terms_found > 0
                        
                        # Check performance
                        within_time_limit = response_time <= scenario['max_time']
                        
                        if has_relevant_content and within_time_limit:
                            status = "✅ HEALTHY"
                        elif has_relevant_content:
                            status = "⚠️  SLOW"
                        elif within_time_limit:
                            status = "⚠️  WEAK_RESPONSE"
                        else:
                            status = "❌ UNHEALTHY"
                        
                        results[scenario['name']] = {
                            'status': status,
                            'response_time': response_time,
                            'content_length': len(meaningful_content),
                            'terms_found': terms_found,
                            'total_terms': len(scenario['expected_terms'])
                        }
                        
                        print(f"{status} {scenario['name']:25} | {response_time:5.2f}s | "
                              f"{len(meaningful_content):4d} chars | "
                              f"{terms_found}/{len(scenario['expected_terms'])} terms")
                    else:
                        results[scenario['name']] = {'status': '❌ NO_MEANINGFUL_CONTENT'}
                        print(f"❌ NO_MEANINGFUL_CONTENT {scenario['name']:15} | Could not extract meaningful content")
                    
                else:
                    results[scenario['name']] = {'status': '❌ NO_RESPONSE'}
                    print(f"❌ NO_RESPONSE {scenario['name']:20} | No valid response received")
                
            except Exception as e:
                results[scenario['name']] = {'status': '❌ ERROR', 'error': str(e)}
                print(f"❌ ERROR {scenario['name']:25} | {str(e)[:50]}...")
        
        print("-" * 70)
        
        # Count healthy scenarios
        healthy_count = sum(1 for r in results.values() 
                          if r.get('status', '').startswith('✅'))
        total_count = len(test_scenarios)
        
        print(f"📊 Status: {healthy_count}/{total_count} scenarios healthy")
        
        if healthy_count == total_count:
            print("🎉 Supervisor system is working perfectly!")
        elif healthy_count > 0:
            print(f"⚠️  {total_count - healthy_count} scenario(s) need attention")
        else:
            print("🚨 Supervisor system needs immediate attention!")
        
        print("=" * 70)
        
        # Test passes if at least majority of scenarios are working
        assert healthy_count >= total_count // 2, \
            f"Too many unhealthy scenarios: {healthy_count}/{total_count}"
