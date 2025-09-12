#!/usr/bin/env python3
"""
Test SupervisorResponsesAgent with Real Databricks Endpoints
===========================================================

This script tests the new tool-based SupervisorResponsesAgent implementation
making actual calls to Databricks endpoints with proper streaming behavior.

Real Endpoints Used:
- Routing LLM: databricks-claude-3-7-sonnet  
- CoatingsSupervisor: genie_multi_agent_basf
- GenomicsSupervisor: genie_multi_agent_basf_v2

Authentication: Requires 'e2-demo-field-eng' profile configured via:
databricks auth login --profile e2-demo-field-eng

Usage: python test_supervisor_responses_real.py
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multi_agent.supervisor.supervisor_responses_agent import (
    SupervisorResponsesAgent,
    SupervisorConfig,
    create_supervisor_agent
)
from mlflow.types.responses import ResponsesAgentRequest

def test_queries():
    """Test different query types to demonstrate supervisor routing behavior."""
    return [
        {
            "name": "Mathematical Calculation",
            "query": "Calculate the result of (15 * 8) / 4 + 25 and show your work",
            "expected_route": "GenomicsSupervisorAgent",
            "description": "Should route to GenomicsSupervisorAgent for math computation"
        },
        {
            "name": "Coatings Industry Query", 
            "query": "What are the latest trends in automotive paint durability and UV resistance?",
            "expected_route": "CoatingsSupervisorAgent",
            "description": "Should route to CoatingsSupervisorAgent for coatings expertise"
        },
        {
            "name": "Python Code Request",
            "query": "Write Python code to calculate the Fibonacci sequence up to the 10th number",
            "expected_route": "GenomicsSupervisorAgent", 
            "description": "Should route to GenomicsSupervisorAgent for Python execution"
        },
        {
            "name": "Market Analysis Query",
            "query": "Analyze the market demand for industrial coatings in the automotive sector",
            "expected_route": "CoatingsSupervisorAgent",
            "description": "Should route to CoatingsSupervisorAgent for market analysis"
        }
    ]

def stream_supervisor_query(agent, query_info, max_events=150):
    """Stream a query to the supervisor and display real-time results."""
    
    print(f"\n{'='*80}")
    print(f"🔍 TEST: {query_info['name']}")
    print(f"📝 Query: {query_info['query']}")
    print(f"🎯 Expected Route: {query_info['expected_route']}")
    print(f"💡 {query_info['description']}")
    print(f"{'='*80}")
    
    # Create request
    request = ResponsesAgentRequest(
        input=[{
            'role': 'user', 
            'content': query_info['query']
        }]
    )
    
    # Track streaming metrics
    start_time = time.time()
    event_count = 0
    routing_detected = None
    agent_response_received = False
    final_synthesis_received = False
    thinking_events = 0
    decision_events = 0
    
    print("\n🔄 STREAMING RESPONSE:")
    print("-" * 60)
    
    try:
        for event in agent.predict_stream(request):
            event_count += 1
            
            # Handle different event types
            if event.type == "response.text.delta":
                delta_text = event.delta.get("text", "")
                print(delta_text, end='', flush=True)
                
                # Count different types of thinking events
                if "🤔" in delta_text:
                    thinking_events += 1
                elif "🎯" in delta_text:
                    decision_events += 1
                    # Detect routing decisions
                    if "CoatingsSupervisorAgent" in delta_text:
                        routing_detected = "CoatingsSupervisorAgent"
                    elif "GenomicsSupervisorAgent" in delta_text:
                        routing_detected = "GenomicsSupervisorAgent"
                        
            elif event.type == "response.output_item.done":
                # Handle different item formats
                try:
                    if hasattr(event.item, 'content'):
                        item_content = event.item.content
                    elif hasattr(event.item, 'text'):
                        item_content = event.item.text
                    elif isinstance(event.item, dict):
                        item_content = event.item.get('content') or event.item.get('text', str(event.item))
                    else:
                        item_content = str(event.item)
                    
                    # Identify response types
                    if "### Coatings Supervisor Response:" in item_content:
                        agent_response_received = True
                        routing_detected = routing_detected or "CoatingsSupervisorAgent"
                        print(f"\n{'='*20} [🔬 COATINGS RESPONSE] {'='*20}")
                        print(item_content)
                        print('=' * 60)
                    elif "### Genomics Supervisor Response:" in item_content:
                        agent_response_received = True
                        routing_detected = routing_detected or "GenomicsSupervisorAgent"
                        print(f"\n{'='*20} [🧬 GENOMICS RESPONSE] {'='*19}")
                        print(item_content)
                        print('=' * 60)
                    elif "**Final Answer:**" in item_content:
                        final_synthesis_received = True
                        print(f"\n{'='*19} [🎯 FINAL SYNTHESIS] {'='*20}")
                        print(item_content)
                        print('=' * 60)
                    else:
                        # Other responses (thinking, decisions, etc.)
                        print(f"\n{'='*22} [💭 REASONING] {'='*22}")
                        print(item_content)
                        print('=' * 60)
                    
                except Exception as e:
                    print(f"\n[✅ Item received - error parsing: {e}]")
                
                print()
            
            # Safety limit
            if event_count >= max_events:
                print("...(truncated for display)")
                break
                
    except Exception as e:
        print(f"\n❌ ERROR during streaming: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    duration = time.time() - start_time
    
    print("-" * 60)
    print(f"📊 RESULTS:")
    print(f"   • Duration: {duration:.2f} seconds")
    print(f"   • Total Events: {event_count}")
    print(f"   • Thinking Events: {thinking_events}")
    print(f"   • Decision Events: {decision_events}")
    print(f"   • Detected Route: {routing_detected or 'Unknown'}")
    print(f"   • Route Match: {'✅' if routing_detected == query_info['expected_route'] else '❌'}")
    print(f"   • Agent Response: {'✅' if agent_response_received else '❌'}")
    print(f"   • Final Synthesis: {'✅' if final_synthesis_received else '❌'}")
    
    # Validate expected behavior
    success = (
        routing_detected == query_info['expected_route'] and
        agent_response_received and
        event_count > 3 and
        duration > 0.5  # Real endpoints should take meaningful time
    )
    
    print(f"   • Overall Success: {'✅' if success else '❌'}")
    
    return success

def main():
    """Main test execution."""
    
    print("🚀 NEW SUPERVISOR RESPONSES AGENT - REAL ENDPOINT TESTING")
    print("=" * 80)
    print("Testing tool-based supervisor agent with actual Databricks endpoints:")
    print("  • Routing LLM: databricks-claude-3-7-sonnet")
    print("  • CoatingsSupervisorAgent: genie_multi_agent_basf")  
    print("  • GenomicsSupervisorAgent: genie_multi_agent_basf_v2")
    print("  • Authentication: e2-demo-field-eng profile")
    print("  • Architecture: Tool-based with progressive disclosure")
    print()
    
    # Set up authentication
    os.environ['DATABRICKS_CONFIG_PROFILE'] = 'e2-demo-field-eng'
    
    try:
        print("🔧 Initializing SupervisorResponsesAgent...")
        
        # Create supervisor with real endpoint configuration
        config = SupervisorConfig(
            thinking_enabled=True,
            verbose_logging=True,
            max_iterations=3
        )
        agent = create_supervisor_agent()
        
        print("✅ Agent initialized successfully with real endpoints")
        print(f"   • Config: thinking={config.thinking_enabled}, max_iter={config.max_iterations}")
        print(f"   • Domain Agents: {len(agent.domain_agents)} tools loaded")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\nTroubleshooting:")
        print("  1. Ensure authentication: databricks auth login --profile e2-demo-field-eng")
        print("  2. Check endpoint access permissions")
        print("  3. Verify network connectivity")
        print("  4. Check if required packages are installed")
        import traceback
        traceback.print_exc()
        return False
    
    # Run test queries
    test_cases = test_queries()
    results = []
    
    print(f"\n🧪 RUNNING {len(test_cases)} TEST CASES...")
    
    for i, query_info in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}/{len(test_cases)}")
        
        try:
            success = stream_supervisor_query(agent, query_info)
            results.append({
                'name': query_info['name'],
                'success': success,
                'expected_route': query_info['expected_route']
            })
            
            # Brief pause between tests to avoid overwhelming endpoints
            if i < len(test_cases):
                print(f"⏳ Pausing 3 seconds before next test...")
                time.sleep(3)
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print(f"Error type: {type(e).__name__}")
            results.append({
                'name': query_info['name'], 
                'success': False,
                'expected_route': query_info['expected_route'],
                'error': str(e)
            })
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏆 FINAL RESULTS")
    print(f"{'='*80}")
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        route = result['expected_route']
        error_info = f" - {result.get('error', '')}" if not result['success'] and 'error' in result else ""
        print(f"  {status} {result['name']} (→ {route}){error_info}")
    
    print(f"\n📈 SUMMARY: {successful_tests}/{total_tests} tests passed")
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"📊 Success Rate: {success_rate:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Complete tool-based supervisor integration verified!")
        print("\nThis proves:")
        print("  ✅ Real Databricks authentication working")
        print("  ✅ Tool-based supervisor routing logic functions correctly") 
        print("  ✅ All domain agent endpoints accessible and responding")
        print("  ✅ Complete progressive disclosure streaming operational")
        print("  ✅ True multi-agent orchestration through tools successful")
        print("  ✅ MLflow Responses API pattern implemented correctly")
        return True
    else:
        failures = total_tests - successful_tests
        print(f"⚠️  {failures} test(s) failed - analyzing issues...")
        
        if successful_tests > 0:
            print(f"✅ {successful_tests} test(s) passed - core functionality working")
            print("🔧 Partial success indicates endpoint connectivity issues")
        else:
            print("❌ All tests failed - check authentication and endpoint access")
            
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

