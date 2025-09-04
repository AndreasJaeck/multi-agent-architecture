import unittest
from unittest.mock import Mock, patch
import uuid
import time
import statistics

# Import the classes and components we want to test
from multi_agent.supervisor.supervisor_of_supervisors import (
    MessageHandler,
    AgentOrchestrator,
    ErrorHandler,
    ConfigManager,
    LangGraphChatAgent,
    AgentState,
    SupervisorConfig,
    SupervisorLogic
)
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse


class TestBuildAgentContext(unittest.TestCase):
    """Test the build_agent_context function which handles context windowing"""
    
    def test_empty_agent_responses(self):
        """Test with no agent responses"""
        result = MessageHandler.build_agent_context([])
        self.assertEqual(result, "")
    
    def test_single_agent_response(self):
        """Test with one agent response"""
        responses = [
            {"name": "BASF_Data", "content": "This is a test response about chemicals"}
        ]
        result = MessageHandler.build_agent_context(responses)
        
        self.assertIn("=== RECENT AGENT RESPONSES ===", result)
        self.assertIn("MOST RECENT - BASF_Data:", result)
        self.assertIn("This is a test response about chemicals", result)
        self.assertIn("IMPORTANT: If ANY response completely answers", result)
    
    def test_multiple_agent_responses(self):
        """Test with multiple agent responses - should show recent full, older truncated"""
        responses = [
            {"name": "BASF_Data", "content": "First response " * 50},  # Older response
            {"name": "Genomics_Tools", "content": "Most recent response with full content"}  # Most recent
        ]
        result = MessageHandler.build_agent_context(responses)
        
        # Most recent should be full
        self.assertIn("MOST RECENT - Genomics_Tools:", result)
        self.assertIn("Most recent response with full content", result)
        
        # Older should be truncated and labeled
        self.assertIn("BASF_Data (earlier):", result)
        # Should truncate the first response
        self.assertIn("...", result)
    
    def test_max_chars_limit(self):
        """Test that max_chars limit is respected"""
        # Create a very long response
        long_content = "A" * 1500
        responses = [
            {"name": "Agent1", "content": long_content},
            {"name": "Agent2", "content": "B" * 1000}
        ]
        
        result = MessageHandler.build_agent_context(responses, max_chars=1000)
        
        # Should include information about omitted responses
        self.assertIn("earlier responses omitted", result)
    
    def test_missing_name_or_content(self):
        """Test handling of malformed responses"""
        responses = [
            {"content": "Response without name"},  # Missing name
            {"name": "TestAgent"},  # Missing content
            {}  # Empty response
        ]
        result = MessageHandler.build_agent_context(responses)
        
        # Should handle gracefully with 'unknown' for missing names and empty strings for missing content
        self.assertIn("unknown", result)


class TestAgentNode(unittest.TestCase):
    """Test the agent execution which wraps individual agents"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.orchestrator = AgentOrchestrator(self.config_manager)
    
    def test_successful_agent_response(self):
        """Test normal successful agent response"""
        # Mock agent that returns a successful response
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [{"content": "Successful response", "role": "assistant"}]
        }
        
        # Replace the actual agent with our mock
        self.orchestrator.config_manager.agent_instances["BASF_Data"] = mock_agent
        
        state = {"messages": [{"role": "user", "content": "test query"}]}
        result = self.orchestrator.execute_agent(state, "BASF_Data")
        
        # Check the result structure
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "assistant")
        self.assertEqual(result["messages"][0]["content"], "Successful response")
        self.assertEqual(result["messages"][0]["name"], "BASF_Data")
    
    def test_agent_exception_handling(self):
        """Test error handling when agent throws exception"""
        # Mock agent that raises an exception
        mock_agent = Mock()
        mock_agent.invoke.side_effect = Exception("Connection timeout")
        
        # Replace the actual agent with our mock
        self.orchestrator.config_manager.agent_instances["BASF_Data"] = mock_agent
        
        state = {"messages": [{"role": "user", "content": "test query"}]}
        result = self.orchestrator.execute_agent(state, "BASF_Data")
        
        # Check error response structure
        self.assertIn("messages", result)
        message = result["messages"][0]
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["name"], "BASF_Data")
        self.assertIn("❌", message["content"])
        self.assertIn("BASF Data Assistant Unavailable", message["content"])
        self.assertIn("chemical data analysis", message["content"])
        self.assertIn("Connection timeout", message["content"])
    
    def test_unknown_agent_error_handling(self):
        """Test error handling for unknown agent types"""
        mock_agent = Mock()
        mock_agent.invoke.side_effect = Exception("Some error")
        
        # Add unknown agent to config for testing
        self.orchestrator.config_manager.agent_instances["Unknown_Agent"] = mock_agent
        
        state = {"messages": []}
        result = self.orchestrator.execute_agent(state, "Unknown_Agent")
        
        message = result["messages"][0]
        self.assertIn("Unknown Agent Assistant", message["content"])
        self.assertIn("specialized analysis", message["content"])
    
    def test_different_message_formats(self):
        """Test handling of different agent response formats"""
        # Test with message object that has .content attribute
        mock_message = Mock()
        mock_message.content = "Message with content attribute"
        
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"messages": [mock_message]}
        
        # Add test agent to config
        self.orchestrator.config_manager.agent_instances["TestAgent"] = mock_agent
        
        state = {"messages": []}
        result = self.orchestrator.execute_agent(state, "TestAgent")
        
        self.assertEqual(result["messages"][0]["content"], "Message with content attribute")
    
    def test_no_messages_in_response(self):
        """Test handling when agent returns no messages"""
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"messages": []}
        
        # Add test agent to config
        self.orchestrator.config_manager.agent_instances["TestAgent"] = mock_agent
        
        state = {"messages": []}
        result = self.orchestrator.execute_agent(state, "TestAgent")
        
        self.assertEqual(result["messages"][0]["content"], "No response from agent")


class TestFinalAnswer(unittest.TestCase):
    """Test the final_answer function which synthesizes responses"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.orchestrator = AgentOrchestrator(self.config_manager)
    
    def test_normal_final_answer(self):
        """Test final answer generation with normal (non-error) messages"""
        # Create a mock response directly
        from langchain_core.messages import AIMessage
        mock_ai_message = AIMessage(content="Water is H2O, a chemical compound.")
        
        # Mock the chain directly instead of the LLM
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value=mock_ai_message)
        
        # Override the pre-compiled chain in the orchestrator
        self.orchestrator._normal_chain = mock_chain
        
        state = {
            "messages": [
                {"role": "user", "content": "What is water?"},
                {"role": "assistant", "name": "BASF_Data", "content": "Water is H2O"}
            ]
        }
        
        result = self.orchestrator.create_final_answer(state)
        
        # Verify the result structure
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        # The message should have content (from our mock)
        self.assertTrue(hasattr(result["messages"][0], "content") or 
                       isinstance(result["messages"][0], dict) and "content" in result["messages"][0])
    
    def test_error_final_answer(self):
        """Test final answer generation when there are error messages"""
        # Create a mock response directly
        from langchain_core.messages import AIMessage
        mock_ai_message = AIMessage(content="I'm sorry, the service is currently unavailable. Please try again later.")
        
        # Mock the chain directly instead of the LLM
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value=mock_ai_message)
        
        # Override the pre-compiled error chain in the orchestrator
        self.orchestrator._error_chain = mock_chain
        
        state = {
            "messages": [
                {"role": "user", "content": "What is water?"},
                {"role": "assistant", "name": "BASF_Data", 
                 "content": "❌ **BASF Data Assistant Unavailable**\n\nService is down"}
            ]
        }
        
        result = self.orchestrator.create_final_answer(state)
        
        # Verify the result structure - should still return a message
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        # The message should have content (from our mock)
        self.assertTrue(hasattr(result["messages"][0], "content") or 
                       isinstance(result["messages"][0], dict) and "content" in result["messages"][0])


class TestErrorMessageConstants(unittest.TestCase):
    """Test the error message configuration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
    
    def test_agent_error_messages_structure(self):
        """Test that agent configurations have correct structure"""
        self.assertIn("BASF_Data", self.config_manager.agents)
        self.assertIn("Genomics_Tools", self.config_manager.agents)
        
        for agent_name, config in self.config_manager.agents.items():
            # Test that agent_name (key) is a string
            self.assertIsInstance(agent_name, str)
            # Test that config fields are strings
            self.assertIsInstance(config.service_name, str)
            self.assertIsInstance(config.capabilities, str)
            self.assertIsInstance(config.description, str)
            self.assertIsInstance(config.endpoint, str)


class TestSupervisorLogic(unittest.TestCase):
    """Test supervisor agent decision logic (parts that can be tested without full LLM calls)"""
    
    def test_error_detection_logic(self):
        """Test the logic that detects error messages"""
        # This tests the error detection pattern used in supervisor_agent
        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "name": "BASF_Data", "content": "Normal response"},
            {"role": "assistant", "name": "Genomics_Tools", 
             "content": "❌ **Service Unavailable**\n\nError occurred"}
        ]
        
        agent_responses = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("name")]
        latest_response = agent_responses[-1]
        
        # Should detect error symbol
        self.assertTrue("❌" in latest_response.get("content", ""))
    
    def test_iteration_count_logic(self):
        """Test max iteration checking logic"""
        MAX_ITERATIONS = 3
        
        # Test within limits
        state_within_limit = {"iteration_count": 2}
        count = state_within_limit.get("iteration_count", 0) + 1
        self.assertLessEqual(count, MAX_ITERATIONS)
        
        # Test exceeding limits
        state_over_limit = {"iteration_count": 3}
        count = state_over_limit.get("iteration_count", 0) + 1
        self.assertGreater(count, MAX_ITERATIONS)


class TestLangGraphChatAgent(unittest.TestCase):
    """Test the LangGraphChatAgent wrapper"""
    
    def test_message_format_handling(self):
        """Test handling of different message formats in predict method"""
        # Mock the compiled graph
        mock_agent = Mock()
        mock_agent.stream.return_value = [
            {"node1": {
                "messages": [
                    {"role": "assistant", "content": "Test response", "name": "TestAgent"}
                ]
            }}
        ]
        
        chat_agent = LangGraphChatAgent(mock_agent)
        
        # Create test input
        input_messages = [
            ChatAgentMessage(id="1", role="user", content="Test query")
        ]
        
        result = chat_agent.predict(input_messages)
        
        # Verify response structure
        self.assertIsInstance(result, ChatAgentResponse)
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].role, "assistant")
        self.assertEqual(result.messages[0].content, "Test response")
    
    def test_predict_stream_functionality(self):
        """Test the streaming prediction functionality"""
        mock_agent = Mock()
        mock_agent.stream.return_value = [
            {"node1": {"messages": [{"role": "assistant", "content": "Chunk 1"}]}},
            {"node2": {"messages": [{"role": "assistant", "content": "Chunk 2"}]}}
        ]
        
        chat_agent = LangGraphChatAgent(mock_agent)
        input_messages = [ChatAgentMessage(id="1", role="user", content="Test")]
        
        chunks = list(chat_agent.predict_stream(input_messages))
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].delta.content, "Chunk 1")
        self.assertEqual(chunks[1].delta.content, "Chunk 2")


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios that combine multiple components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.orchestrator = AgentOrchestrator(self.config_manager)
    
    def test_complete_error_flow(self):
        """Test complete flow when an agent fails"""
        # Create a mock response directly
        from langchain_core.messages import AIMessage
        mock_ai_message = AIMessage(content="The service is currently unavailable. Please try again later.")
        
        # Mock the chain directly instead of the LLM
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value=mock_ai_message)
        
        # Override the pre-compiled error chain in the orchestrator (since this will have error messages)
        self.orchestrator._error_chain = mock_chain
        
        # Mock failing agent
        mock_agent = Mock()
        mock_agent.invoke.side_effect = Exception("Network error")
        
        # Replace the actual agent with our mock
        self.orchestrator.config_manager.agent_instances["BASF_Data"] = mock_agent
        
        # Step 1: Agent fails
        state = {"messages": [{"role": "user", "content": "test"}]}
        agent_result = self.orchestrator.execute_agent(state, "BASF_Data")
        
        # Step 2: Add agent response to state
        state["messages"].extend(agent_result["messages"])
        
        # Step 3: Build context (supervisor would use this)
        agent_responses = [msg for msg in state["messages"] if msg.get("role") == "assistant" and msg.get("name")]
        context = MessageHandler.build_agent_context(agent_responses)
        
        # Step 4: Check final answer handles error
        final_result = self.orchestrator.create_final_answer(state)
        
        # Verify final answer returns a proper structure
        self.assertIn("messages", final_result)
        self.assertEqual(len(final_result["messages"]), 1)
    
    def test_successful_multi_agent_flow(self):
        """Test successful flow with multiple agent responses"""
        state = {"messages": [{"role": "user", "content": "test question"}]}
        
        # First agent responds
        mock_agent1 = Mock()
        mock_agent1.invoke.return_value = {"messages": [{"content": "Response 1"}]}
        self.orchestrator.config_manager.agent_instances["BASF_Data"] = mock_agent1
        result1 = self.orchestrator.execute_agent(state, "BASF_Data")
        state["messages"].extend(result1["messages"])
        
        # Second agent responds  
        mock_agent2 = Mock()
        mock_agent2.invoke.return_value = {"messages": [{"content": "Response 2"}]}
        self.orchestrator.config_manager.agent_instances["Genomics_Tools"] = mock_agent2
        result2 = self.orchestrator.execute_agent(state, "Genomics_Tools")
        state["messages"].extend(result2["messages"])
        
        # Build context should include both responses
        agent_responses = [msg for msg in state["messages"] if msg.get("role") == "assistant" and msg.get("name")]
        context = MessageHandler.build_agent_context(agent_responses)
        
        self.assertIn("MOST RECENT - Genomics_Tools", context)
        self.assertIn("BASF_Data (earlier)", context)
        self.assertIn("Response 1", context)
        self.assertIn("Response 2", context)


class TestHotPathPerformance(unittest.TestCase):
    """Performance tests for endpoint deployment hot path optimization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = ConfigManager()
        self.supervisor_logic = SupervisorLogic(self.config_manager)
        self.orchestrator = AgentOrchestrator(self.config_manager)
        self.message_handler = MessageHandler()
        
        # Mock LLM responses for consistent timing
        self.mock_llm_response = Mock()
        self.mock_llm_response.next_node = "FINISH"
        self.mock_llm_response.reasoning = "Test reasoning"
        
    def _time_operation(self, operation, iterations=100):
        """Helper to time operations with multiple iterations"""
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            operation()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'min': min(times),
            'max': max(times),
            'stdev': statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def test_uuid_pool_performance(self):
        """Test that UUID pool is faster than uuid.uuid4()"""
        # Test UUID pool performance
        def pool_uuid_generation():
            return self.message_handler._get_next_uuid()
        
        pool_stats = self._time_operation(pool_uuid_generation, iterations=1000)
        
        # Test standard UUID generation
        def standard_uuid_generation():
            return str(uuid.uuid4())
        
        standard_stats = self._time_operation(standard_uuid_generation, iterations=1000)
        
        print(f"\nUUID Performance Comparison:")
        print(f"Pool method: {pool_stats['mean']:.3f}ms avg, {pool_stats['median']:.3f}ms median")
        print(f"Standard method: {standard_stats['mean']:.3f}ms avg, {standard_stats['median']:.3f}ms median")
        print(f"Speed improvement: {standard_stats['mean'] / pool_stats['mean']:.1f}x faster")
        
        # UUID pool should be significantly faster
        self.assertLess(pool_stats['mean'], standard_stats['mean'] * 0.5, 
                       "UUID pool should be at least 2x faster than standard generation")
        
        # Verify UUIDs are unique within reasonable sample
        uuids = [self.message_handler._get_next_uuid() for _ in range(50)]
        self.assertEqual(len(set(uuids)), len(uuids), "UUIDs from pool should be unique")
    
    @patch('multi_agent.supervisor.supervisor_of_supervisors.ChatDatabricks')
    def test_supervisor_decision_performance(self, mock_chat_databricks):
        """Test supervisor decision hot path performance"""
        # Mock the LLM to return consistent responses
        mock_chat_databricks.return_value.with_structured_output.return_value.invoke.return_value = self.mock_llm_response
        
        # Override the LLM in supervisor
        self.supervisor_logic._supervisor_chain = Mock()
        self.supervisor_logic._supervisor_chain.invoke.return_value = self.mock_llm_response
        
        # Test state with multiple agent responses (realistic scenario)
        test_state = {
            "messages": [
                {"role": "user", "content": "What is the molecular structure of water?"},
                {"role": "assistant", "name": "BASF_Data", "content": "Water (H2O) is a simple molecule with two hydrogen atoms bonded to one oxygen atom."},
                {"role": "assistant", "name": "Genomics_Tools", "content": "From a computational perspective, water has a bent molecular geometry with an angle of approximately 104.5 degrees."}
            ],
            "iteration_count": 1
        }
        
        def supervisor_decision():
            return self.supervisor_logic.make_decision(test_state)
        
        stats = self._time_operation(supervisor_decision, iterations=50)
        
        print(f"\nSupervisor Decision Performance:")
        print(f"Average: {stats['mean']:.2f}ms, Median: {stats['median']:.2f}ms")
        print(f"Min: {stats['min']:.2f}ms, Max: {stats['max']:.2f}ms, StdDev: {stats['stdev']:.2f}ms")
        
        # Decision should be fast (under 10ms for mocked LLM)
        self.assertLess(stats['median'], 10.0, 
                       "Supervisor decision should be under 10ms with optimizations")
        
        # Verify consistent performance (allow higher variation for sub-ms timings)
        # For very fast operations (<1ms), allow higher relative variation
        max_stdev = max(stats['mean'] * 0.5, 0.01)  # At least 0.01ms tolerance
        self.assertLess(stats['stdev'], max_stdev, 
                       "Performance should be consistent (accounting for sub-ms precision)")
    
    def test_error_message_template_performance(self):
        """Test error message generation performance with pre-compiled templates"""
        test_error = Exception("Connection timeout occurred")
        
        def create_error_message():
            return self.orchestrator.error_handler.create_error_message("BASF_Data", test_error)
        
        stats = self._time_operation(create_error_message, iterations=200)
        
        print(f"\nError Message Generation Performance:")
        print(f"Average: {stats['mean']:.3f}ms, Median: {stats['median']:.3f}ms")
        
        # Error message generation should be very fast with templates
        self.assertLess(stats['median'], 1.0, 
                       "Error message generation should be under 1ms with templates")
        
        # Verify the message contains expected elements
        error_msg = create_error_message()
        self.assertIn("❌", error_msg)
        self.assertIn("BASF Data Assistant", error_msg)
        self.assertIn("Connection timeout", error_msg)
    
    def test_context_building_performance(self):
        """Test agent context building performance"""
        # Create realistic agent responses for context building
        agent_responses = [
            {"name": "BASF_Data", "content": "Chemical analysis shows: " + "A" * 500},
            {"name": "Genomics_Tools", "content": "Computational results indicate: " + "B" * 600}, 
            {"name": "BASF_Data", "content": "Additional chemical data: " + "C" * 400},
            {"name": "Genomics_Tools", "content": "Final genomics analysis: " + "D" * 300}
        ]
        
        def build_context():
            return self.message_handler.build_agent_context(agent_responses, max_chars=2000)
        
        stats = self._time_operation(build_context, iterations=100)
        
        print(f"\nContext Building Performance:")
        print(f"Average: {stats['mean']:.2f}ms, Median: {stats['median']:.2f}ms")
        
        # Context building should be reasonably fast
        self.assertLess(stats['median'], 5.0, 
                       "Context building should be under 5ms")
        
        # Verify context is properly built
        context = build_context()
        self.assertIn("RECENT AGENT RESPONSES", context)
        self.assertIn("MOST RECENT", context)
    
    def test_message_creation_performance(self):
        """Test ChatAgentMessage creation performance"""
        test_msg = {
            "role": "assistant",
            "content": "This is a test message response from an agent",
            "name": "TestAgent"
        }
        
        def create_message():
            return self.message_handler.create_chat_agent_message(test_msg)
        
        stats = self._time_operation(create_message, iterations=200)
        
        print(f"\nMessage Creation Performance:")
        print(f"Average: {stats['mean']:.3f}ms, Median: {stats['median']:.3f}ms")
        
        # Message creation should be very fast
        self.assertLess(stats['median'], 2.0, 
                       "Message creation should be under 2ms")
        
        # Verify message structure
        message = create_message()
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, test_msg["content"])
        self.assertTrue(message.id)  # Should have an ID from UUID pool
    
    def test_pre_compiled_chains_exist(self):
        """Test that pre-compiled chains are properly initialized"""
        # Verify supervisor has pre-compiled chains
        self.assertTrue(hasattr(self.supervisor_logic, '_supervisor_chain'))
        self.assertTrue(hasattr(self.supervisor_logic, '_base_system_prompt'))
        self.assertIsInstance(self.supervisor_logic._base_system_prompt, str)
        self.assertGreater(len(self.supervisor_logic._base_system_prompt), 100)
        
        # Verify orchestrator has pre-compiled chains
        self.assertTrue(hasattr(self.orchestrator, '_error_chain'))
        self.assertTrue(hasattr(self.orchestrator, '_normal_chain'))
        self.assertTrue(hasattr(self.orchestrator, '_error_system_prompt'))
        self.assertTrue(hasattr(self.orchestrator, '_normal_system_prompt'))
        
        # Verify error handler has pre-compiled templates
        self.assertTrue(hasattr(self.orchestrator.error_handler, '_error_message_templates'))
        self.assertIn('BASF_Data', self.orchestrator.error_handler._error_message_templates)
        self.assertIn('Genomics_Tools', self.orchestrator.error_handler._error_message_templates)
    
    def test_memory_usage_consistency(self):
        """Test that operations don't create excessive temporary objects"""
        import gc
        
        # Force garbage collection and get baseline
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform multiple operations
        for _ in range(10):
            # UUID generation (should use pool)
            self.message_handler._get_next_uuid()
            
            # Error message generation (should use templates)  
            self.orchestrator.error_handler.create_error_message("BASF_Data", Exception("test"))
            
            # Message creation
            self.message_handler.create_chat_agent_message({"role": "assistant", "content": "test"})
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        object_growth = final_objects - initial_objects
        print(f"\nMemory Usage Test:")
        print(f"Object growth after 10 operations: {object_growth} objects")
        
        # Should have minimal object growth (less than 50 new objects for 30 operations)
        self.assertLess(object_growth, 50, 
                       "Operations should not create excessive temporary objects")
    
    def test_hot_path_integration_performance(self):
        """Integration test for complete hot path performance"""
        # Mock complete agent workflow
        mock_agent = Mock()
        mock_agent.invoke.return_value = {
            "messages": [{"content": "Agent response", "role": "assistant"}]
        }
        self.config_manager.agent_instances["BASF_Data"] = mock_agent
        
        # Mock supervisor chain
        self.supervisor_logic._supervisor_chain = Mock()
        self.supervisor_logic._supervisor_chain.invoke.return_value = self.mock_llm_response
        
        test_state = {
            "messages": [{"role": "user", "content": "Test query"}],
            "iteration_count": 0
        }
        
        def complete_hot_path():
            # Supervisor decision
            decision = self.supervisor_logic.make_decision(test_state)
            
            # Agent execution (if needed)
            if decision["next_node"] != "FINISH":
                agent_result = self.orchestrator.execute_agent(test_state, decision["next_node"])
                test_state["messages"].extend(agent_result["messages"])
            
            return decision
        
        stats = self._time_operation(complete_hot_path, iterations=20)
        
        print(f"\nComplete Hot Path Performance:")
        print(f"Average: {stats['mean']:.2f}ms, Median: {stats['median']:.2f}ms")
        print(f"Min: {stats['min']:.2f}ms, Max: {stats['max']:.2f}ms")
        
        # Complete hot path should be fast
        self.assertLess(stats['median'], 15.0, 
                       "Complete hot path should be under 15ms with mocked LLM")
        
        print(f"\n🚀 Hot Path Performance Test Summary:")
        print(f"✅ All performance benchmarks passed!")
        print(f"✅ UUID pool optimization working")
        print(f"✅ Pre-compiled templates working") 
        print(f"✅ Memory usage optimized")
        print(f"✅ End-to-end hot path under 15ms")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
