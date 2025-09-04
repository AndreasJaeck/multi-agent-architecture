# Integration Tests

Integration tests that verify the multi-agent supervisor system with real Databricks endpoints.

> **Note**: For running these tests, see the main [README.md](../../README.md#testing) and [CLI_USAGE.md](../../CLI_USAGE.md) for command examples.

## Databricks Endpoint Tests (`test_databricks_connectivity.py`)

### Authentication Tests (`TestDatabricksAuthentication`)
- **Profile Validation**: Verifies Databricks CLI profile is properly configured
- **Workspace Access**: Tests connection to the Databricks workspace  
- **Token Authentication**: Validates OAuth token is working

### Connectivity Tests (`TestEndpointConnectivity`)
- **Endpoint Discovery**: Tests connection to each serving endpoint
- **Client Creation**: Verifies endpoint clients can be instantiated
- **Basic Properties**: Checks endpoint configuration and status

### Request/Response Tests (`TestEndpointRequests`)
- **BASF Endpoint**: Simple chemistry query ("What is water?")
- **Genomics Endpoint**: Basic computational query ("Calculate 2+2")
- **LLM Endpoint**: Direct language model interaction
- **Response Validation**: Checks for non-empty, meaningful responses

### Performance Tests (`TestEndpointPerformance`)
- **Response Time Consistency**: Multiple requests to measure variance
- **Concurrent Requests**: Basic load testing with parallel calls
- **Timeout Handling**: Tests proper timeout behavior

### Health Check Tests (`TestEndpointHealthCheck`)
- **Quick Status Check**: Fast health verification for all endpoints
- **Monitoring Ready**: Suitable for automated health monitoring
- **Summary Report**: Provides status overview with timing

## Supervisor Integration Tests (`test_supervisor/test_supervisor_integration.py`)

End-to-end tests for the complete multi-agent supervisor system.

### Basic Functionality Tests (`TestSupervisorBasicFunctionality`)
- **Agent Initialization**: Verifies supervisor can create and initialize all agents
- **BASF Chemistry Query**: Tests routing to BASF agent with question "What is caffeine?"
- **Genomics Computation Query**: Tests routing to genomics agent with "What is 15*7?"
- **Response Quality**: Validates responses contain expected domain-specific content
- **Agent Attribution**: Confirms responses are properly attributed to specific agents

### Advanced Scenarios (`TestSupervisorAdvancedScenarios`)
- **Multi-Domain Query**: Complex question requiring both chemistry and computation
  - Example: "What is caffeine's molecular formula and calculate the molecular weight?"
  - Tests supervisor's ability to coordinate multiple agents
- **Conversational Flow**: Multi-turn conversation with context preservation
  - First: Ask about caffeine's molecular formula
  - Second: "Now calculate how many carbon atoms are in 10 grams of caffeine"
  - Tests context carrying and follow-up handling

### Error Handling Tests (`TestSupervisorErrorHandling`) 
- **Ambiguous Query**: Questions that could go to multiple agents
  - Tests supervisor's decision-making for unclear routing
- **Empty Message**: Tests handling of empty or whitespace-only input
- **Service Unavailability**: Graceful degradation when individual agents are offline
- **Malformed Input**: Response to invalid message formats

### Performance Tests (`TestSupervisorPerformance`)
- **Response Time Consistency**: Multiple identical queries to measure variance
- **Complex Query Performance**: Multi-step queries within time limits
- **Memory Usage**: Checks for memory leaks in long conversations
- **Concurrent Handling**: Basic load testing with parallel requests

### Health Check Tests (`TestSupervisorHealthCheck`)
- **End-to-End Verification**: Quick test of each routing path
- **Agent Status Check**: Verifies all agents are responsive
- **System Integration**: Complete pipeline health validation
- **Monitoring Metrics**: Response times, success rates, error patterns

## Test Validation Criteria

### Expected Performance
- **Endpoint Health Checks**: < 60 seconds per endpoint
- **Basic Supervisor Queries**: < 2 minutes (single agent routing)
- **Multi-Domain Queries**: < 5 minutes (multiple agent coordination) 
- **Conversational Turns**: < 3 minutes per follow-up
- **Concurrent Request Success**: > 67% success rate

### Content Validation
- **Domain-Specific Responses**: Chemistry queries mention molecular concepts, math queries show calculations
- **Agent Attribution**: Responses clearly indicate which agent provided the answer
- **Context Preservation**: Follow-up questions reference previous conversation
- **Error Handling**: Clear error messages when services are unavailable

### Test Prerequisites
- **Databricks CLI**: Configured with OAuth profile (see main README)
- **Endpoint Access**: All three serving endpoints must be deployed and accessible
- **Network**: Stable connection for consistent timing measurements

## Common Test Scenarios

### Success Cases
```
Query: "What is caffeine?"
Expected: BASF agent response about molecular structure (C₈H₁₀N₄O₂)

Query: "Calculate 15*7"  
Expected: Genomics agent response with calculation result (105)

Query: "What is water's formula and its molecular weight?"
Expected: Coordination between agents for chemistry + computation
```

### Error Cases
```
Empty Query: ""
Expected: BadRequestError with clear message

Service Down: Agent endpoint unavailable  
Expected: "❌ Agent Unavailable" message with fallback

Ambiguous Query: "Tell me about analysis"
Expected: Supervisor chooses most appropriate agent with explanation
```
