# UC Multi-Tool Responses Agent

A sophisticated tool-calling agent implementation built on MLflow's ResponsesAgent framework, designed for complex multi-tool interactions with Unity Catalog functions, vector search, and agent endpoints.

## Quick Start

1. **Update Configuration**: Edit the `CONFIG` dictionary in `agent.py` to use your actual Databricks resources
2. **Replace Placeholders**:
   - `your_model_endpoint` → Your Databricks model serving endpoint
   - `your_catalog.your_schema.*` → Your Unity Catalog functions and indices
   - `your_expert_model_endpoint*` → Your specialized model endpoints
3. **Run**: `python -c "from src.uc_multi_tool_responses_agent.agent import AgentManager; manager = AgentManager(); agent = manager.agent"`

⚠️ **Important**: This agent uses placeholder values that must be replaced with your actual Databricks resources before use.

## Overview

The `agent.py` implements a production-ready tool-calling agent that can:
- Execute Unity Catalog functions for data processing and computations
- Perform semantic vector searches
- Call specialized agent endpoints
- Stream responses with real-time tool execution
- Maintain thread-safe conversation state

## Architecture

### Core Components

#### 1. Configuration (`CONFIG`)
Centralized configuration for LLM endpoint, system prompt, and tool definitions:

```python
CONFIG = {
    "llm_endpoint": "your_model_endpoint",  # Replace with your Databricks model serving endpoint
    "max_iterations": 10,
    "system_prompt": "...",  # Comprehensive tool usage guidelines
    "tools": {
        "uc_functions": [
            "your_catalog.your_schema.your_function_name"
        ],     # Unity Catalog functions
        "vector_search": [
            {
                "index_name": "your_catalog.your_schema.your_vector_index",
                "tool_name": "semantic_search",
                "description": "Search based on semantic similarity",
                "num_results": 5,
                "query_type": "hybrid"
            }
        ],    # Vector search indices
        "agent_endpoints": [
            {
                "endpoint": "your_expert_endpoint",
                "tool_name": "domain_expert",
                "description": "Specialized agent for domain tasks"
            }
        ]   # Specialized agent endpoints
    }
}
```

#### 2. Message System (`Message` dataclass)
Unified message representation supporting multiple formats:
- **Responses API format**: For MLflow ResponsesAgent interface
- **Chat Completion format**: For OpenAI-compatible LLM calls

**Key Methods:**
- `to_cc_format()`: Converts to Chat Completion format for LLM input
- `from_responses_format()`: Creates Message from Responses API format
- `from_cc_format()`: Creates Message from Chat Completion format

#### 3. Tool System

##### ToolInfo Class
```python
@dataclass
class ToolInfo:
    name: str        # Tool identifier
    spec: dict       # OpenAI function specification
    exec_fn: Callable # Execution function
```

##### Tool Types
1. **UC Functions**: Direct calls to Unity Catalog functions
2. **Vector Search**: Semantic similarity searches
3. **Agent Endpoints**: Calls to other Databricks model serving endpoints

#### 4. ToolCallingAgent Class

**Inherits from**: `mlflow.pyfunc.ResponsesAgent`

**Key Features:**
- Thread-safe message handling with `threading.RLock()`
- Automatic retry logic with exponential backoff
- MLflow tracing integration
- Streaming response generation

### Message Flow Architecture

#### 1. Input Processing
```
User Input → ResponsesAgentRequest → Message conversion
```

#### 2. LLM Interaction
```
Messages → prep_msgs_for_cc_llm() → Chat Completion format → LLM call
```

#### 3. Streaming Response Generation
```
LLM chunks → output_to_responses_items_stream() → ResponsesAgentStreamEvent
```

#### 4. Tool Execution
```
Tool calls → execute_tool() → Results → Message updates
```

## Key Methods

### Initialization & Setup

#### `main()`
- Loads all configured tools
- Creates ToolCallingAgent instance
- Registers with MLflow
- Returns agent instance

#### `initialize_tools()`
- Loads UC functions using `UCFunctionToolkit`
- Configures vector search tools with `VectorSearchRetrieverTool`
- Creates agent endpoint executors

### Message Handling

#### `prep_msgs_for_cc_llm(messages)`
Converts Responses API messages to Chat Completion format:
- Filters out empty assistant messages with tool calls (except final message)
- Converts message formats for LLM compatibility

#### `output_to_responses_items_stream(chunks, current_messages, on_messages_updated)`
**Core streaming method:**
- Processes LLM response chunks
- Generates `response.output_item.add_delta` events for streaming text
- Accumulates tool calls from chunks
- Creates final assistant message with tool calls
- Yields `response.output_item.done` events for completed items

### Tool Execution

#### `execute_tool(tool_name, args)`
- Retrieves tool from registry
- Executes with provided arguments
- Includes MLflow tracing and retry logic

#### `handle_tool_call(tool_call)`
- Parses tool call arguments
- Executes tool via `execute_tool()`
- Creates tool result message
- Appends to conversation history

### Thread Safety

#### Message Lock Decorator
```python
def with_message_lock(func):
    """Decorator for thread-safe message operations."""
```

**Protected methods:**
- `_get_messages_thread_safe()`
- `_append_message_thread_safe()`
- `_set_messages_thread_safe()`

### Streaming Events

#### Event Types Generated:
1. `response.output_item.add_delta` - Streaming text content
2. `response.output_item.done` - Completed tool calls and final text

#### Event Structure:
```python
# Text streaming
{
    "type": "response.output_item.add_delta",
    "item": {
        "type": "response.output_text.delta",
        "item_id": "uuid",
        "delta": "text chunk"
    }
}

# Tool completion
{
    "type": "response.output_item.done",
    "item": {
        "type": "function_call_output",
        "call_id": "uuid",
        "output": "result"
    }
}
```

## Tool Integration

### Unity Catalog Functions
```python
def create_tool_info(tool_spec, exec_fn_param=None):
    # Converts UC function specs to ToolInfo
    client = get_uc_function_client()
    function_result = client.execute_function(udf_name, kwargs)
    return function_result.value
```

### Vector Search Tools
```python
vs_tool = VectorSearchRetrieverTool(
    index_name="your_catalog.your_schema.your_index",
    tool_name="semantic_search",
    tool_description="Search for items based on semantic similarity",
    num_results=config.get("num_results", 5),
    query_type=config.get("query_type", "hybrid")
)
```

### Agent Endpoints
```python
@dataclass
class AgentEndpointExecutor:
    def execute(self, instruction: str) -> str:
        # Calls Databricks model serving endpoint
        response = self.client.chat.completions.create(
            model=self.config.endpoint,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.2,
            max_tokens=1000,
            stream=True
        )
```

## Error Handling & Resilience

### Retry Logic
- `@backoff.on_exception(backoff.expo, Exception, max_tries=3)` for tool execution
- `@backoff.on_exception(backoff.expo, openai.RateLimitError)` for LLM calls

### Safe Tool Loading
```python
def safe_tool_loader(loader_func, tool_name):
    try:
        result = loader_func()
        print(f"[INFO] Loaded: {tool_name}")
        return result
    except Exception as e:
        print(f"[WARNING] Failed to load {tool_name}: {e}")
        return None
```

## Memory Management

### Sliding Window
```python
# Keep only last 50 messages + system prompt to prevent memory issues
max_messages = 51  # 1 system + 50 conversation
if len(self.messages) > max_messages:
    self.messages = self.messages[:1] + self.messages[-50:]
```

## MLflow Integration

### Tracing
- `@mlflow.trace(span_type=SpanType.TOOL)` for tool execution
- `@mlflow.trace(span_type=SpanType.LLM)` for LLM calls
- Automatic OpenAI autologging

### Model Registration
```python
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=CONFIG["llm_endpoint"], tools=TOOL_INFOS)
mlflow.models.set_model(AGENT)
```

## Usage Examples

### Basic Tool Calling
```python
from agent import AgentManager

manager = AgentManager()
agent = manager.agent

# Tool calls happen automatically based on user queries
response = agent.predict(request)
```

### Streaming Responses
```python
for event in agent.predict_stream(request):
    if event.type == "response.output_item.add_delta":
        text = event.item.delta
        print(text, end='', flush=True)
    elif event.type == "response.output_item.done":
        # Handle completed items
        pass
```

## Configuration

### Required Setup
1. **Databricks authentication configured**
2. **Unity Catalog functions deployed** (update `your_catalog.your_schema.your_function_name` in CONFIG)
3. **Vector search indices created** (update `your_catalog.your_schema.your_index` in CONFIG)
4. **Model serving endpoints available** (update `your_model_endpoint` in CONFIG)

### Environment Variables
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- Or profile-based authentication

### Customization Required
Before using this agent, you MUST update the `CONFIG` dictionary in `agent.py`:

```python
CONFIG = {
    "llm_endpoint": "your_databricks_model_endpoint",  # Replace with your actual endpoint
    "tools": {
        "uc_functions": [
            "your_catalog.your_schema.your_uc_function"  # Replace with your UC functions
        ],
        "vector_search": [
            {
                "index_name": "your_catalog.your_schema.your_vector_index",  # Replace
                # ... other fields
            }
        ],
        "agent_endpoints": [
            {
                "endpoint": "your_expert_model_endpoint",  # Replace with your endpoints
                # ... other fields
            }
        ]
    }
}
```

## Performance Considerations

### Thread Safety
- All message operations are thread-safe
- Concurrent requests supported

### Memory Management
- Automatic message history pruning
- Configurable sliding window size

### Error Recovery
- Automatic retries with exponential backoff
- Graceful degradation when tools fail

## Extensibility

### Adding New Tools
1. Add configuration to `CONFIG["tools"]`
2. Implement tool executor function
3. Update `initialize_tools()` if needed

### Custom Message Handling
- Extend `Message` class for new formats
- Modify `prep_msgs_for_cc_llm()` for custom conversion logic

### Alternative LLM Providers
- Modify `call_llm()` method
- Update client initialization in `__init__`

## Troubleshooting

### Common Issues

1. **Tool Loading Failures**: Check Unity Catalog permissions and endpoint availability
2. **Placeholder Values Not Updated**: Remember to replace `your_catalog.your_schema.*` and `your_endpoint` with actual values
3. **Streaming Issues**: Verify event type handling in consuming code
4. **Memory Issues**: Adjust sliding window size in `_append_message_thread_safe()`
5. **Authentication Errors**: Ensure Databricks credentials are properly configured

### Debug Information
- Tool loading status printed during initialization
- MLflow traces available for request debugging
- Thread-safe message access for concurrent debugging

## Dependencies

- `mlflow>=3.0.0`
- `databricks-sdk>=0.65.0`
- `databricks-openai>=0.6.0`
- `backoff>=2.2.0`
- `pydantic>=2.11.0`
- `openai>=1.99.5`
