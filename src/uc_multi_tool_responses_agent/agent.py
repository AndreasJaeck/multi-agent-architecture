"""
UC Multi-Tool Responses Agent

This module implements a sophisticated tool-calling agent that integrates with:
- Unity Catalog functions for data processing
- Vector search for semantic similarity queries
- Agent endpoints for specialized domain expertise
- MLflow for tracing and model management

The agent provides streaming responses and handles complex multi-tool interactions.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import threading
import warnings
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
from dataclasses import dataclass

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client




# =============================================================================
# CONFIGURATION
# =============================================================================

# IMPORTANT: Replace all placeholder values (your_catalog.your_schema.*, your_endpoint, etc.)
# with your actual Databricks resources before using this agent
CONFIG = {
    "llm_endpoint": "your_model_endpoint",
    "max_iterations": 10,
    "system_prompt": """You are a helpful assistant that can use tools to solve problems and communicates in a disclosure free manner about what it will do next.

IMPORTANT: Focus only on the current user question. Do not reference or answer previous questions from the conversation history.

TOOL USAGE DECISION PROCESS:
1. First, determine if you need a tool to answer the question (you should have bias for using tools to answer questions)
2. If you need information from Agent SME experts, retrieve tabular data from Genie data room or get context from vector search, perform computations, call the appropriate tool.
3. After receiving tool results, evaluate if you now have enough information to answer.
4. IMPORTANT: If you have all the information needed, decide if synthesis is needed or if the information provided is sufficient.
5. Only call additional tools if you genuinely need more information.

KEY PRINCIPLE: After tool execution, ask yourself: "Do I have enough information to answer now?"
- YES → Provide final answer or leave the tool response as is
- NO → Call another tool if needed, or call the same tool with a different question

Be efficient - don't call tools unnecessarily, but use them when you need specific information or computations.""",

    # Tool configurations - centralized here
    "tools": {
        "uc_functions": [
            "your_catalog.your_schema.your_python_function",
            "your_catalog.your_schema.your_math_function"
        ],
        "vector_search": [
            {
                "index_name": "your_catalog.your_schema.your_vector_index",
                "tool_name": "semantic_search_tool",
                "description": "Search for items based on semantic similarity",
                "num_results": 5,
                "query_type": "hybrid"
            }
        ],
        "agent_endpoints": [
            {
                "endpoint": "your_expert_model_endpoint",
                "tool_name": "domain_expert_1",
                "description": "Specialized agent for domain-specific tasks"
            },
            {
                "endpoint": "your_expert_model_endpoint_2",
                "tool_name": "domain_expert_2",
                "description": "Another specialized agent for different domain tasks"
            }
        ]
    }
}


# =============================================================================
# MESSAGE SYSTEM
# =============================================================================

@dataclass
class Message:
    """
    Unified message representation supporting multiple formats.

    This class handles conversion between different message formats used in the system:
    - Responses API format (OpenAI-compatible format)
    - Chat Completion format (Langgraph/Claude-compatible format)

    It supports complex message types including tool calls, tool results, and streaming content.
    """

    # Common fields
    role: Optional[str] = None
    content: Optional[str] = None

    # Responses API format fields
    type: Optional[str] = None
    id: Optional[str] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None

    # Chat Completion format fields
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None

    def to_cc_format(self) -> dict[str, Any]:
        """Convert to Chat Completion format for LLM."""
        if self.role == "assistant" and self.tool_calls:
            return {
                "role": "assistant",
                "content": self.content or "",
                "tool_calls": self.tool_calls
            }
        elif self.type == "function_call_output" or self.role == "tool":
            return {
                "role": "tool",
                "content": self.output or self.content or "",
                "tool_call_id": self.call_id or self.tool_call_id
            }
        else:
            # Regular user/assistant/system messages
            return {
                "role": self.role,
                "content": self.content or ""
            }

    @classmethod
    def from_responses_format(cls, msg_dict: dict[str, Any]) -> 'Message':
        """Create Message from Responses API format."""
        # Filter to only known fields to avoid TypeError on unknown kwargs
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in msg_dict.items() if k in known_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_cc_format(cls, msg_dict: dict[str, Any]) -> 'Message':
        """Create Message from Chat Completion format."""
        return cls(**msg_dict)


# =============================================================================
# TOOL SYSTEM
# =============================================================================

class ToolInfo(BaseModel):
    """
    Represents a tool that the agent can execute.

    Each tool has three key components:
    - name: Unique identifier for the tool
    - spec: OpenAI-compatible function specification (JSON schema)
    - exec_fn: Python callable that implements the tool's logic

    This abstraction allows the agent to work with different types of tools
    (UC functions, vector search, agent endpoints) through a unified interface.
    """

    name: str
    spec: dict
    exec_fn: Callable


@dataclass
class AgentEndpointConfig:
    """
    Configuration for agent endpoint tools.

    Defines the settings needed to create a tool that calls another
    Databricks model serving endpoint as a specialized agent.
    """
    endpoint: str          # Databricks model serving endpoint name
    tool_name: str         # Name of the tool (used in function calls)
    description: str       # Description for tool routing
    system_prompt: Optional[str] = None  # Optional system prompt for the endpoint


class AgentEndpointExecutor:
    """
    Executor for calling Databricks serving endpoints as tools.

    This class handles the communication with Databricks model serving endpoints,
    allowing the main agent to delegate tasks to specialized sub-agents.

    Features:
    - Automatic retry with exponential backoff
    - Streaming response support
    - Error handling and fallbacks
    """

    def __init__(self, config: AgentEndpointConfig):
        self.config = config
        # Use the same workspace client pattern as the main agent
        self.client: OpenAI = WorkspaceClient().serving_endpoints.get_open_ai_client()

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def execute(self, instruction: str) -> str:
        """Execute agent endpoint with instruction and return response content."""
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": instruction})

        # Use streaming for tool responses to enable real-time streaming
        accumulated_content = ""
        try:
            for chunk in self.client.chat.completions.create(
                model=self.config.endpoint,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
                stream=True,
            ):
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        accumulated_content += delta.content
        except Exception as e:
            print(f"[WARNING] Error streaming from agent endpoint {self.config.endpoint}: {e}")
            return ""

        return accumulated_content.strip() if accumulated_content else ""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    """
    Creates a ToolInfo object from a tool specification.

    This function takes an OpenAI-style tool specification and creates a ToolInfo
    object with the appropriate execution function. For UC functions, it creates
    a wrapper that calls the Unity Catalog function execution client.
    """
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        client = get_uc_function_client()
        function_result = client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


# =============================================================================
# AGENT MANAGEMENT & INITIALIZATION
# =============================================================================

# Global variables for tools and agent
TOOL_INFOS = []
AGENT = None


class AgentManager:
    """
    Singleton manager for the ToolCallingAgent instance.

    This class provides lazy initialization and access to the main agent instance,
    ensuring that tools are loaded only once and the agent is properly configured.

    It implements the singleton pattern to avoid multiple agent instances and
    provides a clean interface for accessing agent tools and the agent itself.
    """

    def __init__(self):
        self._agent = None
        self._tools = None

    @property
    def tools(self):
        """Get the list of tool infos."""
        if self._tools is None:
            self.initialize()
        return self._tools

    @property
    def agent(self):
        """Get the agent instance."""
        if self._agent is None:
            self.initialize()
        return self._agent

    def initialize(self):
        """Initialize the agent and tools."""
        global TOOL_INFOS, AGENT

        if AGENT is None:
            main()  # This sets up TOOL_INFOS and AGENT

        self._tools = TOOL_INFOS
        self._agent = AGENT
        return self._agent


def safe_tool_loader(loader_func, tool_name):
    """Safely load a tool with consistent error handling."""
    try:
        result = loader_func()
        print(f"[INFO] Loaded: {tool_name}")
        return result
    except Exception as e:
        print(f"[WARNING] Failed to load {tool_name}: {e}")
        return None


def initialize_tools():
    """Initialize all tools from CONFIG in a clean, readable way."""
    tools = []

    # UC Functions - simple loop
    for func_name in CONFIG["tools"]["uc_functions"]:
        def load_uc_function():
            toolkit = UCFunctionToolkit(function_names=[func_name])
            for tool_spec in toolkit.tools:
                tools.append(create_tool_info(tool_spec))
            return True  # Indicate success

        safe_tool_loader(load_uc_function, f"UC function: {func_name}")

    # Vector Search - simple loop
    for vs_config in CONFIG["tools"]["vector_search"]:
        def load_vector_search():
            vs_tool = VectorSearchRetrieverTool(
                index_name=vs_config["index_name"],
                tool_name=vs_config["tool_name"],
                tool_description=vs_config["description"],
                num_results=vs_config.get("num_results", 5),
                query_type=vs_config.get("query_type", "hybrid")
            )
            tools.append(create_tool_info(vs_tool.tool, vs_tool.execute))
            return True  # Indicate success

        safe_tool_loader(load_vector_search, f"vector search tool: {vs_config['tool_name']}")

    # Agent Endpoints - simple loop
    for ae_config in CONFIG["tools"]["agent_endpoints"]:
        def load_agent_endpoint():
            config = AgentEndpointConfig(**ae_config)
            executor = AgentEndpointExecutor(config)

            # Create tool spec following OpenAI format
            tool_spec = {
                "type": "function",
                "function": {
                    "name": config.tool_name,
                    "description": config.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instruction": {
                                "type": "string",
                                "description": "The task or question for the agent to perform"
                            }
                        },
                        "required": ["instruction"]
                    }
                }
            }

            tools.append(create_tool_info(tool_spec, executor.execute))
            return True  # Indicate success

        safe_tool_loader(load_agent_endpoint, f"agent endpoint: {ae_config['tool_name']}")

    return tools


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class ToolCallingAgent(ResponsesAgent):
    """
    Main agent class implementing tool-calling capabilities.

    This class extends MLflow's ResponsesAgent to provide:
    - Integration with multiple tool types (UC functions, vector search, agent endpoints)
    - Thread-safe message handling for concurrent requests
    - Streaming response generation with real-time tool execution
    - Automatic retry logic with exponential backoff
    - MLflow tracing and model management

    The agent maintains conversation state and handles complex multi-turn interactions
    involving multiple tool calls within a single conversation.
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo] = None):
        """Initializes the ToolCallingAgent with tools."""
        # Validate llm_endpoint
        if not llm_endpoint:
            raise ValueError("llm_endpoint is required and cannot be empty")
        if not isinstance(llm_endpoint, str):
            raise TypeError("llm_endpoint must be a string")

        # Validate tools (optional)
        if tools is None:
            tools = []
        if not isinstance(tools, list):
            raise TypeError("tools must be a list")
        for i, tool in enumerate(tools):
            if not isinstance(tool, ToolInfo):
                raise TypeError(f"tools[{i}] must be a ToolInfo instance, got {type(tool)}")

        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        # Internal message list holds conversation state in Responses API format
        self.messages: list[dict[str, Any]] = []
        self._tools_dict = {tool.name: tool for tool in tools}
        # Thread lock for safe concurrent access to messages
        self._messages_lock = threading.RLock()

    def with_message_lock(func):
        """Decorator for thread-safe message operations."""
        def wrapper(self, *args, **kwargs):
            with self._messages_lock:
                return func(self, *args, **kwargs)
        return wrapper

    @with_message_lock
    def _get_messages_thread_safe(self) -> list[dict[str, Any]]:
        """Thread-safe access to messages list."""
        return self.messages  # Return reference since lock protects access

    @with_message_lock
    def _get_messages_copy_thread_safe(self) -> list[dict[str, Any]]:
        """Thread-safe access to a copy of messages list for modification."""
        return self.messages.copy()

    @with_message_lock
    def _append_message_thread_safe(self, message: dict[str, Any]) -> None:
        """Thread-safe append to messages list with sliding window to prevent memory issues."""
        self.messages.append(message)
        # Keep only last 50 messages + system prompt to prevent memory issues
        max_messages = 51  # 1 system prompt + 50 conversation messages
        if len(self.messages) > max_messages:
            # Keep system prompt (first message) + last 50 messages
            self.messages = self.messages[:1] + self.messages[-50:]

    @with_message_lock
    def _set_messages_thread_safe(self, messages: list[dict[str, Any]]) -> None:
        """Thread-safe replacement of messages list."""
        self.messages[:] = messages

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    def prep_msgs_for_cc_llm(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Responses API format messages to Chat Completion format."""
        cc_messages = []
        for msg_dict in messages:
            msg = Message.from_responses_format(msg_dict)
            cc_msg = msg.to_cc_format()

            # Skip messages with no content (except tool calls and final assistant)
            if not cc_msg.get("content") and not cc_msg.get("tool_calls"):
                continue

            cc_messages.append(cc_msg)

        return cc_messages

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            # Use the message preparation method (it's now implemented)
            prepared_messages = self.prep_msgs_for_cc_llm(self._get_messages_thread_safe())
            tool_specs = self.get_tool_specs()
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=prepared_messages,
                tools=tool_specs,
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                yield chunk_dict
                
    def output_to_responses_items_stream(self, chunks: Generator[dict[str, Any], None, None], current_messages: list, on_messages_updated: Callable[[list], None]) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Convert LLM chunks into ResponsesAgentStreamEvent objects and update messages via callback."""
        accumulated_content = ""
        current_tool_calls = {}
        text_item_id = str(uuid4())
        new_messages = current_messages.copy()  # Work with a copy

        # Process chunks
        for chunk in chunks:
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            # Handle content
            if content := delta.get("content"):
                accumulated_content += content
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.add_delta",
                    item=self.create_text_delta(content, text_item_id)
                )

            # Handle tool calls
            self._process_tool_call_deltas(delta, current_tool_calls)

            # Check if done
            if chunk.get("choices", [{}])[0].get("finish_reason"):
                break

        # Create assistant message
        assistant_msg = {"role": "assistant", "content": accumulated_content}
        if tool_calls := [tc for tc in current_tool_calls.values() if tc["function"]["name"]]:
            assistant_msg["tool_calls"] = tool_calls
        new_messages.append(assistant_msg)

        # Yield final events
        if accumulated_content:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(accumulated_content, str(uuid4()))
            )

        # Yield tool call events and update messages
        for tc in current_tool_calls.values():
            if tc["function"]["name"]:
                # Add to new messages
                new_messages.append({
                    "type": "function_call",
                    "id": str(uuid4()),
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"]
                })

                # Yield event
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_function_call_item(
                        id=str(uuid4()),
                        call_id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    )
                )

        # Update messages via callback
        on_messages_updated(new_messages)


    def _process_tool_call_deltas(self, delta: dict, current_tool_calls: dict) -> None:
        """Process tool call deltas and accumulate them."""
        for tc_delta in delta.get("tool_calls", []):
            index = tc_delta.get("index")
            if index is None:
                continue

            key = f"tool_call_{index}"

            # Initialize if new
            if key not in current_tool_calls:
                current_tool_calls[key] = {
                    "id": tc_delta.get("id", f"generated_{index}"),
                    "function": {"name": "", "arguments": ""},
                    "type": "function"
                }

            # Update function details
            if func_delta := tc_delta.get("function", {}):
                current_tool_calls[key]["function"]["name"] += func_delta.get("name", "")
                current_tool_calls[key]["function"]["arguments"] += func_delta.get("arguments", "")


    def handle_tool_call(self, tool_call: dict[str, Any]) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        self._append_message_thread_safe(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)


    def call_and_run_tools(self, max_iter: int = 10) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            messages = self._get_messages_thread_safe()
            last_msg = messages[-1]

            # Check if we're done (assistant message without tool calls)
            if last_msg.get("role") == "assistant" and not last_msg.get("tool_calls"):
                return

            # Execute pending tool call
            if last_msg.get("type") == "function_call":
                yield self.handle_tool_call(last_msg)
            else:
                # Generate LLM response
                # Get a copy of current messages for processing
                current_messages = self._get_messages_copy_thread_safe()

                # Define callback to update messages when ready
                def update_messages_callback(updated_messages: list) -> None:
                    self._set_messages_thread_safe(updated_messages)

                # Process streaming response - messages will be updated via callback
                yield from self.output_to_responses_items_stream(
                    chunks=self.call_llm(),
                    current_messages=current_messages,
                    on_messages_updated=update_messages_callback
                )

        # Max iterations reached
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4()))
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        new_messages = [{"role": "system", "content": CONFIG["system_prompt"]}] + [
            i.model_dump() for i in request.input
        ]
        self._set_messages_thread_safe(new_messages)
        yield from self.call_and_run_tools()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main initialization function for the UC Multi-Tool Agent.

    This function:
    1. Loads all configured tools from the CONFIG
    2. Creates and configures the ToolCallingAgent instance
    3. Registers the agent with MLflow for model management and tracing
    4. Returns the configured agent instance

    This is called automatically when the module is imported.
    """
    global TOOL_INFOS, AGENT

    # Initialize tools from CONFIG
    TOOL_INFOS = initialize_tools()
    print(f"[INFO] Total tools loaded: {len(TOOL_INFOS)}")

    # Log the model using MLflow
    mlflow.openai.autolog()
    AGENT = ToolCallingAgent(llm_endpoint=CONFIG["llm_endpoint"], tools=TOOL_INFOS)
    mlflow.models.set_model(AGENT)

    return AGENT


# Initialize the agent when module is imported or run directly
main()