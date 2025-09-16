"""
Self-contained supervisor agent.

This module defines an MLflow ResponsesAgent that:
- Exposes a set of domain experts (sub-agents) as OpenAI function tools for tool-calling
- Streams assistant token deltas and emits final aggregated output items compatible with FMAPI UIs
- Uses a simple in-process AgentRegistry to resolve and execute experts by name

Logical sections:
- Data models: ToolInfo, AgentCallArgs, AgentConfig
- Execution primitives: DomainAgentExecutor (invokes Databricks Model Serving endpoints synchronously)
- Registry: AgentRegistry (tool specs with rich descriptions, fuzzy resolution, listing)
- Agent: RegistryToolCallingAgent (ResponsesAgent subclass implementing the FMAPI event loop)

Notes and best practices (Databricks Mosaic AI Agent Framework):
- Use synchronous execution for tool calls; Databricks manages concurrency when deployed.
- Provide clear tool descriptions, inputs, and outputs to improve tool selection by the LLM.
- Avoid unsupported fields in tool specs for certain models (e.g., Claude does not support the "strict" flag).
"""

from __future__ import annotations

# ---------- Imports: stdlib, third-party, Databricks/MLflow SDKs ----------
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional
from uuid import uuid4
from functools import partial

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI

from pydantic import BaseModel, ValidationError

# Import agent configurations
try:
    from .agent_configs import DEFAULT_AGENTS, AgentConfig, SupervisorConfig
except ImportError:
    # Fallback to example config if agent_configs.py is not found
    from .agent_configs_example import DEFAULT_AGENTS, AgentConfig, SupervisorConfig



# ---------- Data models: tool and expert configuration ----------

@dataclass
class ToolInfo:
    """Lightweight container for a tool's spec and execution.

    - name: tool name as seen by the LLM
    - spec: OpenAI tool/function JSON schema (parameters, description)
    - exec_fn: synchronous callable that performs the tool logic
    """
    name: str
    spec: Dict[str, Any]
    exec_fn: Callable[..., Any]


# ---------- Defaults and simple builder for single-file usage ----------

# DEFAULT_AGENTS is imported from agent_configs.py or agent_configs_example.py

def build_registry(agent_configs: Optional[List[AgentConfig]] = None) -> AgentRegistry:
    """Construct an `AgentRegistry` from agent configurations using Serving executors.

    Parameters
    - agent_configs: List of AgentConfig objects to use. If None, uses DEFAULT_AGENTS.

    Returns an in-memory registry that the supervisor exposes as function tools.
    """
    if agent_configs is None:
        agent_configs = DEFAULT_AGENTS

    executors: List[DomainAgentExecutor] = [DomainAgentExecutor(ac) for ac in agent_configs]
    return AgentRegistry(executors)


# ---------- Typed argument schema for expert tool calls ----------
class AgentCallArgs(BaseModel):
    """Typed arguments for registry tools.

    The supervisor exposes experts as tools with a single required field:
    - instruction: the precise task for the expert to perform
    """
    instruction: str


# ---------- Local simple agent registry & executors ----------

class DomainAgentExecutor:
    """Executes a single expert by calling its Databricks Serving endpoint.

    Builds a simple 2-message conversation (system + user) and returns the
    assistant content from the first choice.
    """
    def __init__(self, config: AgentConfig, workspace_client: Optional[WorkspaceClient] = None):
        self.config = config
        self._workspace_client = workspace_client or WorkspaceClient()
        self._client: OpenAI = self._workspace_client.serving_endpoints.get_open_ai_client()

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def execute(self, query: str) -> str:
        """Invoke the expert endpoint synchronously and return the text content.

        Parameters
        - query: The user instruction that will be passed as the single user message

        Returns
        - The assistant message content string, or an empty string if unavailable
        """
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": query},
        ]

        resp = self._client.chat.completions.create(
            model=self.config.endpoint,
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
        )

        # Check if choices exist and inspect their structure
        if hasattr(resp, "choices") and resp.choices and len(resp.choices) > 0:
            choice = resp.choices[0]
            # Check if choice has message
            if hasattr(choice, "message"):
                message = choice.message
                # Try to get all attributes and their values
                for attr in dir(message):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(message, attr)
                        except Exception as e:
                            pass

                # Try multiple ways to extract content
                content = getattr(message, "content", None)
                if content is not None and len(str(content).strip()) > 0:
                    return str(content)

                # Check for other possible content fields
                for attr in ['text', 'response', 'output', 'result', 'answer']:
                    if hasattr(message, attr):
                        value = getattr(message, attr)
                        if value and isinstance(value, str) and len(value.strip()) > 0:
                            return value

            else:
                # Maybe the content is directly in the choice?
                for attr in dir(choice):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(choice, attr)
                            if value is not None and isinstance(value, str) and len(value.strip()) > 0:
                                return value
                        except Exception as e:
                            pass
        else:
            # Try alternative response format: check for messages directly
            if hasattr(resp, "messages") and resp.messages and len(resp.messages) > 0:
                for msg in resp.messages:
                    if msg.get("role") == "assistant" and "content" in msg:
                        content = msg["content"]
                        if content and isinstance(content, str) and len(content.strip()) > 0:
                            return content

        return ""

# ---------- In-memory agent registry: specs, listing, fuzzy resolution ----------
class AgentRegistry:
    """In-process registry of experts; also provides OpenAI tool specs.

    Responsibilities:
    - Map expert names to executors
    - Fuzzy name resolution (exact, case-insensitive, close match)
    - Generate OpenAI tool/function specs from AgentConfig metadata
    """
    def __init__(self, agents: List[DomainAgentExecutor]):
        self._by_name: Dict[str, DomainAgentExecutor] = {a.config.name: a for a in agents}
        self._configs: List[AgentConfig] = [a.config for a in agents]

    def has(self, name: str) -> bool:
        """Return True if an expert with the given name exists (case-sensitive)."""
        return name in self._by_name

    def get(self, name: str) -> DomainAgentExecutor:
        """Get the executor for an expert by exact name.

        Raises KeyError if the expert does not exist.
        """
        return self._by_name[name]

    def list_configs(self) -> List[AgentConfig]:
        """List the immutable `AgentConfig` objects for all registered experts."""
        return list(self._configs)

    def resolve(self, name: str) -> Optional[str]:
        """Resolve a possibly inexact expert name to the canonical registered name.

        Resolution order: exact match -> case-insensitive match -> close string match.
        Returns the resolved name, or None if no reasonable match is found.
        """
        if name in self._by_name:
            return name
        for k in self._by_name.keys():
            if k.lower() == name.lower():
                return k
        try:
            import difflib as _difflib

            candidates = _difflib.get_close_matches(name, list(self._by_name.keys()), n=1, cutoff=0.6)
            return candidates[0] if candidates else None
        except Exception:
            return None

    def tool_specs(self) -> List[dict]:
        """Create OpenAI-compatible function tool specs for each expert.

        Descriptions include the expert description plus optional capabilities and domain
        hints to help the LLM route calls appropriately. The parameters schema is kept
        minimal (a single required `instruction` string) to keep usage predictable.
        """
        specs: List[dict] = []
        for cfg in self._configs:
            desc_parts = [cfg.description]
            if getattr(cfg, "capabilities", None):
                desc_parts.append(f"Capabilities: {cfg.capabilities}")
            if getattr(cfg, "domain", None):
                desc_parts.append(f"Domain: {cfg.domain}")
            description = " | ".join([p for p in desc_parts if p])
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": cfg.name,
                        # Keep descriptions concise. Good tool docs improve tool selection.
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "instruction": {
                                    "type": "string",
                                    "description": "Precise task for the expert to perform",
                                }
                            },
                            "required": ["instruction"],
                        },
                    },
                }
            )
        return specs


# ---------- Supervisor agent: orchestrates tool-calls and streaming ----------
class RegistryToolCallingAgent(ResponsesAgent):
    """Supervisor agent exposing experts as tools and orchestrating tool calls.

    This agent:
    - Presents each expert in the registry as a function tool to the LLM
    - Streams assistant deltas as they arrive
    - Executes tools when requested and feeds tool outputs back into the loop
    - Emits final output_item.done events compatible with ResponsesAgent UIs
    """

    def __init__(
        self,
        llm_endpoint: str,
        registry: AgentRegistry,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.registry = registry
        self.workspace_client: WorkspaceClient = WorkspaceClient()
        self.model_serving_client: OpenAI = self.workspace_client.serving_endpoints.get_open_ai_client()

        self.system_prompt = (
            system_prompt
            or "You are a supervisor that delegates work to specialist tools (agents). "
            "Avoid answering questions without consulting specialist agents/tools. "
            "Decline answering questions that are unrelated to informations provided by the specialist agents/tools."
            "As you work, stream your chain-of-thought in a disclosure manner as deltas "
            "Provide clear, concise outputs that are user-appropriate."
        )

        # Build tool registry from AgentRegistry specs
        self._tools_dict: Dict[str, ToolInfo] = {}
        for tool_spec in self._build_tool_specs_from_registry():
            tool_name = tool_spec["function"]["name"]
            self._tools_dict[tool_name] = ToolInfo(
                name=tool_name,
                spec=tool_spec,
                exec_fn=partial(self._exec_tool_by_name, tool_name),
            )

    # ---------- Tools ----------
    def _build_tool_specs_from_registry(self) -> List[Dict[str, Any]]:
        """Return tool specs for all registry experts with a strict, typed schema.

        Also removes unsupported flags (e.g., "strict") for Claude models and swaps in
        the Pydantic-generated JSON schema from `AgentCallArgs` to ensure consistent
        validation and clear parameter docs.
        """
        specs = self.registry.tool_specs()
        # Claude models do not support strict flag if present
        for s in specs:
            s.get("function", {}).pop("strict", None)
            # Replace parameters with a richer, typed schema while keeping backward compatibility
            params = AgentCallArgs.model_json_schema()
            # Ensure "instruction" remains required
            s["function"]["parameters"] = params
        return specs

    def get_tool_specs(self) -> List[Dict[str, Any]]:
        """Expose the current tool specifications in OpenAI function-tool format."""
        return [t.spec for t in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool by name with validated args under an MLflow TOOL span."""
        return self._tools_dict[tool_name].exec_fn(**args)

    # ---------- Helpers ----------
    def _exec_tool_by_name(self, tool_name: str, **kwargs: Any) -> Any:
        try:
            args = AgentCallArgs.model_validate(kwargs)
        except ValidationError as ve:
            return {
                "status": "error",
                "data": {"validation_errors": json.loads(ve.json())},
                "meta": {"agent_name": tool_name},
            }
        resolved = self.registry.resolve(tool_name) or tool_name
        result = self.registry.get(resolved).execute(str(args.instruction))

        # Check if the domain agent returned empty data
        if not result or result.strip() == "":
            return {
                "status": "error",
                "data": {"error": f"Domain agent '{resolved}' returned empty response. The agent may not be properly configured or available."},
                "meta": {"agent_name": resolved},
            }

        return {
            "status": "ok",
            "data": result,
            "meta": {"agent_name": resolved},
        }

    # ---------- Message conversion (Responses -> Chat Completions) ----------
    def _responses_to_cc(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert Responses API message items into Chat Completions format.

        Handles function_call, function_call_output, reasoning, and message types,
        filtering out unsupported fields for Serving Chat Completions models.
        """
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",  # claude models cannot accept empty content
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message.get("content"), list):
            return [
                {"role": message["role"], "content": content["text"]}
                for content in message["content"]
            ]
        elif msg_type == "reasoning":
            return [{"role": "assistant", "content": json.dumps(message["summary"])}]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []

    def prep_msgs_for_llm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter/convert messages to be compatible with Serving Chat Completions.

        This ensures we only send supported roles/fields and convert Responses items
        into the chat-completions message format expected by the endpoint.
        """
        chat_msgs: List[Dict[str, Any]] = []
        for msg in messages:
            chat_msgs.extend(self._responses_to_cc(msg))
        return chat_msgs

    # ---------- LLM streaming ----------
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Generator[Dict[str, Any], None, None]:
        """Yield raw streaming chunks from Databricks Serving (as dicts).

        The returned items mirror OpenAI SDK streaming chunk dicts; callers should
        read `choices[0]["delta"]` for incremental content and tool_calls.

        Parameters
        - messages: The conversation messages to send
        - tools: Optional list of tools to provide; defaults to self.get_tool_specs() if None
        """
        tool_specs = tools if tools is not None else self.get_tool_specs()
        for chunk in self.model_serving_client.chat.completions.create(
            model=self.llm_endpoint,
            messages=self.prep_msgs_for_llm(messages),
            tools=tool_specs,
            stream=True,
        ):
            yield chunk.to_dict()

    # ---------- Tool execution handling ----------
    def handle_tool_calls(
        self, messages: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]]
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Execute tool calls and stream their outputs back to the client.

        Each tool result is appended to the conversation as a `tool` role message and
        emitted as a `response.output_item.done` function_call_output event for FMAPI.
        """
        for tool_call in tool_calls:
            function = tool_call["function"]
            args = json.loads(function["arguments"]) if isinstance(function.get("arguments"), str) else (function.get("arguments") or {})
            raw_result = self.execute_tool(tool_name=function["name"], args=args)

            # Handle the result appropriately
            if isinstance(raw_result, dict) and raw_result.get("status") == "error":
                # Tool returned an error - report it clearly
                result_payload = raw_result
                result_str = json.dumps(result_payload)
                messages.append({"role": "tool", "content": result_str, "tool_call_id": tool_call["id"]})
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_function_call_output_item(
                        tool_call["id"],
                        result_str,
                    ),
                )
            elif isinstance(raw_result, dict):
                result_payload = raw_result
                result_str = json.dumps(result_payload)
                messages.append({"role": "tool", "content": result_str, "tool_call_id": tool_call["id"]})
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_function_call_output_item(
                        tool_call["id"],
                        result_str,
                    ),
                )
            else:
                result_payload = {"status": "ok", "data": str(raw_result), "meta": {"agent_name": function["name"]}}
                result_str = json.dumps(result_payload)
                messages.append({"role": "tool", "content": result_str, "tool_call_id": tool_call["id"]})
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_function_call_output_item(
                        tool_call["id"],
                        result_str,
                    ),
                )

    # ---------- Main loop ----------
    def call_and_run_tools(
        self,
        messages: List[Dict[str, Any]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Main fmapi loop: stream LLM deltas, then execute any tool calls.

        Continues until no tool calls are returned or max_iter is reached.
        """
        for iteration in range(max_iter):
            last_msg = messages[-1]
            if tool_calls := last_msg.get("tool_calls", None):
                yield from self.handle_tool_calls(messages, tool_calls)
            elif last_msg.get("role", None) == "assistant" and iteration > 0:
                # Only return if this is an assistant message from a main LLM call (not planning)
                return
            else:
                llm_content = ""
                tool_calls: List[Dict[str, Any]] = []
                # Use a stable item id for all deltas and the final done item
                stream_item_id: str = str(uuid4())
                for chunk in self.call_llm(messages):
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", None)
                    if tc := delta.get("tool_calls"):
                        # Tool call arguments may arrive in multiple streaming chunks.
                        # Accumulate them into a single `tool_calls[0]['function']['arguments']` string.
                        if not tool_calls:
                            tool_calls = tc
                        else:
                            tool_calls[0]["function"]["arguments"] += tc[0]["function"]["arguments"]
                    elif content is not None:
                        llm_content += content
                        yield ResponsesAgentStreamEvent(
                            **self.create_text_delta(content, item_id=stream_item_id)
                        )
                llm_output = {"role": "assistant", "content": llm_content, "tool_calls": tool_calls}
                messages.append(llm_output)

                if llm_output["content"]:
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_text_output_item(
                            llm_output["content"], stream_item_id
                        ),
                    )
                if tool_calls := llm_output.get("tool_calls", None):
                    for tool_call in tool_calls:
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=self.create_function_call_item(
                                str(uuid4()),
                                tool_call["id"],
                                tool_call["function"]["name"],
                                tool_call["function"]["arguments"],
                            ),
                        )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", "max-iter"),
        )

    # ---------- Predict APIs ----------
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming predict API that aggregates output items.

        Returns a `ResponsesAgentResponse` containing only `output_item.done` items.
        """
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming predict API that yields FMAPI events as they occur.

        First does a planning turn without tools to disclose next steps, then proceeds with tool-enabled execution.
        """
        experts_md = "\n".join([f"- {c.name}: {c.description}" for c in self.registry.list_configs()])
        system = f"{self.system_prompt}\n\nAvailable experts:\n{experts_md}"
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}] + [
            i.model_dump() for i in request.input
        ]

        # ---------- Planning turn: stream a brief plan without tools ----------
        # Modify the last user message to include planning instruction
        if messages and messages[-1]["role"] == "user":
            original_content = messages[-1]["content"]
            messages[-1]["content"] = f"Respond with exactly one sentence: what tool will you use for this task? Original query: {original_content}"
        planning_content = ""
        stream_item_id = str(uuid4())
        max_planning_length = 200  # Limit planning content to prevent rambling
        for chunk in self.call_llm(messages, tools=[]):  # No tools for planning
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", None)
            if content is not None:
                planning_content += content
                yield ResponsesAgentStreamEvent(
                    **self.create_text_delta(content, item_id=stream_item_id)
                )
                # Stop planning if we've generated enough content or hit a period (end of sentence)
                if len(planning_content) > max_planning_length or planning_content.strip().endswith('.'):
                    break
        if planning_content:
            planning_response = {"role": "assistant", "content": planning_content}
            messages.append(planning_response)
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(planning_content, stream_item_id),
            )
        # Reset the user message for the main turn
        if messages and messages[-2]["role"] == "user":  # The planning response is now last
            messages[-2]["content"] = original_content

        # ---------- Proceed to tool-enabled execution ----------
        yield from self.call_and_run_tools(messages)


# ---------- Factory helpers: construct a configured supervisor ----------
def create_registry_supervisor_agent(
    llm_endpoint: str,
    registry: AgentRegistry,
    system_prompt: Optional[str] = None,
) -> RegistryToolCallingAgent:
    """Convenience factory to create a RegistryToolCallingAgent."""
    return RegistryToolCallingAgent(llm_endpoint=llm_endpoint, registry=registry, system_prompt=system_prompt)


# ---------- MLflow integration & model export ----------
cfg = SupervisorConfig()
registry = build_registry()
AGENT = create_registry_supervisor_agent(cfg.llm_endpoint, registry)
# Enable OpenAI autologging so LLM/tool calls are captured in MLflow traces when run locally
mlflow.openai.autolog()
mlflow.models.set_model(AGENT)
