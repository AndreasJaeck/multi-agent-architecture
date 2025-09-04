"""
Multi-Agent Supervisor Architecture

A LangGraph-based multi-agent system that orchestrates specialized agents
through a supervisor pattern, with MLflow integration for deployment.
"""

# ===============================
# IMPORTS & DEPENDENCIES  
# ===============================
import functools
import uuid
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generator, Optional, Dict, List

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from pydantic import BaseModel


# ===============================
# CONFIGURATION & CONSTANTS
# ===============================

class SupervisorConfig:
    """Centralized configuration for the multi-agent system."""
    
    # Core settings
    MAX_ITERATIONS = 3
    CONTEXT_MAX_CHARS = 2000
    
    # LLM Configuration
    LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
    
    # Agent Endpoints
    BASF_DATA_ENDPOINT = "genie_multi_agent_basf"
    GENOMICS_TOOLS_ENDPOINT = "genie_multi_agent_basf_v2"
    
    # Decision options
    FINISH_OPTION = "FINISH"


@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    description: str
    endpoint: str
    service_name: str
    capabilities: str


class ConfigManager:
    """Manages agent configurations and system setup."""
    
    def __init__(self):
        self.config = SupervisorConfig()
        self._setup_agents()
        self._setup_llm()
    
    def _setup_agents(self):
        """Initialize agent configurations."""
        self.agents = {
            "BASF_Data": AgentConfig(
                description=(
                    "The BASF Data assistant has access to Gbuilt data from BASF and can perform vector search "
                    "on Valona market insights data. Use this agent for queries about BASF-related data, chemical information, "
                    "or when searching through BASF documentation and datasets."
                ),
                endpoint=self.config.BASF_DATA_ENDPOINT,
                service_name="BASF Data Assistant",
                capabilities="chemical data analysis and market insights"
            ),
            "Genomics_Tools": AgentConfig(
                description=(
                    "The Genomics Tools assistant has access to patient genomics data and can perform "
                    "mathematical computations and execute Python code. Use this agent for genomics analysis, "
                    "data processing, calculations, or when computational tools are needed."
                ),
                endpoint=self.config.GENOMICS_TOOLS_ENDPOINT,
                service_name="Genomics Tools Assistant",
                capabilities="genomics analysis and computational tools"
            )
        }
        
        # Create agent instances
        self.agent_instances = {}
        for name, config in self.agents.items():
            model = ChatDatabricks(endpoint=config.endpoint)
            self.agent_instances[name] = create_react_agent(model, [])
    
    def _setup_llm(self):
        """Initialize the supervisor LLM."""
        self.llm = ChatDatabricks(endpoint=self.config.LLM_ENDPOINT_NAME)
    
    @cached_property
    def formatted_descriptions(self) -> str:
        """Get formatted description string for prompts."""
        return "\n".join(f"- {name}: {config.description}" for name, config in self.agents.items())
    
    @cached_property
    def routing_options(self) -> List[str]:
        """Get all routing options including FINISH."""
        return [self.config.FINISH_OPTION] + list(self.agents.keys())


# ===============================
# DATA MODELS & TYPES
# ===============================

class NextNodeDecision(BaseModel):
    """Decision model for supervisor routing."""
    next_node: str
    reasoning: str


class AgentState(ChatAgentState):
    """Extended state for multi-agent workflow."""
    next_node: str
    iteration_count: int


# ===============================
# UTILITY CLASSES
# ===============================

class MessageHandler:
    """Handles message processing and context building."""
    
    # Pre-generate UUID pool for better performance (avoids crypto calls during runtime)
    _uuid_pool = [str(uuid.uuid4()) for _ in range(100)]
    _uuid_index = 0
    
    @classmethod
    def _get_next_uuid(cls) -> str:
        """Get next UUID from pre-generated pool for performance."""
        uuid_val = cls._uuid_pool[cls._uuid_index]
        cls._uuid_index = (cls._uuid_index + 1) % len(cls._uuid_pool)
        return uuid_val
    
    @staticmethod
    def build_agent_context(agent_responses: List[Dict], max_chars: int = 2000) -> str:
        """Build context with smart truncation that preserves important information."""
        if not agent_responses:
            return ""
        
        # Use list to build parts, then join (more efficient than string concatenation)
        context_parts = ["\n\n=== RECENT AGENT RESPONSES ===\n"]
        response_parts = []
        total_chars = len(context_parts[0])
        
        # Start from most recent and work backwards
        for i, response in enumerate(reversed(agent_responses)):
            agent_name = response.get('name', 'unknown')
            content = response.get('content', '')
            
            # For the most recent response, always include more content
            if i == 0:
                response_text = f"\nMOST RECENT - {agent_name}:\n{content}\n"
            else:
                # For older responses, include summary or key points
                truncated_content = content[:300] + ('...' if len(content) > 300 else '')
                response_text = f"\n{agent_name} (earlier):\n{truncated_content}\n"
            
            if total_chars + len(response_text) > max_chars and i > 0:
                response_parts.append(f"\n... ({len(agent_responses) - i} earlier responses omitted)\n")
                break
                
            response_parts.append(response_text)
            total_chars += len(response_text)
        
        # Reverse response parts to maintain chronological order
        response_parts.reverse()
        
        # Combine all parts efficiently
        all_parts = context_parts + response_parts + ["\nIMPORTANT: If ANY response completely answers the user's question, choose FINISH."]
        return ''.join(all_parts)
    
    @staticmethod
    def extract_message_content(result: Dict) -> str:
        """Extract content from agent result with robust handling."""
        if "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            
            # Handle different message formats
            if hasattr(last_message, "content"):
                content = last_message.content
            elif isinstance(last_message, dict) and "content" in last_message:
                content = last_message["content"]
            else:
                content = str(last_message)
            
            # Ensure content is a string
            return str(content) if content else "No response from agent"
        
        return "No response from agent"
    
    @staticmethod
    def create_agent_message(content: str, name: str) -> Dict:
        """Create a standardized agent message."""
        return {
            "role": "assistant",
            "content": content,
            "name": name,
        }
    
    @classmethod
    def create_chat_agent_message(cls, msg: Any) -> ChatAgentMessage:
        """Create ChatAgentMessage from various message formats - optimized for performance."""
        if isinstance(msg, dict):
            message_dict = {
                "id": cls._get_next_uuid(),
                "role": msg.get("role", "assistant"),
                "content": str(msg.get("content", ""))
            }
            if "name" in msg:
                message_dict["name"] = msg["name"]
            return ChatAgentMessage(**message_dict)
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            return ChatAgentMessage(
                id=cls._get_next_uuid(),
                role=msg.role,
                content=str(msg.content)
            )
        else:
            return ChatAgentMessage(
                id=cls._get_next_uuid(),
                role="assistant",
                content=str(msg)
            )


class ErrorHandler:
    """Handles error processing and user-friendly error messages."""
    
    def __init__(self, config_manager: ConfigManager):
        self.agents = config_manager.agents
        
        # Pre-compile error message templates for performance
        self._error_message_templates = {}
        for agent_name, config in self.agents.items():
            self._error_message_templates[agent_name] = (
                f"❌ **{config.service_name} Unavailable**\n\n"
                f"I'm currently unable to access the service that handles {config.capabilities}.\n\n"
                f"**Please try again in a few moments.** If the issue persists, contact your administrator.\n\n"
            )
        
        # Fallback template for unknown agents
        self._fallback_template = (
            "❌ **{} Assistant Unavailable**\n\n"
            "I'm currently unable to access the service that handles specialized analysis.\n\n"
            "**Please try again in a few moments.** If the issue persists, contact your administrator.\n\n"
        )
    
    def create_error_message(self, agent_name: str, error: Exception) -> str:
        """Create a user-friendly error message - optimized for performance."""
        # Use pre-compiled template if available
        if agent_name in self._error_message_templates:
            base_message = self._error_message_templates[agent_name]
        else:
            # Use fallback template for unknown agents
            base_message = self._fallback_template.format(agent_name.replace('_', ' '))
        
        # Append technical details (truncated for performance)
        error_str = str(error)
        technical_details = error_str[:150] + "..." if len(error_str) > 150 else error_str
        
        return base_message + f"*Technical details: {technical_details}*"
    
    @staticmethod
    def has_error_messages(messages: List[Dict]) -> bool:
        """Check if any messages contain error indicators."""
        return any("❌" in msg.get("content", "") for msg in messages)


# ===============================
# CORE BUSINESS LOGIC
# ===============================

class SupervisorLogic:
    """Handles supervisor decision-making logic."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager.config
        self.config_manager = config_manager
        self.message_handler = MessageHandler()
        
        # Pre-compile the base system prompt template for performance
        self._base_system_prompt = (
            f"You are a supervisor managing these assistants:\n"
            f"{self.config_manager.formatted_descriptions}\n\n"
            f"Given the conversation history, respond with the assistant to act next.\n"
            f"DECISION RULES:\n"
            f"1. If ANY agent has provided a complete answer to the user's question, respond with FINISH\n"
            f"2. Only call another agent if the current answer is incomplete or incorrect\n"
            f"3. Do NOT call additional agents just to verify or add to a complete answer"
        )
        
        # Pre-compile the supervisor chain for performance
        self._preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": self._create_full_prompt(state)}] + state["messages"]
        )
        self._supervisor_chain = self._preprocessor | self.config_manager.llm.with_structured_output(NextNodeDecision)
    
    def should_finish_iteration_limit(self, count: int) -> bool:
        """Check if we've exceeded iteration limits."""
        return count > self.config.MAX_ITERATIONS
    
    def should_finish_error(self, agent_responses: List[Dict]) -> bool:
        """Check if the last response was an error."""
        if agent_responses:
            latest_response = agent_responses[-1]
            return "❌" in latest_response.get("content", "")
        return False
    
    def should_finish_same_agent(self, agent_responses: List[Dict], next_node: str) -> bool:
        """Check if supervisor is routing back to the same agent."""
        return bool(agent_responses and next_node == agent_responses[-1].get("name"))
    
    def _create_full_prompt(self, state: Dict) -> str:
        """Create the complete system prompt with context - optimized for performance."""
        # Extract agent responses once and reuse
        messages = state.get("messages", [])
        agent_responses = [
            msg for msg in messages 
            if msg.get("role") == "assistant" and msg.get("name")
        ]
        
        # Build context only if there are agent responses
        if agent_responses:
            agent_context = self.message_handler.build_agent_context(
                agent_responses, self.config.CONTEXT_MAX_CHARS
            )
            return self._base_system_prompt + agent_context
        else:
            return self._base_system_prompt
    
    def make_decision(self, state: Dict) -> Dict:
        """Make supervisor routing decision - optimized for performance."""
        count = state.get("iteration_count", 0) + 1
        
        # Check max iterations
        if self.should_finish_iteration_limit(count):
            return {"next_node": self.config.FINISH_OPTION}
        
        # Extract agent responses once and reuse
        messages = state.get("messages", [])
        agent_responses = [
            msg for msg in messages 
            if msg.get("role") == "assistant" and msg.get("name")
        ]
        
        # Check if the last response was an error
        if self.should_finish_error(agent_responses):
            return {"next_node": self.config.FINISH_OPTION}
        
        # Get LLM decision using pre-compiled chain
        result = self._supervisor_chain.invoke(state)
        
        # Validate the routing option
        if result.next_node not in self.config_manager.routing_options:
            result.next_node = self.config.FINISH_OPTION
        
        # Prevent routing back to same agent
        if self.should_finish_same_agent(agent_responses, result.next_node):
            return {"next_node": self.config.FINISH_OPTION}
        
        return {
            "iteration_count": count,
            "next_node": result.next_node
        }


class AgentOrchestrator:
    """Handles agent execution and coordination."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.message_handler = MessageHandler()
        self.error_handler = ErrorHandler(config_manager)
        
        # Pre-compile system prompts for performance
        self._error_system_prompt = (
            "The user's request could not be completed because one or more services are unavailable. "
            "Provide a concise, helpful response that acknowledges the service issue and suggests the user try again later. "
            "Do not attempt to answer the original question if the required service failed."
        )
        self._normal_system_prompt = (
            "Using only the content in the messages, respond to the user's question using the answer given by the other agents."
        )
        
        # Pre-compile final answer chains for performance  
        self._error_preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": self._error_system_prompt}] + state["messages"]
        )
        self._normal_preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": self._normal_system_prompt}] + state["messages"]
        )
        self._error_chain = self._error_preprocessor | self.config_manager.llm
        self._normal_chain = self._normal_preprocessor | self.config_manager.llm
    
    def execute_agent(self, state: Dict, agent_name: str) -> Dict:
        """Execute a specific agent and handle the response."""
        agent = self.config_manager.agent_instances[agent_name]
        
        try:
            result = agent.invoke(state)
            content = self.message_handler.extract_message_content(result)
            
            return {
                "messages": [self.message_handler.create_agent_message(content, agent_name)]
            }
            
        except Exception as e:
            error_content = self.error_handler.create_error_message(agent_name, e)
            return {
                "messages": [self.message_handler.create_agent_message(error_content, agent_name)]
            }
    
    def create_final_answer(self, state: Dict) -> Dict:
        """Create the final synthesized response - optimized for performance."""
        messages = state.get("messages", [])
        
        # Use pre-compiled chains based on error presence
        if self.error_handler.has_error_messages(messages):
            chain = self._error_chain
        else:
            chain = self._normal_chain
            
        return {"messages": [chain.invoke(state)]}


# ===============================
# WORKFLOW CONSTRUCTION
# ===============================

class WorkflowBuilder:
    """Builds and configures the LangGraph workflow."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.supervisor_logic = SupervisorLogic(config_manager)
        self.agent_orchestrator = AgentOrchestrator(config_manager)
    
    def _supervisor_agent(self, state: Dict) -> Dict:
        """Supervisor agent function wrapper."""
        return self.supervisor_logic.make_decision(state)
    
    def _create_agent_node(self, agent_name: str):
        """Create an agent node function."""
        return functools.partial(
            self.agent_orchestrator.execute_agent,
            agent_name=agent_name
        )
    
    def _final_answer(self, state: Dict) -> Dict:
        """Final answer function wrapper."""
        return self.agent_orchestrator.create_final_answer(state)
    
    def build_workflow(self) -> CompiledStateGraph:
        """Build and compile the complete workflow."""
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        for agent_name in self.config_manager.agents.keys():
            agent_node = self._create_agent_node(agent_name)
            workflow.add_node(agent_name, agent_node)
        
        # Add supervisor and final answer nodes
        workflow.add_node("supervisor", self._supervisor_agent)
        workflow.add_node("final_answer", self._final_answer)
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Add edges: workers always report back to supervisor
        for agent_name in self.config_manager.agents.keys():
            workflow.add_edge(agent_name, "supervisor")
        
        # Add conditional edges from supervisor
        routing_dict = {
            name: name for name in self.config_manager.agents.keys()
        }
        routing_dict[self.config_manager.config.FINISH_OPTION] = "final_answer"
        
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["next_node"],
            routing_dict
        )
        
        # Final edge to END
        workflow.add_edge("final_answer", END)
        
        return workflow.compile()


# ===============================
# MLFLOW INTEGRATION
# ===============================

class LangGraphChatAgent(ChatAgent):
    """MLflow-compatible chat agent wrapper for LangGraph workflows."""
    
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent
        self.message_handler = MessageHandler()
    
    def _prepare_request(self, messages: List[ChatAgentMessage]) -> Dict:
        """Prepare request format for the agent - optimized for performance."""
        # Pre-allocate list for better memory efficiency
        message_list = []
        message_list.extend(m.model_dump_compat(exclude_none=True) for m in messages)
        return {"messages": message_list}
    
    def _stream_agent_messages(self, request: Dict) -> Generator[ChatAgentMessage, None, None]:
        """Stream agent messages one by one."""
        for event in self.agent.stream(request, stream_mode="updates"):
            yield from (
                self.message_handler.create_chat_agent_message(msg)
                for node_data in event.values()
                for msg in node_data.get("messages", [])
            )
    
    def predict(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        """Predict method for MLflow compatibility."""
        request = self._prepare_request(messages)
        response_messages = list(self._stream_agent_messages(request))
        return ChatAgentResponse(messages=response_messages)
    
    def predict_stream(
        self,
        messages: List[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[Dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        """Streaming predict method for MLflow compatibility."""
        request = self._prepare_request(messages)
        
        for message in self._stream_agent_messages(request):
            yield ChatAgentChunk(delta=message)


# ===============================
# MODULE INITIALIZATION
# ===============================

# Initialize the system
config_manager = ConfigManager()
workflow_builder = WorkflowBuilder(config_manager)
multi_agent = workflow_builder.build_workflow()

# MLflow setup
mlflow.langchain.autolog()
AGENT = LangGraphChatAgent(multi_agent)
mlflow.models.set_model(AGENT)

# Export key components for external use
__all__ = [
    'AGENT',
    'SupervisorConfig',
    'ConfigManager', 
    'WorkflowBuilder',
    'LangGraphChatAgent',
    'multi_agent'
]