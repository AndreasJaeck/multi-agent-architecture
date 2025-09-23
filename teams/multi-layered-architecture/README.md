# Multi-Agent Architecture

## Hackathon Goals

**Core Objectives:**
- Find the optimal approach to orchestrate multiple agents

**Main Challenges:**
- **Runtime Efficiency**: Minimize LLM calls in chat completions, implement progressive disclosure for transparency, leverage Responses API for lightweight tool (agent) calls during streaming responses
- **Intelligent Routing**: Develop clear, descriptive agent integration patterns for effective request routing

## Architecture Documentation

📚 **[Multi-Agent Architectures Overview](../../docs/MultiAgentArchitectures.md)**
- Hierarchical, pipeline, and mesh architecture patterns
- Design principles and implementation guides
- Flow charts and architectural comparisons


## Example Notebooks

📓 **Core Examples:**
- **[FMAPI Streaming Supervisor (new)](../../notebooks/06-supervisor-streaming-disclosure.py)** - Main FMAPI implementation with streaming
- **[MLflow Responses API (new)](../../notebooks/MLFlow_Responses_API/)** - Response API patterns

📓 **Alternative Architectures:**
- **[LangGraph Multi-Agent (legacy)](../../notebooks/03-langgraph-multiagent-genie-pat.py)** - LangGraph-based implementation
- **[Supervisor of Supervisors (legacy)](../../notebooks/05-supervisor-of-supervisors.py)** - Hierarchical supervisor pattern


## Supervisor of Supervisors

🤖 **[FMAPI Supervisor Agent](../../src/multi_agent/supervisor/README_fmapi_supervisor_agent.md)**
- Multi-turn agent with progressive disclosure
- Tool-calling based multi-agent orchestration
- MLflow ResponsesAgent integration
- Streaming response capabilities


## UC Multi-Tool Responses Agent

🤖 **[UC Multi-Tool Responses Agent README](../../src/uc_multi_tool_responses_agent/README.md)**
- Multi-tool integration with Unity Catalog functions, vector search, and agent endpoints
- Thread-safe conversation handling with streaming responses
- Intelligent tool selection and execution orchestration
- MLflow ResponsesAgent integration with FMAPI compatibility
- Production-ready agent with configurable tool registry

📄 **[Agent Implementation](../../src/uc_multi_tool_responses_agent/agent.py)** - Core tool-calling agent with MLflow integration
📄 **[Interface Module](../../src/uc_multi_tool_responses_agent/interface.py)** - Agent interface and API definitions
🧪 **[Integration Tests](../../tests/integration/test_fmapi_supervisor_agent.py)** - Test suite for agent functionality


