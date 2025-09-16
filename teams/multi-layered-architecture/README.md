# Multi-Agent Architecture

Databricks examples for multi-agent systems with MLflow integration.


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


## Multi-layered architecture

🤖 **[FMAPI Supervisor Agent](../../src/multi_agent/supervisor/README_fmapi_supervisor_agent.md)**
- Disclosure multi turn agent
- Tool-calling based multi-agent orchestration
- MLflow ResponsesAgent integration
- Streaming responses


## Quick Start

## Project Structure

```
├── src/                    # Core implementations
│   └── multi_agent/        # Main package
├── notebooks/             # Example implementations
├── docs/                  # Architecture documentation
└── tests/                 # Test suites
```

## Key Features

- **Multiple Architecture Patterns**: Hierarchical, pipeline, and mesh designs
- **MLflow Integration**: Production-ready deployment and monitoring
- **Streaming Support**: Real-time responses with FMAPI compatibility
- **Enterprise Authentication**: Databricks OAuth and service principal support
- **Tool-Calling Orchestration**: Intelligent agent coordination and routing
