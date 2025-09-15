# Multi-Agent Architecture

A comprehensive framework for building scalable multi-agent systems with multiple architectural patterns and production-ready implementations.

## Overview

This project provides implementations of various multi-agent architecture patterns, focusing on hierarchical supervisor systems with MLflow integration for enterprise deployment. The architecture supports different coordination patterns including tool-calling, streaming responses, and circuit breaker resilience.

## Multi-Agent Patterns & Flow Charts

### 1. Hierarchical Architecture

**Description**: Tree-like structure with parent-child relationships where higher-level agents manage and coordinate lower-level specialists.

```mermaid
graph TD
    A[User Query] --> B[Top-Level Supervisor]
    B --> C{Task Analysis}
    C --> D[Domain Supervisor 1]
    C --> E[Domain Supervisor 2]
    C --> F[Domain Supervisor 3]

    D --> D1[Specialist Agent 1A]
    D --> D2[Specialist Agent 1B]

    E --> E1[Specialist Agent 2A]
    E --> E2[Specialist Agent 2B]

    F --> F1[Specialist Agent 3A]

    D1 --> G[Response Aggregation]
    D2 --> G
    E1 --> G
    E2 --> G
    F1 --> G
    G --> H[Final Response]
```

**Key Characteristics:**
- Clear chain of command and responsibility
- Natural delegation of complex tasks
- Fault isolation by hierarchy level
- Scales vertically (adding layers)

### 2. Pipeline/Sequential Architecture

**Description**: Fixed chain of agents where each agent processes and passes results to the next agent in sequence.

```mermaid
graph LR
    A[User Query] --> B[Input Processor]
    B --> C[Data Validator]
    C --> D[Business Logic Agent]
    D --> E[Results Formatter]
    E --> F[Output Generator]
    F --> G[Final Response]

    subgraph "Stage 1: Input"
        B
    end

    subgraph "Stage 2: Processing"
        C
        D
    end

    subgraph "Stage 3: Output"
        E
        F
    end
```

**Key Characteristics:**
- Predictable, linear flow
- Easy to optimize individual stages
- Clear data lineage
- Handles staged processing well

### 3. Flexible Supervisor (Task-Oriented)

**Description**: Intelligent supervisor that can handle tasks directly, use tools, or route to specialists based on task requirements.

```mermaid
graph TD
    A[User Query] --> B[Flexible Supervisor]
    B --> C{Task Assessment}

    C --> D{Can handle directly?}
    D -->|Yes| E[Direct Response]
    D -->|No| F{Requires tools?}

    F -->|Yes| G[Tool Selection]
    F -->|No| H[Specialist Routing]

    G --> I[Tool Execution]
    H --> J[Specialist Agent]

    I --> K[Response Integration]
    J --> K
    E --> L[Final Response]
    K --> L
```

**Key Characteristics:**
- Dynamic routing based on actual needs
- Supervisor maintains direct capabilities
- Planning-driven approach
- Optimizes execution path per request

### 4. Swarm Architecture

**Description**: Many simple, autonomous agents that interact locally to achieve global objectives through emergent behavior.

```mermaid
graph TD
    A[Global Objective] --> B[Agent Swarm]
    B --> C[Agent 1]
    B --> D[Agent 2]
    B --> E[Agent 3]
    B --> F[Agent 4]
    B --> G[Agent 5]

    C --> H[Local Interactions]
    D --> H
    E --> H
    F --> H
    G --> H

    H --> I{Emergent Behavior}
    I --> J[Collective Solution]
    I --> K[Self-Organization]

    J --> L[Global Result]
    K --> L
```

**Key Characteristics:**
- Highly fault-tolerant
- Self-organizing and adaptive
- Natural load distribution
- Emergent intelligence from simple rules

### 5. Blackboard Architecture

**Description**: Shared knowledge space where agents contribute information and solutions collaboratively.

```mermaid
graph TD
    A[User Query] --> B[Blackboard System]
    B --> C[Blackboard<br/>Shared Knowledge]

    D[Agent 1] --> C
    E[Agent 2] --> C
    F[Agent 3] --> C
    G[Agent 4] --> C

    C --> H[Knowledge Integration]
    H --> I{Consensus?}

    I -->|No| J[Agent Refinement]
    J --> C
    I -->|Yes| K[Final Solution]

    D --> L[Expertise Filter]
    E --> L
    F --> L
    G --> L
    L --> C
```

**Key Characteristics:**
- Supports heterogeneous agents
- Natural knowledge accumulation
- Flexible agent participation
- Iterative refinement of solutions

### 6. Federated Architecture

**Description**: Distributed autonomous agents with local decision-making authority that coordinate when needed.

```mermaid
graph TD
    A[Global Task] --> B[Federation Coordinator]

    B --> C[Federation A]
    B --> D[Federation B]
    B --> E[Federation C]

    C --> C1[Local Supervisor A]
    C --> C2[Agent A1]
    C --> C3[Agent A2]

    D --> D1[Local Supervisor B]
    D --> D2[Agent B1]
    D --> D3[Agent B2]

    E --> E1[Local Supervisor C]
    E --> E2[Agent C1]
    E --> E3[Agent C2]

    C1 --> F[Inter-Federation<br/>Coordination]
    D1 --> F
    E1 --> F
    F --> G[Integrated Solution]
```

**Key Characteristics:**
- High autonomy and local optimization
- Natural fault isolation
- Good for geographically distributed systems
- Respects organizational boundaries

## Coordination Patterns

### 1. Market-Based Coordination

```mermaid
graph TD
    A[Task Request] --> B[Market Coordinator]
    B --> C[Bid Collection]

    D[Agent 1] --> C
    E[Agent 2] --> C
    F[Agent 3] --> C

    C --> G[Bid Evaluation]
    G --> H[Winner Selection]
    H --> I[Task Assignment]
    I --> J[Task Execution]
    J --> K[Payment/Feedback]
    K --> L[Market Update]
```

### 2. Democratic/Voting Coordination

```mermaid
graph TD
    A[Decision Required] --> B[Voting Coordinator]
    B --> C[Proposal Generation]

    C --> D[Agent Voting]
    D --> E[Vote Collection]

    F[Agent 1] --> E
    G[Agent 2] --> E
    H[Agent 3] --> E

    E --> I[Vote Tallying]
    I --> J{Consensus?}
    J -->|Yes| K[Decision Implementation]
    J -->|No| L[Refinement Round]
    L --> C
```

### 3. Actor Model Coordination

```mermaid
graph TD
    A[User] --> B[Message Router]
    B --> C[Actor 1 Mailbox]
    B --> D[Actor 2 Mailbox]
    B --> E[Actor 3 Mailbox]

    C --> F[Actor 1]
    D --> G[Actor 2]
    E --> H[Actor 3]

    F --> I[Message Processing]
    G --> I
    H --> I

    I --> J[Response Messages]
    J --> B
    B --> K[Response Aggregation]
    K --> L[Final Response]
```

## Implementation Examples

### FMAPI Supervisor Agent

A hybrid multi-agent architecture combining **hierarchical coordination**, **federated autonomy**, and **progressive disclosure** patterns:

```mermaid
graph TD
    A[User Query] --> B[RegistryToolCallingAgent<br/>Central Coordinator]

    B --> C[Planning Phase<br/>Communicates Intent]
    C --> D[Progressive Disclosure<br/>Execution Planning]

    D --> E{Dynamic Assessment}
    E --> F[Direct Response<br/>Simple Tasks]
    E --> G[Delegation to Domain Experts<br/>Complex Tasks]

    G --> H[Domain Expert 1<br/>Supervisor + Synthesis]
    G --> I[Domain Expert 2<br/>Supervisor + Synthesis]
    G --> J[Domain Expert N<br/>Supervisor + Synthesis]

    H --> H1[Sub-Agent A]
    H --> H2[Sub-Agent B]
    I --> I1[Sub-Agent C]
    I --> I2[Sub-Agent D]
    J --> J1[Sub-Agent E]

    H1 --> HS[Domain Synthesis]
    H2 --> HS
    I1 --> IS[Domain Synthesis]
    I2 --> IS
    J1 --> JS[Domain Synthesis]

    HS --> K[Expert Response<br/>Federated Autonomy]
    IS --> K
    JS --> K

    K --> L[Supervisor Response Integration]
    F --> L
    L --> M[Streaming Updates<br/>Progressive Disclosure]
    M --> N[Final Response]

    M --> E
```

**Hybrid Architecture Characteristics:**
- **Hierarchical**: Multi-level coordination where central supervisor delegates to domain expert supervisors
- **Federated**: Each domain expert maintains autonomous planning and orchestrates sub-agents
- **Progressive Disclosure**: Communicates intentions and dynamically plans execution steps
- **Distributed Synthesis**: Domain experts perform their own synthesis before forwarding to supervisor
- **Real-time Feedback Loop**: Streaming updates can trigger re-assessment during response generation

**Key Features:**
- Intelligent task routing with LLM-driven analysis and intent communication
- Hierarchical domain experts that orchestrate specialized sub-agents
- Distributed synthesis and response integration across multiple levels
- Real-time feedback loops from streaming updates to dynamic assessment for continuous improvement
- Progressive disclosure with streaming updates and reasoning
- Comprehensive error handling and recovery across federated agent hierarchies
- Production observability with MLflow tracing and streaming events

→ [Detailed Documentation](src/multi_agent/supervisor/README_fmapi_supervisor_agent.md)

## Hybrid Patterns

Most production systems combine multiple architectural patterns to achieve optimal results. The FMAPI Supervisor Agent exemplifies this approach by integrating:

- **Multi-level hierarchical coordination** where domain experts themselves orchestrate sub-agents
- **Federated autonomy** with distributed synthesis and independent execution planning
- **Progressive disclosure** for transparent multi-level execution planning
- **Real-time feedback loops** enabling continuous assessment during streaming response generation

This hybrid design creates a sophisticated hierarchy where domain experts act as mini-supervisors, performing their own synthesis before forwarding results to the central coordinator, with real-time feedback loops from streaming updates enabling continuous improvement through dynamic re-assessment.

## Pattern Selection Guide

| Pattern | Complexity | Scalability | Fault Tolerance | Flexibility | Use Case |
|---------|------------|-------------|----------------|-------------|----------|
| Hierarchical | Low | Medium | Low | Low | Enterprise workflows, clear authority |
| Pipeline | Low | Low | Low | Low | ETL, content processing |
| Flexible Supervisor | High | Medium | Medium | High | General assistants, adaptive tasks |
| Swarm | Medium | High | High | High | Optimization, distributed sensing |
| Blackboard | Medium | Medium | Medium | High | Multi-expert collaboration |
| Federated | High | High | High | Medium | Multi-organization systems |
| **Hybrid (Hierarchical + Federated)** | **Medium-High** | **High** | **High** | **High** | **Complex adaptive systems, progressive disclosure** |

## Quick Start

### Basic Usage

```python
# FMAPI Supervisor Agent
from src.multi_agent.supervisor.fmapi_supervisor_agent import AGENT
response = AGENT.predict_stream(request)
```

## Documentation

- [FMAPI Supervisor Agent](src/multi_agent/supervisor/README_fmapi_supervisor_agent.md) - Tool-calling implementation details


## License

MIT License - see individual component documentation for specific licensing details.
