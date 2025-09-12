import mlflow

from .config import SupervisorConfig
from .llm import LLMClient
from .supervisor import SupervisorResponsesAgent
from .tools import AgentRegistry, DomainAgentExecutor
from .types import AgentConfig


DEFAULT_AGENTS = [
    AgentConfig(
        name='CoatingsSupervisorAgent',
        description='Coatings industry SME for market news search and plant data analysis',
        endpoint='genie_multi_agent_basf',
        system_prompt='You are a coatings industry expert. Provide detailed analysis on coating-related topics.',
        capabilities='vector search, SQL-based analytics, market trends, durability/UV materials',
        domain='chemical_data'
    ),
    AgentConfig(
        name='GenomicsSupervisorAgent',
        description='Genomics & computational SME for Python execution and patient data queries',
        endpoint='genie_multi_agent_basf_v2',
        system_prompt='You are a genomics and computational expert with Python and SQL capabilities.',
        capabilities='python code execution, math/statistics, genomics SQL/data querying',
        domain='computational_tools'
    ),
]


def build_registry(config: SupervisorConfig) -> AgentRegistry:
    agents = []
    for ac in DEFAULT_AGENTS:
        llm = LLMClient(ac.endpoint)
        agents.append(DomainAgentExecutor(ac, llm))
    return AgentRegistry(agents)


def create_agent() -> SupervisorResponsesAgent:
    cfg = SupervisorConfig()
    registry = build_registry(cfg)
    agent = SupervisorResponsesAgent(cfg, registry)
    mlflow.openai.autolog()
    mlflow.models.set_model(agent)
    return agent
