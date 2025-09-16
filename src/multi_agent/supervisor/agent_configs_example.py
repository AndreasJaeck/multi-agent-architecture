"""
Example agent configuration file.

This file shows how to configure DOMAIN_AGENTS for the FMAPI supervisor agent.
Copy this file to agent_configs.py and customize with your actual endpoints and resources.

For the actual configuration file, see agent_configs.py (which should be in .gitignore)
"""

from dataclasses import dataclass, field
from typing import List, Optional

# ---------- Configuration: top-level supervisor settings ----------
@dataclass
class SupervisorConfig:
    """Top-level supervisor configuration.

    Attributes
    - llm_endpoint: Name of the Databricks Serving chat-completions model used by the supervisor
    - thinking_enabled: Reserved flag to hint reasoning-style prompts (not consumed here)
    - max_steps: Safety cap for iterative tool-calling loops
    """
    llm_endpoint: str = "databricks-claude-3-7-sonnet"
    thinking_enabled: bool = True
    max_steps: int = 10


@dataclass
class ResourceConfig:
    """Configuration for underlying resources that the agent depends on.

    This includes Genie spaces, functions, vector search indices, and other
    Databricks resources needed for authentication and access.
    """
    genie_spaces: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    vector_search_indices: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for a domain expert (backed by a Serving endpoint).

    - name: human/LLM-facing expert name (also used as tool name)
    - description: brief description for tool routing
    - endpoint: Databricks model serving endpoint name for this expert
    - system_prompt: system message used when invoking the expert
    - capabilities/domain: optional metadata surfaced in tool descriptions to aid routing
    - resources: underlying Databricks resources (Genie spaces, functions, vector indices)
    """
    name: str
    description: str
    endpoint: str
    system_prompt: str
    capabilities: Optional[str] = None
    domain: Optional[str] = None
    resources: ResourceConfig = field(default_factory=ResourceConfig)


# Example default experts. Adjust endpoints and prompts to your environment.
# Replace the placeholder values with your actual Databricks resources.
DOMAIN_AGENTS: List[AgentConfig] = [
    AgentConfig(
        name="ChemicalDataAgent",
        description="Chemical industry expert for market analysis and plant data",
        endpoint="your_chemical_data_endpoint",
        system_prompt="You are a chemical industry expert. Provide detailed analysis on chemical-related topics.",
        capabilities="semantic similarity search for chemical industry news, SQL-based analytics for manufacturing plants",
        domain="chemical_data",
        resources=ResourceConfig(
            genie_spaces=["your_genie_space_id_1"],
            vector_search_indices=["your_catalog.your_schema.your_vector_index"],
        ),
    ),
    AgentConfig(
        name="ComputationalAgent",
        description="Computational expert for mathematical calculations and data analysis",
        endpoint="your_computational_endpoint",
        system_prompt="You are a computational expert specialized in mathematical calculations and data analysis.",
        capabilities="arithmetic calculations, mathematical computations, Python code execution, data queries",
        domain="computational_tools",
        resources=ResourceConfig(
            genie_spaces=["your_genie_space_id_2"],
            functions=[
                "your_catalog.your_schema.compute_math_function",
                "your_catalog.your_schema.execute_python_function",
            ],
        ),
    ),
    # Add more agents as needed
    # AgentConfig(
    #     name="YourCustomAgent",
    #     description="Description of your custom agent",
    #     endpoint="your_custom_endpoint",
    #     system_prompt="System prompt for your custom agent",
    #     capabilities="What your agent can do",
    #     domain="domain_category",
    #     resources=ResourceConfig(
    #         genie_spaces=["your_genie_space_ids"],
    #         functions=["your_function_names"],
    #         vector_search_indices=["your_vector_indices"],
    #     ),
    # ),
]
