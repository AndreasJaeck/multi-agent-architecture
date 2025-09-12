from dataclasses import dataclass


@dataclass
class SupervisorConfig:
    llm_endpoint: str = "databricks-claude-3-7-sonnet"
    thinking_enabled: bool = True
    planning_temperature: float = 0.1
    execution_temperature: float = 0.2
    max_tokens_plan: int = 800
    max_tokens_exec: int = 1000
    # Prefer tools overall, but allow direct answers for trivial arithmetic to avoid unnecessary calls
    allow_direct_for_trivial: bool = True
