from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AgentConfig:
    name: str
    description: str
    endpoint: str
    system_prompt: str
    capabilities: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class PlanStep:
    agent_name: str
    instruction: str
    hint_synthesize: bool = False


@dataclass
class AgentResponse:
    agent_name: str
    content: str
