from dataclasses import dataclass
from typing import Dict, List

from .llm import LLMClient
from .types import AgentConfig


class DomainAgentExecutor:
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        self.config = config
        self.llm = llm_client

    def execute(self, query: str) -> str:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": query},
        ]
        resp = self.llm.make_llm_call(messages, temperature=0.2, max_tokens=1000)
        return self.llm.extract_response_content(resp)

class AgentRegistry:
    def __init__(self, agents: List[DomainAgentExecutor]):
        self._by_name: Dict[str, DomainAgentExecutor] = {a.config.name: a for a in agents}
        self._configs: List[AgentConfig] = [a.config for a in agents]

    def has(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> DomainAgentExecutor:
        return self._by_name[name]

    def list_configs(self) -> List[AgentConfig]:
        return list(self._configs)
