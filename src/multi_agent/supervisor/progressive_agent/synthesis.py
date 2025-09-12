from .llm import LLMClient
from .prompts import synthesis_prompt
from .types import AgentResponse


class ResponseSynthesizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def synthesize(self, question: str, responses: list[AgentResponse]) -> str:
        responses_md = "\n".join([f"{r.agent_name}: {r.content}" for r in responses])
        prompt = synthesis_prompt(question, responses_md)
        resp = self.llm.make_llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
        )
        return self.llm.extract_response_content(resp)
