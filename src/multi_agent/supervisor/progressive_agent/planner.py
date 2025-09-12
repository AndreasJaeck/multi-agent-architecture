from typing import List

from .llm import LLMClient
from .prompts import planning_prompt
from .types import AgentConfig, PlanStep


class MarkdownPlanner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, question: str, experts: List[AgentConfig]) -> List[PlanStep]:
        def mk_line(e: AgentConfig) -> str:
            caps = f" | capabilities: {e.capabilities}" if getattr(e, "capabilities", None) else ""
            dom = f" | domain: {e.domain}" if getattr(e, "domain", None) else ""
            return f"- {e.name}: {e.description}{caps}{dom}"
        experts_md = "\n".join([mk_line(e) for e in experts])
        prompt = planning_prompt(question, experts_md)
        resp = self.llm.make_llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )
        md = self.llm.extract_response_content(resp).strip()
        steps = self._parse_markdown(md)
        # Cap number of steps to avoid excessive calls
        return steps[:3]

    @staticmethod
    def _parse_markdown(md: str) -> List[PlanStep]:
        steps: List[PlanStep] = []
        hint_synth = False
        for line in md.splitlines():
            s = line.strip()
            if not s.startswith("-"):
                continue
            content = s[1:].strip()
            if content.lower().startswith("consider synthesizing"):
                hint_synth = True
                continue
            # expected variants: "Call AgentName to <instruction>" or "Use AgentName to <instruction>"
            lower = content.lower()
            if (lower.startswith("call ") or lower.startswith("use ")) and " to " in content:
                # remove leading verb (Call/Use)
                if lower.startswith("call "):
                    after = content[5:]
                else:
                    after = content[4:]
                agent_name, instruction = after.split(" to ", 1)
                steps.append(PlanStep(agent_name=agent_name.strip(), instruction=instruction.strip()))
        if steps:
            steps[-1].hint_synthesize = hint_synth
        return steps


def render_plan_markdown(steps: List[PlanStep]) -> str:
    lines = [f"- Call {s.agent_name} to {s.instruction}" for s in steps]
    if any(s.hint_synthesize for s in steps):
        lines.append("- Consider synthesizing the responses if needed")
    return "\n".join(lines)
