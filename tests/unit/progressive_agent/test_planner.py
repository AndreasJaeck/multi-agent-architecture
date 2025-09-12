import pytest

from multi_agent.supervisor.progressive_agent.planner import MarkdownPlanner, render_plan_markdown
from multi_agent.supervisor.progressive_agent.types import AgentConfig


class FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def make_llm_call(self, messages, temperature: float, max_tokens: int):
        class R:
            def __init__(self, c):
                self.choices = [type("X", (), {"message": type("Y", (), {"content": c})()})]
        return R(self._content)

    @staticmethod
    def extract_response_content(resp):
        return resp.choices[0].message.content


def test_markdown_planner_parses_bullets():
    experts = [
        AgentConfig(name="Alpha", description="desc", endpoint="e", system_prompt="s"),
        AgentConfig(name="Beta", description="desc", endpoint="e", system_prompt="s"),
    ]
    plan_md = """
- Call Alpha to do X
- Call Beta to do Y
- Consider synthesizing the responses if needed
""".strip()
    planner = MarkdownPlanner(FakeLLM(plan_md))

    steps = planner.plan("question", experts)

    assert len(steps) == 2
    assert steps[0].agent_name == "Alpha"
    assert steps[0].instruction == "do X"
    assert steps[-1].hint_synthesize is True

    rendered = render_plan_markdown(steps)
    assert rendered.splitlines()[0].startswith("- Call Alpha to ")


def test_markdown_planner_ignores_noise():
    experts = [AgentConfig(name="Alpha", description="desc", endpoint="e", system_prompt="s")]
    plan_md = """
random text
- Not a valid bullet
- Call Alpha to handle this
""".strip()
    planner = MarkdownPlanner(FakeLLM(plan_md))
    steps = planner.plan("q", experts)
    assert len(steps) == 1
    assert steps[0].agent_name == "Alpha"
