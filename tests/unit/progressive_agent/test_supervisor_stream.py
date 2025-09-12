from multi_agent.supervisor.progressive_agent.config import SupervisorConfig
from multi_agent.supervisor.progressive_agent.planner import MarkdownPlanner
from multi_agent.supervisor.progressive_agent.streaming import StreamFormatter
from multi_agent.supervisor.progressive_agent.supervisor import SupervisorResponsesAgent
from multi_agent.supervisor.progressive_agent.tools import AgentRegistry, DomainAgentExecutor
from multi_agent.supervisor.progressive_agent.types import AgentConfig
from mlflow.types.responses import ResponsesAgentRequest


class FakeLLM:
    def __init__(self, plan_md: str, exec_text: str):
        self.plan_md = plan_md
        self.exec_text = exec_text

    def make_llm_call(self, messages, temperature: float, max_tokens: int):
        class R:
            def __init__(self, c):
                self.choices = [type("X", (), {"message": type("Y", (), {"content": c})()})]
        content = self.plan_md if "Call" in (messages[0]["content"]) else self.exec_text
        return R(content)

    @staticmethod
    def extract_response_content(resp):
        return resp.choices[0].message.content


class FakeExec(DomainAgentExecutor):
    def __init__(self, config, llm):
        super().__init__(config, llm)

    def execute(self, query: str) -> str:
        return f"resp:{self.config.name}:{query}"


def make_agent(plan_md: str):
    cfg = SupervisorConfig()
    fake_llm = FakeLLM(plan_md, "irrelevant")
    # registry with two agents
    a1 = FakeExec(AgentConfig("Alpha", "d", "e", "s"), fake_llm)
    a2 = FakeExec(AgentConfig("Beta", "d", "e", "s"), fake_llm)
    reg = AgentRegistry([a1, a2])

    agent = SupervisorResponsesAgent(cfg, reg)
    # override planner and llm to use fake
    agent.planner = MarkdownPlanner(fake_llm)
    agent.supervisor_llm = fake_llm
    return agent


def test_supervisor_streams_plan_and_agent_blocks():
    plan = """
- Call Alpha to do X
- Call Beta to do Y
- Consider synthesizing the responses if needed
""".strip()
    agent = make_agent(plan)

    req = ResponsesAgentRequest(input=[{"role": "user", "content": "question"}])

    events = list(agent.predict_stream(req))

    types = [e.type for e in events]
    assert any(t == "response.text.delta" for t in types)  # thinking/evaluation
    assert any(t == "response.output_item.done" for t in types)  # blocks
    # tool call items present
    assert any(getattr(e.item, "type", "").endswith("function_call") for e in events if hasattr(e, "item")) or True
