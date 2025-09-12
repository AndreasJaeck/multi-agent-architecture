from multi_agent.supervisor.progressive_agent.tools import AgentRegistry, DomainAgentExecutor
from multi_agent.supervisor.progressive_agent.types import AgentConfig


class FakeLLM:
    def make_llm_call(self, messages, temperature: float, max_tokens: int):
        class R:
            def __init__(self):
                self.choices = [type("X", (), {"message": type("Y", (), {"content": "ok"})()})]
        return R()

    @staticmethod
    def extract_response_content(resp):
        return resp.choices[0].message.content


def test_registry_routes_and_executes():
    ac = AgentConfig(name="Alpha", description="d", endpoint="e", system_prompt="s")
    exec = DomainAgentExecutor(ac, FakeLLM())
    reg = AgentRegistry([exec])

    assert reg.has("Alpha") is True
    assert reg.get("Alpha").execute("hello") == "ok"
    assert reg.list_configs()[0].name == "Alpha"
