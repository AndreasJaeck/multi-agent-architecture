from multi_agent.supervisor.progressive_agent.synthesis import ResponseSynthesizer
from multi_agent.supervisor.progressive_agent.types import AgentResponse


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


def test_synthesize_combines_responses():
    llm = FakeLLM("final")
    synth = ResponseSynthesizer(llm)
    out = synth.synthesize("q", [AgentResponse(agent_name="A", content="x"), AgentResponse(agent_name="B", content="y")])
    assert out == "final"
