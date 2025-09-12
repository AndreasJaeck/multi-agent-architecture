from uuid import uuid4
from typing import Optional
from mlflow.types.responses import ResponsesAgentStreamEvent


class ThoughtStreamer:
    def __init__(self, thinking_enabled: bool = True):
        self.thinking_enabled = thinking_enabled

    def thinking(self, text: str, item_id: Optional[str] = None) -> Optional[ResponsesAgentStreamEvent]:
        if not self.thinking_enabled:
            return None
        return ResponsesAgentStreamEvent(
            type="response.text.delta",
            item_id=item_id or str(uuid4()),
            delta={"text": f"**Thinking**: {text}\n\n"},
        )

    def evaluation(self, text: str, item_id: Optional[str] = None) -> ResponsesAgentStreamEvent:
        return ResponsesAgentStreamEvent(
            type="response.text.delta",
            item_id=item_id or str(uuid4()),
            delta={"text": f"**Evaluation**: {text}\n\n"},
        )


class StreamFormatter:
    @staticmethod
    def output_block(agent_instance, text: str, item_id: Optional[str] = None) -> ResponsesAgentStreamEvent:
        return ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=agent_instance.create_text_output_item(text, item_id or str(uuid4())),
        )

    @staticmethod
    def agent_block(agent_instance, agent_name: str, response: str) -> ResponsesAgentStreamEvent:
        text = f"## {agent_name} Response\n{response}\n\n"
        return StreamFormatter.output_block(agent_instance, text)

    @staticmethod
    def final_block(agent_instance, final_answer: str) -> ResponsesAgentStreamEvent:
        text = f"##  Synthesized Final Answer\n{final_answer}"
        return StreamFormatter.output_block(agent_instance, text)

    @staticmethod
    def tool_call_item(agent_instance, call_id: str, name: str, arguments: str) -> ResponsesAgentStreamEvent:
        return ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=agent_instance.create_function_call_item(str(uuid4()), call_id, name, arguments),
        )

    @staticmethod
    def tool_output_item(agent_instance, call_id: str, output: str) -> ResponsesAgentStreamEvent:
        return ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=agent_instance.create_function_call_output_item(call_id, output),
        )
