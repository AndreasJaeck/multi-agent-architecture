import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, ResponsesAgentStreamEvent

from .config import SupervisorConfig
from .llm import LLMClient
from .planner import MarkdownPlanner, render_plan_markdown
from .streaming import ThoughtStreamer, StreamFormatter
from .synthesis import ResponseSynthesizer
from .tools import AgentRegistry
from .types import AgentResponse


class SupervisorResponsesAgent(ResponsesAgent):
    def __init__(self, config: SupervisorConfig, registry: AgentRegistry):
        self.config = config
        self.registry = registry
        self.supervisor_llm = LLMClient(config.llm_endpoint)
        self.planner = MarkdownPlanner(self.supervisor_llm)
        self.streamer = ThoughtStreamer(thinking_enabled=config.thinking_enabled)
        self.synthesizer = ResponseSynthesizer(self.supervisor_llm)

    # --- Simple output evaluator heuristics ---
    @staticmethod
    def _classify_output(text: str):
        t = text.lower()
        if any(k in t for k in ["could you please provide", "i need to know", "please provide", "clarify", "which calculation"]):
            return {"type": "needs_clarification"}
        if any(k in t for k in ["error", "failed", "unable to", "timeout"]):
            return {"type": "tool_failure"}
        if any(k in t for k in ["final answer", "therefore", "result is", "the answer is", "equals "]):
            return {"type": "answer_complete"}
        return {"type": "ok"}

    @staticmethod
    def _is_verification_step(instruction: str) -> bool:
        s = instruction.lower()
        return any(k in s for k in ["verify", "confirm", "double-check", "check"])

    @staticmethod
    def _try_autofill(user_message: str, prior: list[AgentResponse]) -> str:
        # Very simple heuristic: if prior contains a numeric result, incorporate it
        for r in reversed(prior):
            # extract last number-like token
            import re
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", r.content)
            if nums:
                return f" Use the original expression from the user and verify the computed result equals {nums[-1]}, showing all steps."
        return ""

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(self, request: ResponsesAgentRequest):
        user_message = request.input[-1].content if request.input else ""

        thought = self.streamer.thinking("Analyzing question and creating plan")
        if thought:
            yield thought

        try:
            steps = self.planner.plan(user_message, self.registry.list_configs())
            plan_md = render_plan_markdown(steps)
            yield StreamFormatter.output_block(self, plan_md)

            collected: list[AgentResponse] = []
            i = 0
            while i < len(steps):
                step = steps[i]
                if not self.registry.has(step.agent_name):
                    warn = self.streamer.thinking(f"⚠️ {step.agent_name} unavailable, skipping")
                    if warn:
                        yield warn
                    i += 1
                    continue

                exec_thought = self.streamer.thinking(f"Executing: Call {step.agent_name} to {step.instruction}")
                if exec_thought:
                    yield exec_thought

                call_id = f"call-{step.agent_name}"
                yield StreamFormatter.tool_call_item(self, call_id, step.agent_name, f'{{"instruction": "{step.instruction}"}}')

                out = self.registry.get(step.agent_name).execute(step.instruction)
                yield StreamFormatter.tool_output_item(self, call_id, out)

                current_response = AgentResponse(agent_name=step.agent_name, content=out)
                collected.append(current_response)

                # Adaptive handling
                cls = self._classify_output(out)
                if cls["type"] == "needs_clarification":
                    addon = self._try_autofill(user_message, collected)
                    if addon:
                        retry_instr = f"{step.instruction}.{addon}".strip()
                        retry_call_id = f"{call_id}-retry"
                        yield StreamFormatter.tool_call_item(self, retry_call_id, step.agent_name, f'{{"instruction": "{retry_instr}"}}')
                        retry_out = self.registry.get(step.agent_name).execute(retry_instr)
                        yield StreamFormatter.tool_output_item(self, retry_call_id, retry_out)
                        collected.append(AgentResponse(agent_name=step.agent_name, content=retry_out))
                        # proceed to next step after one retry
                        i += 1
                        continue
                    else:
                        # Ask user: stream a clarification prompt and stop
                        q = "The expert needs more details to proceed. Please specify the exact expression or details to verify."
                        ask = self.streamer.evaluation(q)
                        yield ask
                        break
                elif cls["type"] == "tool_failure":
                    # Simple replan: skip remaining verification steps
                    i += 1
                    continue
                elif cls["type"] == "answer_complete" and self._is_verification_step(step.instruction):
                    # If verification follows a complete answer, skip redundant steps
                    i += 1
                    continue
                else:
                    i += 1

            needs_synth = any(s.hint_synthesize for s in steps) and len(collected) > 1
            if needs_synth:
                eval_ev = self.streamer.evaluation("Synthesizing responses…")
                yield eval_ev
                final = self.synthesizer.synthesize(user_message, collected)
                yield StreamFormatter.final_block(self, final)
            else:
                end_ev = self.streamer.evaluation("No synthesis requested - presenting expert responses.")
                yield end_ev
        except Exception as e:
            err_block = f" **Planning Error:** {str(e)}\n\nFalling back to consulting all available domain experts..."
            yield StreamFormatter.output_block(self, err_block)

            collected: list[AgentResponse] = []
            for exec_agent in self.registry.list_configs():
                call_id = f"fallback-{exec_agent.name}"
                yield StreamFormatter.tool_call_item(self, call_id, exec_agent.name, f'{{"instruction": "{user_message}"}}')
                out = self.registry.get(exec_agent.name).execute(user_message)
                yield StreamFormatter.tool_output_item(self, call_id, out)
                collected.append(AgentResponse(agent_name=exec_agent.name, content=out))
                # Prefer structured tool events only: do not emit redundant text blocks per SME

            final = self.synthesizer.synthesize(user_message, collected) if collected else "No agents available."
            yield StreamFormatter.final_block(self, final)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item for event in self.predict_stream(request) if isinstance(event, ResponsesAgentStreamEvent) and event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
