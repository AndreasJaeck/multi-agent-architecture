## Progressive Disclosure Supervisor (ResponsesAgent)

This package implements a clean, production-ready progressive disclosure supervisor on top of MLflow ResponsesAgent. The supervisor plans briefly, streams status, calls domain SME agents as structured tools, adapts to their outputs, and synthesizes when useful. It emits structured events compatible with AI Playground, Agent Evaluation, and Monitoring.

- Stateless per request, synchronous execution
- Progressive flow: Plan → Execute → Adapt → Synthesize
- Structured tool messages only: function_call and function_call_output
- Traced at AGENT/LLM/TOOL boundaries via MLflow

---

### Quick start
```python
from mlflow.types.responses import ResponsesAgentRequest
from multi_agent.supervisor.progressive_agent.factory import create_agent

AGENT = create_agent()

request = ResponsesAgentRequest(input=[{"role": "user", "content": "What is (15*8)/4+25?"}])
for event in AGENT.predict_stream(request):
    print(event)

resp = AGENT.predict(request)
print(resp.output)
```

---

### Progressive disclosure workflow
1) Input
- Extract latest user message
- Stream a brief Thinking delta

2) Planning
- Planner returns 2–4 bullets: “Call {Agent} to …”
- Plan is streamed once as an output block

3) Execution (structured tools)
- For each step: emit function_call → call SME → emit function_call_output
- No duplicate free-form blocks for SME content

4) Adaptive reaction
- needs_clarification: enrich instruction from context and retry once; otherwise ask user
- answer_complete + verification step: skip redundant verification
- tool_failure: skip or continue conservatively

5) Synthesis (optional)
- If hinted or multiple SME answers present, synthesize final answer as one output block

6) Termination
- predict aggregates output_item.done items into ResponsesAgentResponse

---

### Files and modules
- config.py
  - SupervisorConfig: global settings (endpoint, temperatures, tokens, toggles)

- types.py
  - AgentConfig: SME definition (name, description, endpoint, system_prompt)
  - PlanStep: agent_name, instruction, hint_synthesize
  - AgentResponse: agent_name + content

- llm.py
  - DatabricksClientManager: provides WorkspaceClient and OpenAI (serving) client
  - LLMClient
    - make_llm_call(messages, temperature, max_tokens)
    - extract_response_content(response)

- prompts.py
  - supervisor_system_prompt(experts_md)
  - planning_prompt(question, experts_md) – includes guidance to put exact expressions/expected values into verification steps
  - synthesis_prompt(question, responses_md)

- planner.py
  - MarkdownPlanner
    - plan(question, experts) -> list[PlanStep]
    - _parse_markdown(md) -> list[PlanStep]
  - render_plan_markdown(steps) -> str

- tools.py
  - DomainAgentExecutor(config, llm_client)
    - execute(query) -> str (system + user messages on SME endpoint)
  - AgentRegistry(agents)
    - has(name), get(name), list_configs()

- streaming.py
  - ThoughtStreamer(thinking_enabled)
    - thinking(text) -> response.text.delta | None
    - evaluation(text) -> response.text.delta
  - StreamFormatter
    - output_block(agent, text) -> response.output_item.done
    - tool_call_item(agent, call_id, name, arguments)
    - tool_output_item(agent, call_id, output)

- synthesis.py
  - ResponseSynthesizer
    - synthesize(question, responses) -> str

- supervisor.py
  - SupervisorResponsesAgent(ResponsesAgent)
    - predict_stream(request)
      - Streams plan, emits function_call/function_call_output per step
      - Adaptive logic:
        - _classify_output(text) -> {type}
        - _is_verification_step(instruction)
        - _try_autofill(user_message, prior) -> str
      - Optional synthesis
    - predict(request) – aggregates output items

- factory.py
  - DEFAULT_AGENTS: example agent configs (replace with your own)
  - build_registry(config) -> AgentRegistry
  - create_agent() -> SupervisorResponsesAgent; enables mlflow.openai.autolog() and mlflow.models.set_model

---

### Execution flow (detailed)
1. Receive ResponsesAgentRequest → extract latest user content
2. Stream Thinking delta → call planner → stream plan block
3. For each PlanStep:
   - Stream “Executing …” delta
   - Emit function_call → execute SME → emit function_call_output
   - If SME asks for details, auto-enrich once (add exact expression/result) and retry; otherwise ask user and stop
   - If previous step already answered and the current is a verification step, skip
4. If synthesis needed, generate final block; otherwise end
5. predict returns collected output_item.done items

---

### Configuration
- SupervisorConfig (config.py):
  - llm_endpoint, thinking_enabled
  - planning_temperature, execution_temperature
  - max_tokens_plan, max_tokens_exec
- DEFAULT_AGENTS (factory.py): update names, endpoints, and prompts for your SMEs

---

### Testing
- Unit tests (fast):
  - tests/unit/progressive_agent/test_planner.py
  - tests/unit/progressive_agent/test_tools_registry.py
  - tests/unit/progressive_agent/test_synthesizer.py
  - tests/unit/progressive_agent/test_supervisor_stream.py
- Integration (real endpoints; opt-in):
  - tests/integration/test_progressive_agent_real_endpoints.py
```bash
export DATABRICKS_CONFIG_PROFILE=<profile>
export RUN_PROGRESSIVE_AGENT_INTEGRATION=1
pytest -q -s tests/integration/test_progressive_agent_real_endpoints.py | cat
```

---

### Design choices and best practices
- ResponsesAgent pattern; synchronous; stateless per call
- Structured tool events, no duplicate SME text blocks
- Deterministic routing via PlanStep.agent_name + AgentRegistry
- Adaptive controller prevents wasted steps and handles SME clarifications
- MLflow autologging and spans for observability

---

### Extending the agent
- Add SMEs: extend DEFAULT_AGENTS
- Smarter evaluation: replace _classify_output with LLM or ruleset
- Long-running tools: stream progress, chunk outputs; add timeouts/budgets
- Harden retries/backoff in LLMClient if needed

---

### Troubleshooting
- Duplicate outputs: ensure you’re not printing both tool_output and custom blocks for the same content
- Auth: set DATABRICKS_CONFIG_PROFILE for integration tests
- Planner missing details: planner prompt enforces including exact expressions for verification

