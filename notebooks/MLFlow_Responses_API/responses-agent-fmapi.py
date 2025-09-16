# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author and deploy an OpenAI Responses API agent using a model hosted on Mosaic AI Foundation Model API
# MAGIC
# MAGIC This notebook shows how to author an OpenAI Responses agent and wrap it using the [`ResponsesAgent`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent) interface to make it compatible with Mosaic AI. In this notebook you learn to:
# MAGIC
# MAGIC - Author an [Open AI Responses API](https://platform.openai.com/docs/api-reference/responses) agent (wrapped with `ResponsesAgent`) that calls an LLM hosted using Mosaic AI Foundation Models.
# MAGIC - Manually test the agent
# MAGIC - Evaluate the agent using Mosaic AI Agent Evaluation
# MAGIC - Log and deploy the agent
# MAGIC
# MAGIC To learn more about authoring an agent using Mosaic AI Agent Framework, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/author-agent) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/create-chat-model)).
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md

# MAGIC ## Define the agent in code
# MAGIC Define the agent code in a single cell below. This lets you easily write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC
# MAGIC #### Agent tools
# MAGIC This agent code adds the built-in Unity Catalog function `system.ai.python_exec` to the agent. The agent code also includes commented-out sample code for adding a vector search index to perform unstructured data retrieval.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see Databricks documentation ([AWS](https://docs.databricks.com/aws/generative-ai/agent-framework/agent-tool) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool))


# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import json
# MAGIC from typing import Any, Callable, Generator, Optional
# MAGIC from uuid import uuid4
# MAGIC
# MAGIC import backoff
# MAGIC import mlflow
# MAGIC import openai
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
# MAGIC from mlflow.entities import SpanType
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC from openai import OpenAI
# MAGIC from pydantic import BaseModel
# MAGIC from unitycatalog.ai.core.base import get_uc_function_client
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC
# MAGIC # TODO: Update with your system prompt
# MAGIC SYSTEM_PROMPT = """
# MAGIC You are a helpful assistant that can run Python code.
# MAGIC """
# MAGIC
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC class ToolInfo(BaseModel):
# MAGIC     """
# MAGIC     Class representing a tool for the agent.
# MAGIC     - "name" (str): The name of the tool.
# MAGIC     - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
# MAGIC     - "exec_fn" (Callable): Function that implements the tool logic
# MAGIC     """
# MAGIC
# MAGIC     name: str
# MAGIC     spec: dict
# MAGIC     exec_fn: Callable
# MAGIC
# MAGIC
# MAGIC def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
# MAGIC     """
# MAGIC     Factory function to create ToolInfo objects from a given tool spec
# MAGIC     and (optionally) a custom execution function.
# MAGIC     """
# MAGIC     # Remove 'strict' property, as Claude models do not support it in tool specs.
# MAGIC     tool_spec["function"].pop("strict", None)
# MAGIC     tool_name = tool_spec["function"]["name"]
# MAGIC     # Converts tool name with double underscores to UDF dot notation.
# MAGIC     udf_name = tool_name.replace("__", ".")
# MAGIC
# MAGIC     # Define a wrapper that accepts kwargs for the UC tool call,
# MAGIC     # then passes them to the UC tool execution client
# MAGIC     def exec_fn(**kwargs):
# MAGIC         function_result = uc_function_client.execute_function(udf_name, kwargs)
# MAGIC         # Return error message if execution fails, result value if not.
# MAGIC         if function_result.error is not None:
# MAGIC             return function_result.error
# MAGIC         else:
# MAGIC             return function_result.value
# MAGIC
# MAGIC     return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)
# MAGIC
# MAGIC # List to store information about all tools available to the agent.
# MAGIC TOOL_INFOS = []
# MAGIC
# MAGIC # UDFs in Unity Catalog can be exposed as agent tools.
# MAGIC # The following code enables a python code interpreter tool using the system.ai.python_exec UDF.
# MAGIC
# MAGIC # TODO: Add additional tools
# MAGIC UC_TOOL_NAMES = ["system.ai.python_exec"]
# MAGIC
# MAGIC uc_function_client = get_uc_function_client()
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC for tool_spec in uc_toolkit.tools:
# MAGIC     TOOL_INFOS.append(create_tool_info(tool_spec))
# MAGIC
# MAGIC
# MAGIC # Use Databricks vector search indexes as tools
# MAGIC # See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# MAGIC # List to store vector search tool instances for unstructured retrieval.
# MAGIC VECTOR_SEARCH_TOOLS = []
# MAGIC
# MAGIC # To add vector search retriever tools,
# MAGIC # use VectorSearchRetrieverTool and create_tool_info,
# MAGIC # then append the result to TOOL_INFOS.
# MAGIC # Example:
# MAGIC # VECTOR_SEARCH_TOOLS.append(
# MAGIC #     VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # )
# MAGIC
# MAGIC for vs_tool in VECTOR_SEARCH_TOOLS:
# MAGIC     TOOL_INFOS.append(create_tool_info(vs_tool.tool, vs_tool.execute))
# MAGIC
# MAGIC
# MAGIC class ToolCallingAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     Class representing a tool-calling Agent.
# MAGIC     Handles both tool execution via exec_fn and LLM interactions via model serving.
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
# MAGIC         """Initializes the ToolCallingAgent with tools."""
# MAGIC         self.llm_endpoint = llm_endpoint
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         self.model_serving_client: OpenAI = (
# MAGIC             self.workspace_client.serving_endpoints.get_open_ai_client()
# MAGIC         )
# MAGIC         # Internal message list holds conversation state in completion-message format
# MAGIC         self.messages: list[dict[str, Any]] = None
# MAGIC         self._tools_dict = {tool.name: tool for tool in tools}
# MAGIC
# MAGIC     def get_tool_specs(self) -> list[dict]:
# MAGIC         """Returns tool specifications in the format OpenAI expects."""
# MAGIC         return [tool_info.spec for tool_info in self._tools_dict.values()]
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC     def execute_tool(self, tool_name: str, args: dict) -> Any:
# MAGIC         """Executes the specified tool with the given arguments."""
# MAGIC         return self._tools_dict[tool_name].exec_fn(**args)
# MAGIC
# MAGIC     def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
# MAGIC         """Convert from a Responses API output item to  a list of ChatCompletion messages."""
# MAGIC         msg_type = message.get("type")
# MAGIC         if msg_type == "function_call":
# MAGIC             return [
# MAGIC                 {
# MAGIC                     "role": "assistant",
# MAGIC                     "content": "tool call",  # empty content is not supported by claude models
# MAGIC                     "tool_calls": [
# MAGIC                         {
# MAGIC                             "id": message["call_id"],
# MAGIC                             "type": "function",
# MAGIC                             "function": {
# MAGIC                                 "arguments": message["arguments"],
# MAGIC                                 "name": message["name"],
# MAGIC                             },
# MAGIC                         }
# MAGIC                     ],
# MAGIC                 }
# MAGIC             ]
# MAGIC         elif msg_type == "message" and isinstance(message.get("content"), list):
# MAGIC             return [
# MAGIC                 {"role": message["role"], "content": content["text"]}
# MAGIC                 for content in message["content"]
# MAGIC             ]
# MAGIC         elif msg_type == "reasoning":
# MAGIC             return [{"role": "assistant", "content": json.dumps(message["summary"])}]
# MAGIC         elif msg_type == "function_call_output":
# MAGIC             return [
# MAGIC                 {
# MAGIC                     "role": "tool",
# MAGIC                     "content": message["output"],
# MAGIC                     "tool_call_id": message["call_id"],
# MAGIC                 }
# MAGIC             ]
# MAGIC         compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
# MAGIC         filtered = {k: v for k, v in message.items() if k in compatible_keys}
# MAGIC         return [filtered] if filtered else []
# MAGIC
# MAGIC     def prep_msgs_for_llm(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
# MAGIC         """Filter out message fields that are not compatible with LLM message formats and convert from Responses API to ChatCompletion compatible"""
# MAGIC         chat_msgs = []
# MAGIC         for msg in messages:
# MAGIC             chat_msgs.extend(self._responses_to_cc(msg))
# MAGIC         return chat_msgs
# MAGIC
# MAGIC     @backoff.on_exception(backoff.expo, openai.RateLimitError)
# MAGIC     @mlflow.trace(span_type=SpanType.LLM)
# MAGIC     def call_llm(self) -> Generator[dict[str, Any], None, None]:
# MAGIC         for chunk in self.model_serving_client.chat.completions.create(
# MAGIC             model=self.llm_endpoint,
# MAGIC             messages=self.prep_msgs_for_llm(self.messages),
# MAGIC             tools=self.get_tool_specs(),
# MAGIC             stream=True,
# MAGIC         ):
# MAGIC             yield chunk.to_dict()
# MAGIC
# MAGIC     def handle_tool_calls(
# MAGIC         self, tool_calls: list[dict[str, Any]]
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         """
# MAGIC         Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
# MAGIC         """
# MAGIC         for tool_call in tool_calls:
# MAGIC             function = tool_call["function"]
# MAGIC             args = json.loads(function["arguments"])
# MAGIC             # Cast tool result to a string, since not all tools return as tring
# MAGIC             result = str(self.execute_tool(tool_name=function["name"], args=args))
# MAGIC             self.messages.append(
# MAGIC                 {"role": "tool", "content": result, "tool_call_id": tool_call["id"]}
# MAGIC             )
# MAGIC             yield ResponsesAgentStreamEvent(
# MAGIC                 type="response.output_item.done",
# MAGIC                 item=self.create_function_call_output_item(
# MAGIC                     tool_call["id"],
# MAGIC                     result,
# MAGIC                 ),
# MAGIC             )
# MAGIC
# MAGIC     def call_and_run_tools(
# MAGIC         self,
# MAGIC         max_iter: int = 10,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         for _ in range(max_iter):
# MAGIC             last_msg = self.messages[-1]
# MAGIC             if tool_calls := last_msg.get("tool_calls", None):
# MAGIC                 yield from self.handle_tool_calls(tool_calls)
# MAGIC             elif last_msg.get("role", None) == "assistant":
# MAGIC                 return
# MAGIC             else:
# MAGIC                 # aggregate the chat completions stream to add to internal state
# MAGIC                 llm_content = ""
# MAGIC                 tool_calls = []
# MAGIC                 msg_id = None
# MAGIC                 for chunk in self.call_llm():
# MAGIC                     delta = chunk["choices"][0]["delta"]
# MAGIC                     msg_id = chunk.get("id", None)
# MAGIC                     content = delta.get("content", None)
# MAGIC                     if tc := delta.get("tool_calls"):
# MAGIC                         if not tool_calls:  # only accomodate for single tool call right now
# MAGIC                             tool_calls = tc
# MAGIC                         else:
# MAGIC                             tool_calls[0]["function"]["arguments"] += tc[0]["function"]["arguments"]
# MAGIC                     elif content is not None:
# MAGIC                         llm_content += content
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             **self.create_text_delta(content, item_id=msg_id)
# MAGIC                         )
# MAGIC                 llm_output = {"role": "assistant", "content": llm_content, "tool_calls": tool_calls}
# MAGIC                 self.messages.append(llm_output)
# MAGIC
# MAGIC                 # yield an `output_item.done` `output_text` event that aggregates the stream
# MAGIC                 # this enables tracing and payload logging
# MAGIC                 if llm_output["content"]:
# MAGIC                     yield ResponsesAgentStreamEvent(
# MAGIC                         type="response.output_item.done",
# MAGIC                         item=self.create_text_output_item(
# MAGIC                             llm_output["content"], msg_id
# MAGIC                         ),
# MAGIC                     )
# MAGIC                 # yield an `output_item.done` `function_call` event for each tool call
# MAGIC                 if tool_calls := llm_output.get("tool_calls", None):
# MAGIC                     for tool_call in tool_calls:
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             type="response.output_item.done",
# MAGIC                             item=self.create_function_call_item(
# MAGIC                                 str(uuid4()),
# MAGIC                                 tool_call["id"],
# MAGIC                                 tool_call["function"]["name"],
# MAGIC                                 tool_call["function"]["arguments"],
# MAGIC                             ),
# MAGIC                         )
# MAGIC
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC             type="response.output_item.done",
# MAGIC             item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
# MAGIC         )
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     @mlflow.trace(span_type=SpanType.AGENT)
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         self.messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
# MAGIC             i.model_dump() for i in request.input
# MAGIC         ]
# MAGIC         yield from self.call_and_run_tools()
# MAGIC
# MAGIC
# MAGIC # Log the model using MLflow
# MAGIC mlflow.openai.autolog()
# MAGIC AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
# MAGIC mlflow.models.set_model(AGENT)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since we manually traced methods within `ResponsesAgent`, you can view the trace for each step the agent takes, with any LLM calls made via the OpenAI SDK automatically traced by autologging.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

result = AGENT.predict({"input": [{"role": "user", "content": "What is 6*7 in Python"}]})
print(result.model_dump(exclude_none=True))

# COMMAND ----------

for chunk in AGENT.predict_stream(
    {"input": [{"role": "user", "content": "What is 6*7 in Python?"}]}
):
    print(chunk.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC ### Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends declaring resource dependencies for the agent upfront during logging. This enables automatic authentication passthrough when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model().`
# MAGIC
# MAGIC   - **TODO**: If your Unity Catalog tool queries a [vector search index](docs link) or leverages [external functions](docs link), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See docs ([AWS](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#resources)).
# MAGIC
# MAGIC

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
from agent import UC_TOOL_NAMES, VECTOR_SEARCH_TOOLS
import mlflow
from mlflow.models.resources import DatabricksFunction
from pkg_resources import get_distribution

resources = []
for tool in VECTOR_SEARCH_TOOLS:
    resources.extend(tool.resources)
for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        pip_requirements=[
            "databricks-openai",
            "backoff",
            f"databricks-connect=={get_distribution('databricks-connect').version}",
        ],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with Agent Evaluation
# MAGIC
# MAGIC Use Mosaic AI Agent Evaluation to evalaute the agent's responses based on expected responses and other evaluation criteria. Use the evaluation criteria you specify to guide iterations, using MLflow to track the computed quality metrics.
# MAGIC See Databricks documentation ([AWS]((https://docs.databricks.com/aws/generative-ai/agent-evaluation) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)).
# MAGIC
# MAGIC
# MAGIC To evaluate your tool calls, add custom metrics. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/agent-evaluation/custom-metrics.html#evaluating-tool-calls) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/custom-metrics#evaluating-tool-calls)).

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, RetrievalGroundedness, RetrievalRelevance, Safety

eval_dataset = [
    {
        "inputs": {"input": [{"role": "user", "content": "Calculate the 15th Fibonacci number"}]},
        "expected_response": "The 15th Fibonacci number is 610.",
    }
]

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[RelevanceToQuery(), Safety()],  # add more scorers here if they're applicable
)

# Review the evaluation results in the MLfLow UI (see console output)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks)).

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"input": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Before you deploy the agent, you must register the agent to Unity Catalog.
# MAGIC
# MAGIC - **TODO** Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "your_catalog"
schema = "your_schema"
model_name = "your_model"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"endpointSource": "docs"},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details