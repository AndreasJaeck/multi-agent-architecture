# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Agent Framework: Author and deploy a multi-agent system with Genie
# MAGIC
# MAGIC This notebook demonstrates how to build a multi-agent system using Mosaic AI Agent Framework and [LangGraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/), where [Genie](https://www.databricks.com/product/ai-bi/genie) is one of the agents.
# MAGIC In this notebook, you:
# MAGIC 1. Author a multi-agent system using LangGraph.
# MAGIC 1. Wrap the LangGraph agent with MLflow `ChatAgent` to ensure compatibility with Databricks features.
# MAGIC 1. Manually test the multi-agent system's output.
# MAGIC 1. Log and deploy the multi-agent system.
# MAGIC
# MAGIC This example is based on [LangGraph documentation - Multi-agent supervisor example](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/multi_agent/agent_supervisor.ipynb)
# MAGIC
# MAGIC ## Why use a Genie agent?
# MAGIC
# MAGIC Multi-agent systems consist of multiple AI agents working together, each with specialized capabilities. As one of those agents, Genie allows users to interact with their structured data using natural language.
# MAGIC
# MAGIC Unlike **SQL functions which can only run pre-defined queries**, Genie has the **flexibility to create novel queries** to answer user questions.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Create a Genie Space, see Databricks documentation ([AWS](https://docs.databricks.com/aws/genie/set-up) | [Azure](https://learn.microsoft.com/azure/databricks/genie/set-up)).

# COMMAND ----------

# MAGIC %pip install -U -qqq mlflow langgraph databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Define the multi-agent system
# MAGIC
# MAGIC Create a multi-agent system in LangGraph using a supervisor agent node directing the following agent nodes:
# MAGIC - **GenieAgent**: The Genie agent that queries and reasons over structured data.
# MAGIC - **Tool-calling agent**: An agent that calls Unity Catalog function tools.
# MAGIC
# MAGIC In this example, the tool-calling agent uses the Unity Catalog functions we created previously. There are also some other tools you can add to your agents, see Databricks documentation ([Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/agent-tool#agent-tool-examples)).
# MAGIC
# MAGIC
# MAGIC #### Wrap the LangGraph agent using the `ChatAgent` interface
# MAGIC
# MAGIC Databricks recommends using `ChatAgent` to ensure compatibility with Databricks AI features and to simplify authoring multi-turn conversational agents using an open source standard. 
# MAGIC
# MAGIC The `LangGraphChatAgent` class implements the `ChatAgent` interface to wrap the LangGraph agent.
# MAGIC
# MAGIC See MLflow's [ChatAgent documentation](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent).
# MAGIC
# MAGIC #### Write whole agent code to file
# MAGIC
# MAGIC Define the agent code in a single cell below. This lets you write the agent code to a local Python file, using the `%%writefile` magic command, for subsequent logging and deployment.
# MAGIC

# COMMAND ----------

# MAGIC %%writefile agent_v2.py
# MAGIC import functools
# MAGIC import os
# MAGIC from typing import Any, Generator, Literal, Optional
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC     VectorSearchRetrieverTool
# MAGIC )
# MAGIC from databricks_langchain.genie import GenieAgent
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt import create_react_agent
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC from pydantic import BaseModel
# MAGIC
# MAGIC ###################################################
# MAGIC ## Create a GenieAgent with access to a Genie Space
# MAGIC ###################################################
# MAGIC
# MAGIC # Add GENIE_SPACE_ID and a description for this space
# MAGIC # You can find the ID in the URL of the genie room /genie/rooms/<GENIE_SPACE_ID>
# MAGIC GENIE_SPACE_ID = "your_genie_space_id"
# MAGIC genie_agent_description = "The genie agent can analyze data related to genomics, including demographic information, medical diagnoses, exposures, and expression profiles."
# MAGIC
# MAGIC genie_agent = GenieAgent(
# MAGIC     genie_space_id=GENIE_SPACE_ID,
# MAGIC     genie_agent_name="Genie",
# MAGIC     description=genie_agent_description,
# MAGIC     client=WorkspaceClient(
# MAGIC         host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
# MAGIC         token=os.getenv("DATABRICKS_GENIE_PAT"),
# MAGIC     ),
# MAGIC )
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC
# MAGIC # TODO: Replace with your model serving endpoint
# MAGIC # multi-agent Genie works best with claude 3.7 or gpt 4o models.
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC
# MAGIC ############################################################
# MAGIC # Create a multi-tool agent
# MAGIC # You can also extend agents with access to additional tools
# MAGIC ############################################################
# MAGIC tools = []
# MAGIC
# MAGIC # TODO if desired, add additional tools and update the description of this agent
# MAGIC uc_tool_names = [
# MAGIC     "your_catalog.your_schema.compute_math",
# MAGIC     "your_catalog.your_schema.execute_python_code"
# MAGIC ]
# MAGIC
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC
# MAGIC # # Create and include a retriever tool for each vector search index
# MAGIC # # Direct pythhon integration
# MAGIC # vector_search_index_tools = [
# MAGIC #     VectorSearchRetrieverTool(
# MAGIC #         index_name="your_catalog.your_schema.your_index",
# MAGIC #         # TODO: specify index description for better agent tool selection
# MAGIC #         tool_name="find_news",
# MAGIC #         tool_description="Find latest news with similarity search based on description of content. This helps users to find latest news about about automotive and coatings industry. Returns results_itemId, title, created_date, modified_date, status, source, item_link, search_text, CAT_NAME, CAT_ID",
# MAGIC #         query_type="hybrid",
# MAGIC #         num_results=5
# MAGIC #     )
# MAGIC # ]
# MAGIC # tools.extend(vector_search_index_tools)
# MAGIC
# MAGIC multi_tool_agent_description = (
# MAGIC     "The multi tool agent can solve programming challenges, generating code snippets, debugging issues, and explaining complex coding concepts. Also able to compute mathematical functions and can analyze data related to genomics, including demographic information, medical diagnoses, exposures, and expression profiles.",
# MAGIC )
# MAGIC multi_tool_agent = create_react_agent(llm, tools=tools)
# MAGIC
# MAGIC #############################
# MAGIC # Define the supervisor agent
# MAGIC #############################
# MAGIC
# MAGIC # TODO update the max number of iterations between supervisor and worker nodes
# MAGIC # before returning to the user
# MAGIC MAX_ITERATIONS = 3
# MAGIC
# MAGIC worker_descriptions = {
# MAGIC     "Genie": genie_agent_description,
# MAGIC     "Multi-Tool-Agent": multi_tool_agent_description,
# MAGIC }
# MAGIC
# MAGIC # formatted_descriptions = "\n".join(
# MAGIC #     f"- {name}: {desc}" for name, desc in worker_descriptions.items()
# MAGIC # )
# MAGIC formatted_descriptions = (
# MAGIC     "\n".join(f"- {name}: {desc}" for name, desc in worker_descriptions.items()) +
# MAGIC     "\nIMPORTANT: Respond with FINISH as soon as any agent has provided a complete answer to the user's question." +
# MAGIC     "\nDo not call additional agents unless the current answer is insufficient or the question requires multiple perspectives."
# MAGIC )
# MAGIC
# MAGIC
# MAGIC system_prompt = f"Decide between routing between the following workers or ending the conversation if an answer is provided. \n{formatted_descriptions}"
# MAGIC options = ["FINISH"] + list(worker_descriptions.keys())
# MAGIC FINISH = {"next_node": "FINISH"}
# MAGIC
# MAGIC def supervisor_agent(state):
# MAGIC     count = state.get("iteration_count", 0) + 1
# MAGIC     if count > MAX_ITERATIONS:
# MAGIC         return FINISH
# MAGIC     
# MAGIC     class nextNode(BaseModel):
# MAGIC         next_node: Literal[tuple(options)]
# MAGIC
# MAGIC     preprocessor = RunnableLambda(
# MAGIC         lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
# MAGIC     )
# MAGIC     supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
# MAGIC     next_node = supervisor_chain.invoke(state).next_node
# MAGIC     
# MAGIC     # if routed back to the same node, exit the loop
# MAGIC     if state.get("next_node") == next_node:
# MAGIC         return FINISH
# MAGIC     return {
# MAGIC         "iteration_count": count,
# MAGIC         "next_node": next_node
# MAGIC     }
# MAGIC
# MAGIC #######################################
# MAGIC # Define our multiagent graph structure
# MAGIC #######################################
# MAGIC
# MAGIC def agent_node(state, agent, name):
# MAGIC     result = agent.invoke(state)
# MAGIC     return {
# MAGIC         "messages": [
# MAGIC             {
# MAGIC                 "role": "assistant",
# MAGIC                 "content": result["messages"][-1].content,
# MAGIC                 "name": name,
# MAGIC             }
# MAGIC         ]
# MAGIC     }
# MAGIC
# MAGIC
# MAGIC def final_answer(state):
# MAGIC     prompt = "Using only the content in the messages, respond to the previous user question using the answer given by the other agents without dublicating the answer."
# MAGIC     preprocessor = RunnableLambda(
# MAGIC         lambda state: state["messages"] + [{"role": "user", "content": prompt}]
# MAGIC     )
# MAGIC     final_answer_chain = preprocessor | llm
# MAGIC     return {"messages": [final_answer_chain.invoke(state)]}
# MAGIC
# MAGIC
# MAGIC class AgentState(ChatAgentState):
# MAGIC     next_node: str
# MAGIC     iteration_count: int
# MAGIC
# MAGIC
# MAGIC tool_node = functools.partial(agent_node, agent=multi_tool_agent, name="Multi-Tool-Agent")
# MAGIC genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")
# MAGIC
# MAGIC workflow = StateGraph(AgentState)
# MAGIC workflow.add_node("Genie", genie_node)
# MAGIC workflow.add_node("Multi-Tool-Agent", tool_node)
# MAGIC workflow.add_node("supervisor", supervisor_agent)
# MAGIC workflow.add_node("final_answer", final_answer)
# MAGIC
# MAGIC workflow.set_entry_point("supervisor")
# MAGIC # We want our workers to ALWAYS "report back" to the supervisor when done
# MAGIC for worker in worker_descriptions.keys():
# MAGIC     workflow.add_edge(worker, "supervisor")
# MAGIC
# MAGIC # Let the supervisor decide which next node to go
# MAGIC workflow.add_conditional_edges(
# MAGIC     "supervisor",
# MAGIC     lambda x: x["next_node"],
# MAGIC     {**{k: k for k in worker_descriptions.keys()}, "FINISH": "final_answer"},
# MAGIC )
# MAGIC workflow.add_edge("final_answer", END)
# MAGIC multi_agent = workflow.compile()
# MAGIC
# MAGIC
# MAGIC ###################################
# MAGIC # Wrap our multi-agent in ChatAgent
# MAGIC ###################################
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {
# MAGIC             "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
# MAGIC         }
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {
# MAGIC             "messages": [m.model_dump_compat(exclude_none=True) for m in messages]
# MAGIC         }
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg})
# MAGIC                     for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC mlflow.langchain.autolog()
# MAGIC AGENT = LangGraphChatAgent(multi_agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC **TODO**: Replace this placeholder `input_example` with a domain-specific prompt for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Personal Access Token (PAT) as a Databricks secret
# MAGIC In order to access the Genie Space and its underlying resources, we need to create a PAT
# MAGIC - This can either be your own PAT or that of a System Principal ([AWS](https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/oauth-m2m)). You will have to rotate this token yourself upon expiry.
# MAGIC - Add secrets-based environment variables to a model serving endpoint ([AWS](https://docs.databricks.com/aws/en/machine-learning/model-serving/store-env-variable-model-serving#add-secrets-based-environment-variables) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/store-env-variable-model-serving#add-secrets-based-environment-variables)).
# MAGIC - You can reference the table in the deploy docs for the right permissions level for each resource: ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent#automatic-authentication-passthrough) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/deploy-agent#automatic-authentication-passthrough)).
# MAGIC   - Provision with `CAN RUN` on the Genie Space
# MAGIC   - Provision with `CAN USE` on the SQL Warehouse powering the Genie Space
# MAGIC   - Provision with `SELECT` on underlying Unity Catalog Tables 
# MAGIC   - Provision with `EXECUTE` on underyling Unity Catalog Functions 

# COMMAND ----------

import os
from dbruntime.databricks_repl_context import get_context

# # Set secret_scope_name and secret_key_name to access your PAT
# secret_scope_name = ""
# secret_key_name = ""

# os.environ["DB_MODEL_SERVING_HOST_URL"] = "https://" + get_context().workspaceUrl
#os.environ["DATABRICKS_GENIE_PAT"] = dbutils.secrets.get(
#     scope=secret_scope_name, key=secret_key_name
# )

os.environ["DB_MODEL_SERVING_HOST_URL"] = "<put your workspace URL here>"
assert os.environ["DB_MODEL_SERVING_HOST_URL"] is not None

os.environ["DATABRICKS_GENIE_PAT"] = "<put your access token here>"
assert os.environ["DATABRICKS_GENIE_PAT"] is not None, (
    "The DATABRICKS_GENIE_PAT was not properly set to the PAT secret"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Test Genie Agent

# COMMAND ----------

from agent_v2 import AGENT

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Explain the datasets and capabilities that the Genie agent has access to.",
        }
    ]
}
# AGENT.predict(input_example)

for event in AGENT.predict_stream(input_example):
  print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Multi-Tool Agent - Genie

# COMMAND ----------

from agent_v2 import AGENT

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is the average year of birth of individuals in the demographics table?",
        }
    ]
}
AGENT.predict(input_example)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC ### Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends `declaring resource dependencies for the agent upfront` during logging. This enables **automatic authentication passthrough** when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model().`

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent_v2 import GENIE_SPACE_ID, LLM_ENDPOINT_NAME, tools
from databricks_langchain import UnityCatalogTool
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint
)
from pkg_resources import get_distribution
from mlflow.models import infer_signature

# Manually include underlying resources if needed.
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME), 
    DatabricksGenieSpace(genie_space_id=GENIE_SPACE_ID), 
]

for tool in tools:
    if isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent_v2",
        python_model="agent_v2.py",
        input_example=input_example,
        extra_pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"databricks-agents=={get_distribution('databricks-agents').version}",
            f"databricks-sdk=={get_distribution('databricks-sdk').version}",
            f"langgraph=={get_distribution('langgraph').version}"
        ],
        # Declare the resources to enable automatic credentials passthrough
        resources=resources,
        streamable=True,
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/model-serving/model-serving-debug.html#validate-inputs) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/model-serving-debug#before-model-deployment-validation-checks))."

# COMMAND ----------

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is the average year of birth of individuals in the demographics table?",
        }
    ]
}

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent_v2",
    input_data=input_example,
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "your_catalog"
schema = "your_schema"
model_name = "your_multi_agent_v2"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
from pyspark.sql import SparkSession

# This will make feedback app copy into fail 
#if SparkSession.getActiveSession():
#    SparkSession.getActiveSession().stop()

# "DATABRICKS_GENIE_PAT": f"{{{{secrets/{secret_scope_name}/{secret_key_name}}}}}"

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"endpointSource": "playground", "RemoveAfter": "2025-12-31"},
    environment_vars={
        "DATABRICKS_GENIE_PAT": "<put your access token here>",
        "DB_MODEL_SERVING_HOST_URL": "<put your workspace URL here>"
    },
    endpoint_name="your_multi_agent_endpoint_v2"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See Databricks documentation ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)).