# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Create a Supervisor Agent to Orchestrate other Supervisor Agents
# MAGIC
# MAGIC In this notebook, we demonstrate how to create a Supervisor Agent that delegates queries to two other specialized supervisor agents:
# MAGIC
# MAGIC 1. **company_data_agent**:
# MAGIC    - Contains a Genie agent for Gbuilt data.
# MAGIC    - Includes a vector search tool agent for Valona data.
# MAGIC
# MAGIC 2. **genomics_tools_agent**:
# MAGIC    - Contains a Genie agent for patient genomics data.
# MAGIC    - Includes two function tool agents: one for mathematical computations and another for Python code execution.
# MAGIC
# MAGIC This setup enables modular, scalable handling of diverse data sources and computational tasks by leveraging specialized agents under a unified supervisor.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC If you want to call gbuilt agent use gbuilt tag:
# MAGIC
# MAGIC
# MAGIC **company_data: company_data_agent**:
# MAGIC    - Contains a Genie agent for Gbuilt data.
# MAGIC    - Includes a vector search tool agent for Valona data.
# MAGIC
# MAGIC **genomics: genomics_tools_agent**:
# MAGIC    - Contains a Genie agent for patient genomics data.
# MAGIC    - Includes two function tool agents: one for mathematical computations and another for Python code execution.

# COMMAND ----------

# MAGIC %pip install -U -qqq mlflow langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %%writefile supervisor_of_supervisors.py
# MAGIC import functools
# MAGIC import os
# MAGIC import uuid
# MAGIC from typing import Any, Generator, Literal, Optional
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC )
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
# MAGIC from mlflow.deployments import get_deploy_client
# MAGIC from pydantic import BaseModel
# MAGIC
# MAGIC ###################################################
# MAGIC ## Create agents from existing endpoints
# MAGIC ###################################################
# MAGIC
# MAGIC # Company Data Agent (Gbuilt data + Valona vector search)
# MAGIC company_data_agent_description = (
# MAGIC     "The Company Data assistant has access to Gbuilt data from Company and can perform vector search "
# MAGIC     "on Valona market insights data. Use this agent for queries about Company-related data, chemical information, "
# MAGIC     "or when searching through Company documentation and datasets."
# MAGIC )
# MAGIC
# MAGIC COMPANY_DATA_ENDPOINT = "your_company_data_endpoint"
# MAGIC company_data_model = ChatDatabricks(endpoint=COMPANY_DATA_ENDPOINT)
# MAGIC company_data_agent = create_react_agent(company_data_model, [])
# MAGIC
# MAGIC # Genomics Tools Agent (patient genomics data + computational tools)
# MAGIC genomics_tools_agent_description = (
# MAGIC     "The Genomics Tools assistant has access to patient genomics data and can perform "
# MAGIC     "mathematical computations and execute Python code. Use this agent for genomics analysis, "
# MAGIC     "data processing, calculations, or when computational tools are needed."
# MAGIC )
# MAGIC
# MAGIC GENOMICS_TOOLS_ENDPOINT = "your_genomics_tools_endpoint"
# MAGIC genomics_tools_model = ChatDatabricks(endpoint=GENOMICS_TOOLS_ENDPOINT)
# MAGIC genomics_tools_agent = create_react_agent(genomics_tools_model, [])
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC
# MAGIC # Multi-agent orchestration using Claude Sonnet 4
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC #############################
# MAGIC # Define the supervisor agent
# MAGIC #############################
# MAGIC
# MAGIC # Maximum number of iterations between supervisor and worker nodes
# MAGIC MAX_ITERATIONS = 3
# MAGIC
# MAGIC worker_descriptions = {
# MAGIC     "Company_Data": company_data_agent_description,
# MAGIC     "Genomics_Tools": genomics_tools_agent_description,
# MAGIC }
# MAGIC
# MAGIC formatted_descriptions = "\n".join(
# MAGIC     f"- {name}: {desc}" for name, desc in worker_descriptions.items()
# MAGIC )
# MAGIC
# MAGIC options = ["FINISH"] + list(worker_descriptions.keys())
# MAGIC
# MAGIC def supervisor_agent(state):
# MAGIC     count = state.get("iteration_count", 0) + 1
# MAGIC     print('iteration count', count)
# MAGIC     
# MAGIC     # Check max iterations
# MAGIC     if count > MAX_ITERATIONS:
# MAGIC         return {"next_node": "FINISH"}
# MAGIC     
# MAGIC     # Check if we have any agent responses
# MAGIC     messages = state.get("messages", [])
# MAGIC     agent_responses = [msg for msg in messages if msg.get("role") == "assistant" and msg.get("name")]
# MAGIC     
# MAGIC     # Build context about latest response if available
# MAGIC     latest_context = ""
# MAGIC     if agent_responses:
# MAGIC         latest_agent = agent_responses[-1]
# MAGIC         latest_context = (
# MAGIC             f"\n\nLatest response from {latest_agent.get('name', 'unknown')}:\n"
# MAGIC             f"{latest_agent.get('content', '')[:500]}...\n\n"
# MAGIC             f"IMPORTANT: If this response completely answers the user's question, choose FINISH."
# MAGIC         )
# MAGIC     
# MAGIC     # Single unified prompt
# MAGIC     system_prompt = (
# MAGIC         f"You are a supervisor managing these assistants:\n"
# MAGIC         f"{formatted_descriptions}\n\n"
# MAGIC         f"Given the conversation history, respond with the assistant to act next.\n"
# MAGIC         f"DECISION RULES:\n"
# MAGIC         f"1. If ANY agent has provided a complete answer to the user's question, respond with FINISH\n"
# MAGIC         f"2. Only call another agent if the current answer is incomplete or incorrect\n"
# MAGIC         f"3. Do NOT call additional agents just to verify or add to a complete answer"
# MAGIC         f"{latest_context}"
# MAGIC     )
# MAGIC     
# MAGIC     class nextNode(BaseModel):
# MAGIC         next_node: Literal[tuple(options)]
# MAGIC         reasoning: str  # Optional: helps debug decisions
# MAGIC     
# MAGIC     preprocessor = RunnableLambda(
# MAGIC         lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
# MAGIC     )
# MAGIC     supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
# MAGIC     
# MAGIC     result = supervisor_chain.invoke(state)
# MAGIC     print(f"Supervisor decision: {result.next_node}, Reasoning: {result.reasoning}")
# MAGIC     
# MAGIC     # If routed back to the same node that just responded, finish
# MAGIC     if agent_responses and result.next_node == agent_responses[-1].get("name"):
# MAGIC         print(f"Supervisor trying to route back to {result.next_node}, forcing FINISH")
# MAGIC         return {"next_node": "FINISH"}
# MAGIC     
# MAGIC     return {
# MAGIC         "iteration_count": count,
# MAGIC         "next_node": result.next_node
# MAGIC     }
# MAGIC #######################################
# MAGIC # Define our multiagent graph structure
# MAGIC #######################################
# MAGIC
# MAGIC def agent_node(state, agent, name):
# MAGIC     result = agent.invoke(state)
# MAGIC     # Extract the content from the result
# MAGIC     if "messages" in result and len(result["messages"]) > 0:
# MAGIC         last_message = result["messages"][-1]
# MAGIC         # Handle different message formats
# MAGIC         if hasattr(last_message, "content"):
# MAGIC             content = last_message.content
# MAGIC         elif isinstance(last_message, dict) and "content" in last_message:
# MAGIC             content = last_message["content"]
# MAGIC         else:
# MAGIC             content = str(last_message)
# MAGIC         
# MAGIC         # Ensure content is a string
# MAGIC         if not isinstance(content, str):
# MAGIC             content = str(content)
# MAGIC     else:
# MAGIC         content = "No response from agent"
# MAGIC     
# MAGIC     return {
# MAGIC         "messages": [
# MAGIC             {
# MAGIC                 "role": "assistant",
# MAGIC                 "content": content,
# MAGIC                 "name": name,
# MAGIC             }
# MAGIC         ]
# MAGIC     }
# MAGIC
# MAGIC def final_answer(state):
# MAGIC     system_prompt = "Using only the content in the messages, respond to the user's question using the answer given by the other agents."
# MAGIC     preprocessor = RunnableLambda(
# MAGIC         lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
# MAGIC     )
# MAGIC     final_answer_chain = preprocessor | llm
# MAGIC     return {"messages": [final_answer_chain.invoke(state)]}
# MAGIC
# MAGIC class AgentState(ChatAgentState):
# MAGIC     next_node: str
# MAGIC     iteration_count: int
# MAGIC
# MAGIC company_data_node = functools.partial(agent_node, agent=company_data_agent, name="Company_Data")
# MAGIC genomics_tools_node = functools.partial(agent_node, agent=genomics_tools_agent, name="Genomics_Tools")
# MAGIC
# MAGIC workflow = StateGraph(AgentState)
# MAGIC workflow.add_node("Company_Data", company_data_node)
# MAGIC workflow.add_node("Genomics_Tools", genomics_tools_node)
# MAGIC workflow.add_node("supervisor", supervisor_agent)
# MAGIC workflow.add_node("final_answer", final_answer)
# MAGIC
# MAGIC workflow.set_entry_point("supervisor")
# MAGIC
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
# MAGIC         response_messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 node_messages = node_data.get("messages", [])
# MAGIC                 for msg in node_messages:
# MAGIC                     # Handle different message formats
# MAGIC                     if isinstance(msg, dict):
# MAGIC                         # Ensure all required fields are present
# MAGIC                         message_dict = {
# MAGIC                             "id": str(uuid.uuid4()),
# MAGIC                             "role": msg.get("role", "assistant"),
# MAGIC                             "content": str(msg.get("content", ""))
# MAGIC                         }
# MAGIC                         # Add optional fields if present
# MAGIC                         if "name" in msg:
# MAGIC                             message_dict["name"] = msg["name"]
# MAGIC                         response_messages.append(ChatAgentMessage(**message_dict))
# MAGIC                     elif hasattr(msg, "role") and hasattr(msg, "content"):
# MAGIC                         # If it's already a message object
# MAGIC                         response_messages.append(ChatAgentMessage(
# MAGIC                             id=str(uuid.uuid4()),
# MAGIC                             role=msg.role,
# MAGIC                             content=str(msg.content)
# MAGIC                         ))
# MAGIC                     else:
# MAGIC                         # Fallback for other formats
# MAGIC                         response_messages.append(ChatAgentMessage(
# MAGIC                             id=str(uuid.uuid4()),
# MAGIC                             role="assistant",
# MAGIC                             content=str(msg)
# MAGIC                         ))
# MAGIC         
# MAGIC         return ChatAgentResponse(messages=response_messages)
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
# MAGIC                 node_messages = node_data.get("messages", [])
# MAGIC                 for msg in node_messages:
# MAGIC                     # Create chunk from message
# MAGIC                     if isinstance(msg, dict):
# MAGIC                         chunk_data = {
# MAGIC                             "role": msg.get("role", "assistant"),
# MAGIC                             "content": str(msg.get("content", ""))
# MAGIC                         }
# MAGIC                         if "name" in msg:
# MAGIC                             chunk_data["name"] = msg["name"]
# MAGIC                     else:
# MAGIC                         chunk_data = {
# MAGIC                             "role": "assistant",
# MAGIC                             "content": str(msg)
# MAGIC                         }
# MAGIC                     yield ChatAgentChunk(delta=chunk_data)
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
# MAGIC
# MAGIC ## Test Company Data Agent

# COMMAND ----------

from supervisor_of_supervisors import AGENT

input_example = {"messages": [{"role": "user", "content": "Please calculate 2x2x3x6x9/2 with a math tool"}]}

AGENT.predict(input_example)

# COMMAND ----------

from supervisor_of_supervisors import AGENT

input_example = {"messages": [{"role": "user", "content": "List who's case id's have year_of_birth after 1970 and year_of_death in 2010 "}]}

AGENT.predict(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Test Genomics Tools Agent

# COMMAND ----------

from supervisor_of_supervisors import AGENT

input_example = {"messages": [{"role": "user", "content": "Get me the automotive news from Hyundai"}]}

AGENT.predict(input_example)

# COMMAND ----------

from supervisor_of_supervisors import AGENT

input_example = {"messages": [{"role": "user", "content": "List customer plants in germany"}]}

AGENT.predict(input_example)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Log the agent as an MLflow model
# MAGIC
# MAGIC Log the agent as code from the `05-supervisor-of-supervisors.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).
# MAGIC
# MAGIC ### Enable automatic authentication for Databricks resources
# MAGIC For the most common Databricks resource types, Databricks supports and recommends `declaring resource dependencies for the agent upfront` during logging. This enables **automatic authentication passthrough** when you deploy the agent. With automatic authentication passthrough, Databricks automatically provisions, rotates, and manages short-lived credentials to securely access these resource dependencies from within the agent endpoint.
# MAGIC
# MAGIC To enable automatic authentication, specify the dependent Databricks resources when calling `mlflow.pyfunc.log_model().`

# COMMAND ----------

import mlflow
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex
)
from pkg_resources import get_distribution
from supervisor_of_supervisors import AGENT, LLM_ENDPOINT_NAME, COMPANY_DATA_ENDPOINT, GENOMICS_TOOLS_ENDPOINT

resources = [
    # Supervisor endpoint
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    
    # Sub-agent endpoints
    DatabricksServingEndpoint(endpoint_name=COMPANY_DATA_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=GENOMICS_TOOLS_ENDPOINT),
    
    # Company agent resources
    DatabricksGenieSpace(genie_space_id="your_genie_space_id_1"),
    DatabricksVectorSearchIndex(index_name="your_catalog.your_schema.your_index"),
    
    # Genomics agent resources  
    DatabricksGenieSpace(genie_space_id="your_genie_space_id_2"),
    DatabricksFunction(function_name="your_catalog.your_schema.compute_math"),
    DatabricksFunction(function_name="your_catalog.your_schema.execute_python_code"),
]

# Define input example
input_example = {
    "messages": [
        {
            "role": "user",
            "content":  "Please calculate 2x2x3x6x9/2"
        }
    ]
}

# Log the supervisor agent
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="supervisor_agent",
        python_model="supervisor_of_supervisors.py",
        input_example=input_example,
        extra_pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"databricks-agents=={get_distribution('databricks-agents').version}",
            f"databricks-sdk=={get_distribution('databricks-sdk').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"pydantic=={get_distribution('pydantic').version}",
        ],
        resources=resources,
        streamable=True,
        model_config={
            "agent_type": "supervisor",
            "sub_agents": {
                "Company_Data": COMPANY_DATA_ENDPOINT,
                "Genomics_Tools": GENOMICS_TOOLS_ENDPOINT
            },
            "max_iterations": 3,
            "llm_endpoint": LLM_ENDPOINT_NAME
        }
    )
    
    # Log additional metadata
    mlflow.log_param("supervisor_llm", LLM_ENDPOINT_NAME)
    mlflow.log_param("company_data_endpoint", COMPANY_DATA_ENDPOINT)
    mlflow.log_param("genomics_tools_endpoint", GENOMICS_TOOLS_ENDPOINT)
    mlflow.log_param("num_resources", len(resources))
    
    # Create a summary of resources
    resource_summary = {}
    for resource in resources:
        resource_type = type(resource).__name__
        if resource_type not in resource_summary:
            resource_summary[resource_type] = []
        resource_summary[resource_type].append(str(resource.__dict__))
    
    mlflow.log_dict(resource_summary, "resource_summary.json")

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/supervisor_agent",
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
model_name = "your_supervisor_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

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
        "DB_MODEL_SERVING_HOST_URL": "<put your workspace URL here>"
    },
    endpoint_name="your_supervisor_agent_endpoint"
)