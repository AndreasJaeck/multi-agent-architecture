# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # Create a Supervisor Agent to Orchestrate other Supervisor Agents
# MAGIC
# MAGIC In this notebook, we demonstrate how to create a Supervisor Agent that delegates queries to two other specialized supervisor agents:
# MAGIC
# MAGIC 1. **genie_multi_agent_basf**:  
# MAGIC    - Contains a Genie agent for Gbuilt data (from BASF).
# MAGIC    - Includes a vector search tool agent for Valona data (also from BASF).
# MAGIC
# MAGIC 2. **genie_multi_agent_basf_v2**:  
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
# MAGIC **gbuilt: genie_multi_agent_basf**:  
# MAGIC    - Contains a Genie agent for Gbuilt data (from BASF).
# MAGIC    - Includes a vector search tool agent for Valona data (also from BASF).
# MAGIC
# MAGIC **genomics: genie_multi_agent_basf_v2**:  
# MAGIC    - Contains a Genie agent for patient genomics data.
# MAGIC    - Includes two function tool agents: one for mathematical computations and another for Python code execution.

# COMMAND ----------

# MAGIC %pip install -U -qqq mlflow langgraph==0.3.4 databricks-langchain databricks-agents uv
# MAGIC dbutils.library.restartPython()


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
# MAGIC ## Test genie_multi_agent_basf

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
# MAGIC ## Test genie_multi_agent_basf_v2

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
from supervisor_of_supervisors import AGENT, LLM_ENDPOINT_NAME, BASF_DATA_ENDPOINT, GENOMICS_TOOLS_ENDPOINT

resources = [
    # Supervisor endpoint
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    
    # Sub-agent endpoints
    DatabricksServingEndpoint(endpoint_name=BASF_DATA_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=GENOMICS_TOOLS_ENDPOINT),
    
    # BASF agent resources
    DatabricksGenieSpace(genie_space_id="01f0273483ce143a9a12df723f5b960e"),
    DatabricksVectorSearchIndex(index_name="hong_zhu_demo_catalog.basf_genie_agent.valona_optimized_index"),
    
    # Genomics agent resources  
    DatabricksGenieSpace(genie_space_id="01f0671302ab1092bf22c090aa1d8fc2"),
    DatabricksFunction(function_name="hong_zhu_demo_catalog.basf_genie_agent.compute_math"),
    DatabricksFunction(function_name="hong_zhu_demo_catalog.basf_genie_agent.execute_python_code"),
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
                "BASF_Data": BASF_DATA_ENDPOINT,
                "Genomics_Tools": GENOMICS_TOOLS_ENDPOINT
            },
            "max_iterations": 3,
            "llm_endpoint": LLM_ENDPOINT_NAME
        }
    )
    
    # Log additional metadata
    mlflow.log_param("supervisor_llm", LLM_ENDPOINT_NAME)
    mlflow.log_param("basf_data_endpoint", BASF_DATA_ENDPOINT)
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
catalog = "hong_zhu_demo_catalog"
schema = "basf_genie_agent"
model_name = "supervisor_agent"
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
    endpoint_name="supervisor_agent_basf"
)