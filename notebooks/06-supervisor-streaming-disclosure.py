# Databricks notebook source
# MAGIC %md
# MAGIC # Multi-Agent Supervisor using FMAPI with Tool Calling
# MAGIC
# MAGIC This notebook demonstrates how to deploy a multi-agent supervisor that orchestrates domain-specific agents
# MAGIC using the MLflow Responses API (FMAPI) with tool calling. The supervisor features:
# MAGIC
# MAGIC - **Tool-based Architecture**: Treats domain agents as OpenAI function tools for maximum flexibility
# MAGIC - **Intelligent Routing**: Dynamic routing based on question analysis with LLM-powered planning
# MAGIC - **Progressive Disclosure**: Streams thought process and decision making to users
# MAGIC - **Comprehensive Coverage**: Handles both chemical data analysis and computational genomics domains
# MAGIC
# MAGIC In this notebook you learn to:
# MAGIC
# MAGIC - Deploy a supervisor agent that coordinates multiple domain experts
# MAGIC - Test the agent's intelligent routing and tool calling capabilities
# MAGIC - Evaluate the agent using Mosaic AI Agent Evaluation
# MAGIC - Set up proper authentication for multiple domain agent endpoints
# MAGIC - Log and deploy the supervisor agent
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Access to both domain agent endpoints: `your_agent_endpoint_1` and `your_agent_endpoint_2`
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()



# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the Supervisor Agent
# MAGIC
# MAGIC The supervisor agent is implemented in our source code with the following key features:
# MAGIC 
# MAGIC ### Agent Architecture
# MAGIC - **CoatingsSupervisorAgent**: Handles market news search and manufacturing plant data analysis
# MAGIC - **GenomicsSupervisorAgent**: Handles Python/mathematical execution and patient data queries
# MAGIC
# MAGIC ### Key Components
# MAGIC - **RegistryToolCallingAgent**: FMAPI-compliant supervisor that orchestrates tool calls
# MAGIC - **AgentRegistry**: In-memory registry of domain experts with fuzzy resolution
# MAGIC - **DomainAgentExecutor**: Executes domain agent requests via Databricks Serving endpoints
# MAGIC - **Progressive Disclosure**: Streams thought process and decision making to users

# COMMAND ----------

# Import the supervisor agent from our source code
import sys
import os
sys.path.append('/Workspace/src')

from multi_agent.supervisor.fmapi_supervisor_agent import (
    AGENT,
    DEFAULT_AGENTS,
    SupervisorConfig,
    ResourceConfig,
    build_registry,
    create_registry_supervisor_agent,
)

print("Available Domain Agents:")
for agent in DEFAULT_AGENTS:
    print(f"- {agent.name}: {agent.description}")
    print(f"  Endpoint: {agent.endpoint}")
    if agent.resources.genie_spaces:
        print(f"  Genie Spaces: {agent.resources.genie_spaces}")
    if agent.resources.vector_search_indices:
        print(f"  Vector Search Indices: {agent.resources.vector_search_indices}")
    if agent.resources.functions:
        print(f"  Functions: {agent.resources.functions}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the Supervisor Agent
# MAGIC
# MAGIC The supervisor agent demonstrates intelligent routing by analyzing questions and creating execution plans.
# MAGIC It provides progressive disclosure of its thinking process and can synthesize responses from multiple domain experts.

# COMMAND ----------

# Test with a chemical industry question
print("=== Testing Question ===")
chemical_result = AGENT.predict({
    "input": [{"role": "user", "content": "What are the latest trends in automotive coating technologies and their impact on manufacturing processes?"}]
})
print("Response structure:", chemical_result.model_dump(exclude_none=True)['output'][0].keys())
print()

# COMMAND ----------

# Test with a genomics/computational question  
print("=== Testing Question ===")
genomics_result = AGENT.predict({
    "input": [{"role": "user", "content": "Can you analyze a gene expression dataset and calculate statistical significance using Python?"}]
})
print("Response structure:", genomics_result.model_dump(exclude_none=True)['output'][0].keys())
print()

# COMMAND ----------

# Test with a question that might require multiple agents
print("=== Testing Complex Multi-Domain Question ===")
complex_result = AGENT.predict({
    "input": [{"role": "user", "content": "How can computational models help optimize chemical coating formulations for better performance?"}]
})
print("Response structure:", complex_result.model_dump(exclude_none=True)['output'][0].keys())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Response Test
# MAGIC
# MAGIC Test the FMAPI streaming feature by observing the supervisor's thought process and tool calls in real-time.

# COMMAND ----------

print("=== Testing FMAPI Streaming with Tool Calls ===")
for chunk in AGENT.predict_stream({
    "input": [{"role": "user", "content": "What are the computational challenges in genomics data analysis and how do they relate to chemical data processing?"}]
}):
    chunk_data = chunk.model_dump(exclude_none=True)
    
    # Show different types of streaming events
    if chunk.type == "response.text.delta":
        print(f"[STREAM] {chunk_data.get('delta', {}).get('text', '')}", end='')
    elif chunk.type == "response.output_item.done":
        item = chunk_data.get('item', {})
        if item.get('type') == 'output_text':
            print(f"\n[OUTPUT] Text item completed: {len(item.get('content', ''))} characters")
        elif item.get('type') == 'function_call':
            print(f"\n[OUTPUT] Function call: {item.get('name', 'unknown')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the FMAPI Supervisor Agent as an MLflow Model
# MAGIC
# MAGIC Log the supervisor agent with proper resource dependencies for automatic authentication.
# MAGIC This includes both domain agent endpoints to ensure secure access during deployment.

# COMMAND ----------

import mlflow
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex
)
from pkg_resources import get_distribution

# Define all endpoints used by the supervisor
domain_agent_endpoints = [agent.endpoint for agent in DEFAULT_AGENTS]

# Attach resources for automatic authentication passthrough
resources = [
    # Supervisor LLM endpoint (use default from supervisor config)
    DatabricksServingEndpoint(endpoint_name=SupervisorConfig().llm_endpoint),
]

# Dynamically add all resources from DEFAULT_AGENTS configuration
for agent in DEFAULT_AGENTS:
    # Add the serving endpoint
    resources.append(DatabricksServingEndpoint(endpoint_name=agent.endpoint))

    # Add Genie spaces
    for genie_space_id in agent.resources.genie_spaces:
        resources.append(DatabricksGenieSpace(genie_space_id=genie_space_id))

    # Add vector search indices
    for index_name in agent.resources.vector_search_indices:
        resources.append(DatabricksVectorSearchIndex(index_name=index_name))

    # Add functions
    for function_name in agent.resources.functions:
        resources.append(DatabricksFunction(function_name=function_name))

print("Supervisor agent resources configured for authentication:")
print(f"- Supervisor LLM: {SupervisorConfig().llm_endpoint}")
for endpoint in domain_agent_endpoints:
    print(f"- Domain agent: {endpoint}")
print(f"- Total resources: {len(resources)} (endpoints, genie spaces, vector search, functions)")

# COMMAND ----------

# Define input example for the supervisor agent
input_example = {
    "input": [
        {
            "role": "user",
            "content": "What are the latest trends in automotive coating technologies?"
        }
    ]
}

# Log the model with source code included for proper deployment
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="supervisor_agent",
        python_model=AGENT,  # Use the existing agent instance
        code_paths=[
            "src/multi_agent/supervisor/fmapi_supervisor_agent.py",
            "src/multi_agent/supervisor/agent_configs.py"
        ],
        input_example=input_example,
        extra_pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-openai=={get_distribution('databricks-openai').version}",
            f"backoff=={get_distribution('backoff').version}",
        ],
        resources=resources,
        streamable=True,
        model_config={
            "agent_type": "supervisor_fmapi",
            "architecture": "tool_calling_supervisor",
            "domain_agents": {
                agent.name: {
                    "endpoint": agent.endpoint,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "domain": agent.domain,
                }
                for agent in DEFAULT_AGENTS
            },
            "supervisor_llm": SupervisorConfig().llm_endpoint,
            "max_iterations": 10,
            "source_files_logged": [
                "src/multi_agent/supervisor/fmapi_supervisor_agent.py",
                "src/multi_agent/supervisor/agent_configs.py"
            ]
        }
    )

    # Log additional metadata
    mlflow.log_param("supervisor_llm", SupervisorConfig().llm_endpoint)
    mlflow.log_param("num_domain_agents", len(DEFAULT_AGENTS))
    mlflow.log_param("num_resources", len(resources))

    # Create a summary of domain agents
    agent_summary = {}
    for agent in DEFAULT_AGENTS:
        agent_summary[agent.name] = {
            "endpoint": agent.endpoint,
            "description": agent.description,
        }

    mlflow.log_dict(agent_summary, "domain_agents_summary.json")

    # Create a summary of resources
    resource_summary = {}
    for resource in resources:
        resource_type = type(resource).__name__
        if resource_type not in resource_summary:
            resource_summary[resource_type] = []
        resource_summary[resource_type].append(str(resource.__dict__))

    mlflow.log_dict(resource_summary, "resource_summary.json")

print(f"Supervisor agent logged successfully!")
print(f"Run ID: {logged_agent_info.run_id}")
print(f"Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the FMAPI Supervisor Agent
# MAGIC
# MAGIC Use Mosaic AI Agent Evaluation to evaluate the supervisor's routing decisions,
# MAGIC response quality, and tool calling capabilities across different domain queries.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# Create evaluation dataset covering both domains and multi-domain scenarios
eval_dataset = [
    {
        "inputs": {"input": [{"role": "user", "content": "Recommend some classic rock songs for a road trip playlist"}]},
        "expected_response": "Classic rock songs perfect for road trips include Born to Run by Bruce Springsteen, Life is a Highway by Tom Cochrane, Take It Easy by Eagles, and Sweet Home Alabama by Lynyrd Skynyrd.",
    },
    {
        "inputs": {"input": [{"role": "user", "content": "What's the best time to visit Japan for cherry blossoms?"}]},
        "expected_response": "The best time to visit Japan for cherry blossoms is typically late March to early April, depending on the region. Tokyo usually peaks in late March, while Kyoto peaks in early April.",
    },
    {
        "inputs": {"input": [{"role": "user", "content": "Suggest music for a relaxing beach vacation"}]},
        "expected_response": "For a relaxing beach vacation, consider chill music like reggae (Bob Marley), acoustic folk (Jack Johnson), ambient electronic (Bonobo), or tropical house music that matches the laid-back beach atmosphere.",
    },
    {
        "inputs": {"input": [{"role": "user", "content": "How much does a week-long trip to Paris cost?"}]},
        "expected_response": "A week-long trip to Paris typically costs $1,500-$3,000 per person including flights, mid-range hotels, meals, and attractions, depending on season and travel style. Budget travelers can manage $1,000 while luxury travelers might spend $5,000+.",
    },
    {
        "inputs": {"input": [{"role": "user", "content": "What jazz artists should I listen to if I like Miles Davis?"}]},
        "expected_response": "If you like Miles Davis, try other jazz greats like John Coltrane, Charlie Parker, Thelonious Monk, Billie Holiday, Duke Ellington, and contemporary artists like Kamasi Washington or Robert Glasper.",
    }
]

print(f"Evaluating supervisor agent with {len(eval_dataset)} test cases...")

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[
        RelevanceToQuery(),
        Safety()
    ],
        eval_name="supervisor_fmapi_eval"
)

print("Evaluation completed! Check MLflow UI for detailed results.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment Validation
# MAGIC
# MAGIC Validate the logged supervisor agent before deployment to ensure proper functionality
# MAGIC and resource access.

# COMMAND ----------

# Test the logged model
print("=== Validating Logged Model ===")
validation_result = mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/supervisor_agent",
    input_data=input_example,
    env_manager="uv",
)

print("Validation successful!")
print(f"Response type: {type(validation_result)}")
print(f"Response structure: {list(validation_result.keys()) if isinstance(validation_result, dict) else 'Not a dict'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the FMAPI Supervisor Agent to Unity Catalog
# MAGIC
# MAGIC Register the supervisor agent to Unity Catalog for deployment and governance.
# MAGIC
# MAGIC - **TODO** Update the `catalog`, `schema`, and `model_name` below to register the supervisor agent.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "your_catalog"
schema = "your_schema"  
model_name = "supervisor_fmapi_tool_calling"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

print(f"Unity Catalog model name: {UC_MODEL_NAME}")

# Uncomment to register the model to UC
# uc_registered_model_info = mlflow.register_model(
#     model_uri=logged_agent_info.model_uri, 
#     name=UC_MODEL_NAME
# )
# print(f"Model registered to Unity Catalog: {UC_MODEL_NAME}")
# print(f"Version: {uc_registered_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the FMAPI Supervisor Agent
# MAGIC
# MAGIC Deploy the supervisor agent with automatic authentication for domain agent endpoints.

# COMMAND ----------

from databricks import agents

# Uncomment to deploy the agent (requires UC registration first)
# agents.deploy(
#     UC_MODEL_NAME,
#     uc_registered_model_info.version,
#     tags={"endpointSource": "supervisor", "architecture": "fmapi_tool_calling", "RemoveAfter": "2025-12-31"},
#     environment_vars={
#         "DB_MODEL_SERVING_HOST_URL": "<put your workspace URL here>"
#     },
#     endpoint_name="supervisor_fmapi_tool_calling"
# )
# print(f"Supervisor agent deployed successfully to endpoint: supervisor_fmapi_tool_calling")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Deployment (Optional)
# MAGIC
# MAGIC Once deployed, you can test the supervisor agent endpoint directly.

# COMMAND ----------

# Example deployment testing (uncomment after deployment)
# from databricks.sdk import WorkspaceClient
# 
# w = WorkspaceClient()
# 
# # Test the deployed endpoint
# response = w.serving_endpoints.query(
#     name="supervisor_fmapi_tool_calling",
#     inputs={"input": [{"role": "user", "content": "Test the deployed FMAPI supervisor agent with tool calling"}]}
# )
# 
# print("Deployment test successful!")
# print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC After your supervisor agent is deployed, you can:
# MAGIC
# MAGIC 1. **Chat in AI Playground**: Test the progressive disclosure and routing capabilities
# MAGIC 2. **Share with SMEs**: Get feedback on domain expertise and routing accuracy  
# MAGIC 3. **Monitor Performance**: Track routing decisions and response quality
# MAGIC 4. **Extend Domains**: Add new domain agents by updating `AVAILABLE_AGENTS`
# MAGIC 5. **Production Integration**: Embed in applications requiring multi-domain expertise
# MAGIC
# MAGIC ### Key Features to Explore:
# MAGIC - **Intelligent Routing**: Ask questions spanning multiple domains
# MAGIC - **Tool Calling**: Observe the supervisor's tool selection and execution process
# MAGIC - **Response Synthesis**: Test complex questions requiring multiple expert consultations
# MAGIC - **Streaming Experience**: Experience real-time thought process and tool calls
# MAGIC
# MAGIC The FMAPI supervisor agent provides a scalable pattern for orchestrating domain experts
# MAGIC while maintaining transparency and user engagement through tool calling and streaming.
