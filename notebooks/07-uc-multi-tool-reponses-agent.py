# Databricks notebook source
# MAGIC %md
# MAGIC # UC Multi-Tool Responses Agent with FMAPI
# MAGIC
# MAGIC This notebook demonstrates how to deploy a UC Multi-Tool Responses Agent that integrates multiple tools
# MAGIC using the MLflow Responses API (FMAPI) with tool calling. The agent features:
# MAGIC
# MAGIC - **Multi-Tool Integration**: Combines Unity Catalog functions, vector search, and agent endpoints
# MAGIC - **Intelligent Tool Selection**: Dynamic tool calling based on question analysis with LLM-powered planning
# MAGIC - **Progressive Disclosure**: Streams thought process and decision making to users
# MAGIC - **Comprehensive Coverage**: Handles computational tasks, semantic search, and specialized agent interactions
# MAGIC
# MAGIC In this notebook you learn to:
# MAGIC
# MAGIC - Deploy a tool-calling agent that integrates multiple tool types
# MAGIC - Test the agent's intelligent tool selection and execution capabilities
# MAGIC - Evaluate the agent using Mosaic AI Agent Evaluation
# MAGIC - Set up proper authentication for multiple tool endpoints
# MAGIC - Log and deploy the tool-calling agent
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Access to configured Unity Catalog functions, vector search indices, and agent endpoints
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq backoff databricks-openai uv databricks-agents mlflow-skinny[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Import the UC Multi-Tool Responses Agent from our source code
import sys
import os
sys.path.append('<set your databricks path to src>')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Import the UC Multi-Tool Responses Agent
# MAGIC
# MAGIC The UC Multi-Tool Responses Agent is implemented in our source code with the following key features:
# MAGIC
# MAGIC ### Agent Architecture
# MAGIC - **ToolCallingAgent**: FMAPI-compliant agent that orchestrates tool calls
# MAGIC - **Multi-Tool Support**: Integrates UC functions, vector search, and agent endpoints
# MAGIC - **Thread-Safe Operations**: Safe concurrent access for streaming responses
# MAGIC - **Automatic Tool Loading**: Configurable tool initialization from CONFIG
# MAGIC
# MAGIC ### Key Components
# MAGIC - **ToolInfo**: Unified interface for different tool types (UC functions, vector search, agent endpoints)
# MAGIC - **AgentEndpointExecutor**: Handles communication with specialized agent endpoints
# MAGIC - **Message System**: Unified message handling for Responses API and Chat Completion formats
# MAGIC - **Progressive Disclosure**: Streams thought process and tool execution to users

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the UC Multi-Tool Responses Agent
# MAGIC
# MAGIC The UC Multi-Tool Responses Agent demonstrates intelligent tool selection by analyzing questions and executing appropriate tools.
# MAGIC It provides progressive disclosure of its thinking process and can combine results from multiple tools.

# COMMAND ----------

from uc_multi_tool_responses_agent.agent import AGENT

# Test with a computational question that might use math tools
print("=== Testing Computational Question ===")
result = AGENT.predict({
    "input": [{"role": "user", "content": "Calculate root of 3x3x3"}]
})
print("Response structure:", result.model_dump(exclude_none=True)['output'][0].keys())
print()

# COMMAND ----------

# Test with a question that might use vector search
print("=== Testing Vector Search Question ===")
result = AGENT.predict({
    "input": [{"role": "user", "content": "Find a blue sweater and don't use any filters and provide a summary"}]
})
print("Response structure:", result.model_dump(exclude_none=True)['output'][0].keys())
print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming Response Test
# MAGIC
# MAGIC Test the FMAPI streaming feature by observing the agent's thought process and tool calls in real-time.

# COMMAND ----------

print("=== Testing FMAPI Streaming with Tool Calls ===")
for chunk in AGENT.predict_stream({
    "input": [{"role": "user", "content": "Calculate 2 + 3 with math and pyhton tool"}]
}):
    chunk_data = chunk.model_dump(exclude_none=True)
    
    # CORRECTED: UC agent uses response.output_item.add_delta
    if chunk.type == "response.output_item.add_delta":
        item = chunk_data.get('item', {})
        if item.get('type') == 'response.output_text.delta':
            delta_text = item.get('delta', '')
            if delta_text:
                print(f"[STREAM] {delta_text}", end='')
                
    elif chunk.type == "response.output_item.done":
        item = chunk_data.get('item', {})
        if item.get('type') == 'output_text':
            content = item.get('content', '')
            if isinstance(content, list) and content:
                text_content = content[0].get('text', '') if isinstance(content[0], dict) else str(content[0])
                print(f"\n[OUTPUT TEXT] {text_content}")
            else:
                print(f"\n[OUTPUT TEXT] {content}")
                
        elif item.get('type') == 'function_call':
            name = item.get('name', 'unknown')
            args = item.get('arguments', '')
            print(f"\n[FUNC CALL] {name}: {args}")
            
        elif item.get('type') == 'function_call_output':
            output = item.get('output', '')
            print(f"\n[FUNC RESULT] {output}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the FMAPI UC Multi-Tool Agent as an MLflow Model
# MAGIC
# MAGIC Log the UC Multi-Tool agent with proper resource dependencies for automatic authentication.
# MAGIC This includes all configured tool endpoints to ensure secure access during deployment.

# COMMAND ----------

import mlflow
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex
)
from pkg_resources import get_distribution
from uc_multi_tool_responses_agent.agent import CONFIG

# Attach resources for automatic authentication passthrough
resources = []

# Add agent LLM endpoint
resources.append(DatabricksServingEndpoint(endpoint_name=CONFIG["llm_endpoint"]))

# Add agent endpoints from config if present
domain_agent_endpoints = [
    agent_info["endpoint"]
    for agent_info in CONFIG.get("tools", {}).get("agent_endpoints", [])
    if agent_info.get("endpoint")
]
resources.extend([DatabricksServingEndpoint(endpoint_name=ep) for ep in domain_agent_endpoints])

# Add vector search indices from config if present
vector_search_indices = [
    vs["index_name"]
    for vs in CONFIG.get("tools", {}).get("vector_search", [])
    if vs.get("index_name")
]
resources.extend([
    DatabricksVectorSearchIndex(index_name=idx) for idx in vector_search_indices
])

# Add UC functions from config if present
uc_functions = [
    fn for fn in CONFIG.get("tools", {}).get("uc_functions", [])
]
resources.extend([
    DatabricksFunction(function_name=fn)
    for fn in uc_functions
])

print("UC Multi-Tool agent resources configured for authentication:")
print(f"- Agent LLM: {CONFIG['llm_endpoint']}")
for endpoint in domain_agent_endpoints:
    print(f"- Agent endpoint: {endpoint}")
if vector_search_indices:
    print(f"- Vector search indices: {vector_search_indices}")
if uc_functions:
    print(f"- UC functions: {uc_functions}")

# COMMAND ----------

from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse, Message

# Define input example for the UC Multi-Tool agent
input_example = ResponsesAgentRequest(input=[Message(role="user", content="calculate the root of 3x3x3")])

# Log the model with proper dependencies using the in-memory agent instance
with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="uc_multi_tool_responses_agent",
        python_model="/Workspace/Users/andreas.jack@databricks.com/multi-agent-architecture/src/uc_multi_tool_responses_agent/agent.py",
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
            "agent_type": "uc_multi_tool_responses",
            "llm_endpoint": CONFIG["llm_endpoint"],
            "max_iterations": 10
        },
        metadata={"task": "agent/v1/responses"}
    )

print(f"UC Multi-Tool agent logged successfully!")
print(f"Run ID: {logged_agent_info.run_id}")
print(f"Model URI: {logged_agent_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the FMAPI UC Multi-Tool Agent
# MAGIC
# MAGIC Use Mosaic AI Agent Evaluation to evaluate the agent's tool selection decisions,
# MAGIC response quality, and tool calling capabilities across different computational queries.

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety

# Create evaluation dataset covering different tool usage scenarios
eval_dataset = [
    {
        "inputs": {"input": [{"role": "user", "content": "What is the root of 3x3x3?"}]},
        "expected_response": "The root of 3x3x3 is 3",
    },
]

print(f"Evaluating UC Multi-Tool agent with {len(eval_dataset)} test cases...")

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=lambda input: AGENT.predict({"input": input}),
    scorers=[
        RelevanceToQuery(),
        Safety()
    ]
)


print("Evaluation completed! Check MLflow UI for detailed results.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment Validation
# MAGIC
# MAGIC Validate the logged UC Multi-Tool agent before deployment to ensure proper functionality
# MAGIC and resource access.

# COMMAND ----------

# Test the logged model
print("=== Validating Logged Model ===")
validation_result = mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/uc_multi_tool_responses_agent",
    input_data=input_example,
    env_manager="uv",
)

print("Validation successful!")
print(f"Response type: {type(validation_result)}")
print(f"Response structure: {list(validation_result.keys()) if isinstance(validation_result, dict) else 'Not a dict'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the FMAPI UC Multi-Tool Agent to Unity Catalog
# MAGIC
# MAGIC Register the UC Multi-Tool agent to Unity Catalog for deployment and governance.
# MAGIC
# MAGIC - **TODO** Update the `catalog`, `schema`, and `model_name` below to register the UC Multi-Tool agent.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "<set your catalog>"
schema = "<set your schema>"  
model_name = "uc_multi_tool_responses_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

print(f"Unity Catalog model name: {UC_MODEL_NAME}")

# Uncomment to register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, 
    name=UC_MODEL_NAME
)
print(f"Model registered to Unity Catalog: {UC_MODEL_NAME}")
print(f"Version: {uc_registered_model_info.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the FMAPI UC Multi-Tool Agent
# MAGIC
# MAGIC Deploy the UC Multi-Tool agent with automatic authentication for all configured tool endpoints.

# COMMAND ----------

from databricks import agents

endpoint_name = "uc_multi_tool_responses_agent"

# Uncomment to deploy the agent (requires UC registration first)
agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    tags={"endpointSource": "uc_multi_tool", "architecture": "fmapi_tool_calling", "RemoveAfter": "2025-12-31", "endpointSource": "docs"},
    endpoint_name=endpoint_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Deployment (Optional)
# MAGIC
# MAGIC Once deployed, you can test the UC Multi-Tool agent endpoint directly.

# COMMAND ----------

# Example deployment testing (uncomment after deployment)
from databricks.sdk import WorkspaceClient 

w = WorkspaceClient()

# Test the deployed endpoint
response = w.serving_endpoints.query(
     name="uc_multi_tool_responses_agent",
     inputs={"input": [{"role": "user", "content": "Test the deployed FMAPI UC Multi-Tool agent with tool calling"}]}
 )
 
print("Deployment test successful!")
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC After your UC Multi-Tool agent is deployed, you can:
# MAGIC
# MAGIC 1. **Chat in AI Playground**: Test the progressive disclosure and tool calling capabilities
# MAGIC 2. **Share with Users**: Get feedback on tool selection and response quality  
# MAGIC 3. **Monitor Performance**: Track tool usage patterns and response accuracy
# MAGIC 4. **Extend Tools**: Add new tools by updating the CONFIG in `uc_multi_tool_responses_agent/agent.py`
# MAGIC 5. **Production Integration**: Embed in applications requiring multi-tool interactions
# MAGIC
# MAGIC ### Key Features to Explore:
# MAGIC - **Intelligent Tool Selection**: Ask questions that require different tool combinations
# MAGIC - **Tool Calling**: Observe the agent's tool selection and execution process
# MAGIC - **Response Synthesis**: Test complex questions requiring multiple tool interactions
# MAGIC - **Streaming Experience**: Experience real-time thought process and tool calls
# MAGIC
# MAGIC The FMAPI UC Multi-Tool agent provides a scalable pattern for orchestrating multiple tools
# MAGIC while maintaining transparency and user engagement through tool calling and streaming.