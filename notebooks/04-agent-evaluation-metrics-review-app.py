# Databricks notebook source
# MAGIC %md 
# MAGIC # Mosaic AI Agent Evaluation: Custom metrics, guidelines and domain expert labels
# MAGIC
# MAGIC This notebook demonstrates how to evaluate a GenAI app using Agent Evaluation's proprietary LLM judges, custom metrics, and labels from domain experts. It demonstrates:
# MAGIC - Loading production logs (traces) into an evaluation dataset
# MAGIC - Running evaluation and doing root cause analysis
# MAGIC - Writing custom metrics to automatically detect quality issues
# MAGIC - Sending production logs for SMEs to label and evolve the evaluation dataset
# MAGIC
# MAGIC To learn more about Mosaic AI Agent Evaluation, see Databricks documentation ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/)).
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC - See the requirements of Agent Evaluation ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/evaluate-agent#requirements) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/evaluate-agent#requirements))
# MAGIC - Serverless or classic cluster running Databricks Runtime 15.4 LTS or above, or Databricks Runtime for Machine Learning 15.4 LTS or above.
# MAGIC - CREATE TABLE access in a Unity Catalog Schema
# MAGIC
# MAGIC <img src="http://docs.databricks.com/images/generative-ai/agent-evaluation/review-app/overview.png"/>

# COMMAND ----------

# MAGIC %pip install -U -qqqq 'mlflow>=2.20.3' 'langchain==0.3.20' 'langgraph==0.3.4' 'databricks-langchain>=0.3.0' pydantic 'databricks-agents>=0.17.2' uv databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select a Unity Catalog schema
# MAGIC
# MAGIC Ensure you have CREATE TABLE access in this schema.  By default, these values are set to your workspace's default catalog & schema.

# COMMAND ----------

UC_CATALOG = 'hong_zhu_demo_catalog'
UC_SCHEMA = 'basf_genie_agent'
UC_PREFIX = f"{UC_CATALOG}.{UC_SCHEMA}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select (pre)production logs
# MAGIC
# MAGIC Since this is a demo notebook, we generate example production logs below to demonstrate the new features in Agent Evaluation. We call our agent directly and log traces in MLFlow. 
# MAGIC
# MAGIC **_NOTE:_** MLFlow tracing will visualize each trace (with pagination) in the cell output when you call your agent or retrieve traces using [`mlflow.search_traces`](https://mlflow.org/docs/latest/tracing/api/search).
# MAGIC
# MAGIC After you've completed the notebook, and you already have an agent deployed on Databricks, locate the `request_ids` to be reviewed from the `<model_name>_payload_request_logs `inference table. The inference table is in the same Unity Catalog catalog and schema where the model was registered. Sample code for this is near the bottom of this notebook.

# COMMAND ----------

import mlflow
from agent import AGENT

# Include example questions
examples = [
    "Explain the datasets and capabilities that the Genie agent has access to.",
    "Run similarity search about Volkswagen.",
    "What are some of the innovations featured in the new Passat model?",
    "What were Volkswagen sales and operating profit in the third quarter of 2023?",
    "How many units have been produced each month at customer plants?"
]

# Below, we are calling the agent and logging the traces in an MLFlow run. These traces will become our evaluation dataset.
with mlflow.start_run(run_name="hongzhu-genie-multi-agent-logs") as run:
    for example in examples:
        AGENT.predict({"messages": [{"role": "user", "content": example}]})

requests = mlflow.search_traces(run_id=run.info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the traces into an evaluation dataset
# MAGIC
# MAGIC **Important**: Before running this cell, ensure the values of `uc_catalog` and `uc_schema` widgets are set to a Unity Catalog schema where you have CREATE TABLE permissions. Re-running this cell will re-create the evaluation dataset. 

# COMMAND ----------

from databricks.agents import datasets
from databricks.sdk.errors.platform import NotFound

# Make sure you have updated the uc_catalog & uc_schema widgets to a valid catalog/schema where you have CREATE TABLE permissions.
UC_TABLE_NAME = f'{UC_PREFIX}.agent_evaluation_set'

# Remove the evaluation dataset if it already exists
try:
  datasets.delete_dataset(UC_TABLE_NAME)
except NotFound:
  pass

# Create the evaluation dataset
dataset = datasets.create_dataset(UC_TABLE_NAME)

# Add the traces from the production logs we gathered in the above cell.
dataset.insert(requests)

# Show the resulting evaluation set
display(spark.table(UC_TABLE_NAME))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run an evaluation
# MAGIC
# MAGIC ##### Agent Evaluation's built-in judges
# MAGIC - Judges that run without ground-truth labels or retrieval in traces:
# MAGIC   - `guidelines`: guidelines allows developers write plain-language checklists or rubrics in their evaluation, improving transparency and trust with business stakeholders through easy-to-understand, structured grading rubrics. 
# MAGIC   - `safety`: making sure the response is safe
# MAGIC   - `relevance_to_query`: making sure the response is relevant
# MAGIC - For traces with retrieved docs (spans of type `RETRIEVER`):
# MAGIC    - `groundedness`: detect hallucinations
# MAGIC    - `chunk_relevance`: chunk-level relevance to the query
# MAGIC - Later, when we collect ground-truth labels using the Review app, we will benefit from two more judges:
# MAGIC    - `correctness`: will be ignored until we collect labels like `expected_facts`
# MAGIC    - `context_sufficiency`: will be ignored until we collect labels like `expected_facts`
# MAGIC
# MAGIC See the full list of built-in judges ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-reference) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/llm-judge-reference)) and how to run a subset of judges or customize judges ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/advanced-agent-eval) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/advanced-agent-eval)).
# MAGIC
# MAGIC ##### Custom metrics
# MAGIC - Check the quality of tool calling
# MAGIC    - `tool_calls_are_logical`: assert that the selected tools in the trace were logical given the user's request. 
# MAGIC    - `grounded_in_tool_outputs`: assert that the LLM's responses are grounded in the outputs of the tools and not hallucinating
# MAGIC - Measure the agent's cost & latency
# MAGIC    - `latency`: extracts the latency from the MLflow trace
# MAGIC    - `cost`: extracts the total tokens used and multiplies by the LLM token rate
# MAGIC
# MAGIC This notebook creates custom metrics ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/custom-metrics) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/custom-metrics)) that use Mosaic AI callable judges. Custom metrics can be any Python function. More examples: ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-reference#examples-6) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/llm-judge-reference#examples-6)).
# MAGIC
# MAGIC <img src="http://docs.databricks.com/images/generative-ai/agent-evaluation/review-app/eval_1.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the custom metrics

# COMMAND ----------

from databricks.agents.evals import judges
from mlflow.evaluation import Assessment
from databricks.agents.evals import metric
from mlflow.entities import SpanType


@metric
def tool_calls_are_logical(request, tool_calls):
    # If no tool calls, don't run this metric
    if len(tool_calls) == 0:
        return None

    # We assume that the tools available to the FIRST llm call is the same as what is presented to all other LLM calls.  Adjust if this doesn't hold true for your use case.
    available_tools = tool_calls[0].available_tools

    # Get ALL called tools across ALL LLM calls - this will happen if the LLM does multiple iterations to call tools (e.g., calls a set of tools & then decides to call more tools based on that output)
    requested_tools = []
    for item in tool_calls:
        requested_tools.append(
            {"tool_name": item.tool_name, "tool_call_args": item.tool_call_args}
        )

    is_logical = judges.guideline_adherence(
        request=f"User's request: {request}\nAvailable tools: {available_tools}",
        response=str(requested_tools),
        guidelines=[
            "The response is a set of selected tool calls. The selected tools must be logical, given the user's request."
        ],
    )
    # See https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-reference#examples-6 or 
    # https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/llm-judge-reference#examples-6
    return Assessment(
        name="tool_calls_are_logical",
        value=is_logical.value,
        rationale=is_logical.rationale,
    )


@metric
def grounded_in_tool_outputs(request, response, tool_calls):
    # If no tool calls, don't run this metric
    if len(tool_calls) == 0:
        return None

    # Customize the built-in groundness judge for the tool calling outputs
    tool_outputs = [{'result': t.tool_call_result["content"], 'args': t.tool_call_args, 'name': t.tool_name} for t in tool_calls]
    contexts = []

    # Format the tool calls as "Called tool tool_name(param1=value, param2=value) that returned ```return value```"".
    for tool in tool_outputs:
        args_str = ', '.join(f"{k}={v}" for k, v in tool['args'].items())
        contexts.append(f"Called tool `{tool['name']}({args_str})` that returned ```{tool['result']}```")
    
    
    context_to_evaluate = "\n".join(contexts)

    # Extract the user's request & LLM's response
    user_request = next(item for item in request['messages'] if item['role'] == 'user')['content']
    assistant_response = response['messages'][-1]["content"]

    # Create a guidelines judge to evaluate if the assistant's response is grounded in the context of the tool calls.
    out = judges.guideline_adherence(
        request=f"<user_request>{user_request}<user_request><context_to_evaluate>{context_to_evaluate}<context_to_evaluate>",
        response=f"<assistant_response>{assistant_response}<assistant_response>",
        guidelines=["The <assistant_response>'s to the <user_request> must be grounded in the <context_to_evaluate> which represent tools that were called when trying to answer the <user_request>."]
    )

    return Assessment(
        name="grounded_in_tool_outputs", value=out.value, rationale=out.rationale
    )


@metric
def is_answer_relevant(request, response):
    # Extract the user's request & LLM's response
    user_request = next(item for item in request['messages'] if item['role'] == 'user')['content']
    assistant_response = response['messages'][-1]["content"]

    # Use the guideline's judge to assess the relevance of the LLM's response.  We take this approach (rather than the built-in answer_relevance judge) to account for the fact that the LLM may (correctly) refuse to answer a question that violates our policies.
    out = judges.guideline_adherence(
        request=request,
        response=assistant_response,
        guidelines=["Determine if the response provides an answer to the user's request.  A refusal to answer is considered relevant.  However, if the response is NOT a refusal BUT also doesn't provide relevant information, then the answer is not relevant."]
    )
    return Assessment(
        name="is_answer_relevant", value=out.value, rationale=out.rationale
    )

@metric
def latency(trace):
    return trace.info.execution_time_ms / 1000

@metric 
def cost(trace):
    INPUT_TOKEN_COST = 2 # per 1M tokens
    OUTPUT_TOKEN_COST = 15 # per 1M tokens
    input_tokens = trace.search_spans(span_type=SpanType.CHAT_MODEL)[0].outputs['llm_output']['prompt_tokens']
    output_tokens = trace.search_spans(span_type=SpanType.CHAT_MODEL)[0].outputs['llm_output']['completion_tokens']
    cost = ((input_tokens/1000000) * INPUT_TOKEN_COST) + ((output_tokens/1000000) * OUTPUT_TOKEN_COST)
    return round(cost, 3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the evaluation

# COMMAND ----------

# Define our global guidelines.  Guidelines are plain langauge 
guidelines = {'pricing': ["The agent should always refuse to answer questions about product pricing; it should never provide anything more than 'I can't talk about pricing'."]}

with mlflow.start_run(run_name="hongzhu-genie-agent-eval-prod-logs"):
    eval_results = mlflow.evaluate(
        # Each row["inputs"] from the dataset is passed to the model. We suppory any dict[str, Any] as inputs.
        model=lambda inputs: AGENT.predict(inputs),
        data=spark.table(UC_TABLE_NAME),
        model_type="databricks-agent",
        # Enable our custom metrics
        extra_metrics=[grounded_in_tool_outputs, tool_calls_are_logical, is_answer_relevant, latency, cost],

        # Configure which built-in judges are used and customize the guidelines used
        evaluator_config={
            "databricks-agent": {"global_guidelines": guidelines, "metrics": [
                "chunk_relevance", # Check if the retrieved documents are relevant to the user's query
                "guideline_adherence", # Run the global guidelines defined in `guidelines`
                # Disable the built-in groundness & relevance judge in favor of our custom defined version of these metrics
                # "groundedness", 
                # "relevance_to_query",
                "safety", # Check if the LLM's response has any toxicity
                # context_sufficiency & correctness requires labeled ground truth, which we will collect later in this notebook, so disable them for now.
                # "context_sufficiency", 
                # "correctness",
            ],},
            
        },
    )
    # Review the evaluation results in the MLFLow UI (see console output), or access them in place:
    display(eval_results.tables["eval_results"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fix issues and re-evaluate
# MAGIC
# MAGIC Now that we have an evaluation set with judges we can try, let's attempt to fix the issues by:
# MAGIC
# MAGIC - Improving our system prompt to let the agent know it's ok if no tools are being called
# MAGIC - Adding a doc to our knowledge base about latest spark version
# MAGIC - Add a new addition tool
# MAGIC
# MAGIC <img src="http://docs.databricks.com/images/generative-ai/agent-evaluation/review-app/eval_2.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Collect *expectations* (ground-truth labels)
# MAGIC
# MAGIC Now that we have improved our agent, we want to make sure that certain responses always get the facts right.
# MAGIC
# MAGIC Using the review app ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/review-app) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/review-app)), we will send our evals to a labeling session for our SMEs to provide:
# MAGIC - `expected_facts` so we can benefit from the `correctness` ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-reference#correctness) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/llm-judge-reference#correctness)) and `context_sufficiency` ([AWS](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/llm-judge-reference#context-sufficiency) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-evaluation/llm-judge-reference#context-sufficiency)) judges.
# MAGIC - `guidelines` so our SMEs can add additional plain language criteria for each question based on their business context.  This will extend the guidelines we already have defined at a global level.
# MAGIC - If they liked the response, so our stakeholders can have confidence that the new model is indeed better. We do this using a [custom label schema](https://api-docs.databricks.com/python/databricks-agents/latest/databricks_agent_eval.html#databricks.agents.review_app.ReviewApp.create_label_schema).
# MAGIC
# MAGIC
# MAGIC <img src="http://docs.databricks.com/images/generative-ai/agent-evaluation/review-app/review_app_1.png"/>

# COMMAND ----------

from databricks.agents import review_app

# OPTIONAL: Update the assigned_users widget with a comma separated list of users to assign the review app to.
# If not provided, only the user running this notebook will be granted access to the review app.
ASSIGNED_USERS = []

# Set the MLflow experiment used for agent deployment
mlflow.set_experiment("/Users/hong.zhu@databricks.com/Genie in multi-agent systems/03-langgraph-multiagent-genie-pat")
my_review_app = review_app.get_review_app()

my_review_app.add_agent(
    agent_name="Genie-multi-agent",
    model_serving_endpoint="<put your workspace URL here>/serving-endpoints/genie_multi_agent_basf/invocations",
)

my_review_app.create_label_schema(
  name="good_response",
  # Type can be "expectation" or "feedback".
  type="feedback",
  title="Is this a good response?",
  input=review_app.label_schemas.InputCategorical(options=["Yes", "No"]),
  instruction="Optional: provide a rationale below.",
  enable_comment=True,
  overwrite=True
)

# CHANGE TO YOUR PAYLOAD REQUEST LOGS TABLE
PAYLOAD_REQUEST_LOGS_TABLE = "hong_zhu_demo_catalog.basf_genie_agent.multi_agent_basf_payload_request_logs"
traces = spark.table(PAYLOAD_REQUEST_LOGS_TABLE).select("trace").limit(3).toPandas()

my_session = my_review_app.create_labeling_session(
  name="payload_request_logs",
  assigned_users=ASSIGNED_USERS,
  label_schemas=[review_app.label_schemas.GUIDELINES,review_app.label_schemas.EXPECTED_FACTS,  "good_response"]
)

my_session.add_traces(traces)

# Share with the SME.
print("Review App URL:", my_review_app.url)
print("Labeling session URL: ", my_session.url)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-evaluation with the collected `expected_facts`
# MAGIC
# MAGIC After the SMEs are done with the labeling, we will sync the labels into our evaluation dataset and re-evaluate. Note that the `correctness` judge should run for any eval row with `expected_facts`.

# COMMAND ----------

# Let's see the progress of our labeling sesssion, by selecting traces associated with the labeling session run.
def is_response_good(assessments):
    for assessment in assessments:
        if assessment.name == "good_response":
            return assessment.feedback.value == "Yes"
    return None

# View how many labels the SME provided.
traces = mlflow.search_traces(run_id=my_session.mlflow_run_id)
response_values = traces["assessments"].apply(is_response_good).value_counts(dropna=False)
print(
    f"Got {response_values.get(True, 0)} good responses, "
    f"{response_values.get(False, 0)} bad responses, and "
    f"{response_values.get(None, 0)} not yet labeled.")

# Move the SME's labels to the evaluation dataset that we created earlier.
my_session.sync_expectations(to_dataset=UC_TABLE_NAME)

with mlflow.start_run(run_name="with-human-labels") as run:
    eval_results = mlflow.evaluate(
        # Each row["inputs"] from the dataset is passed to the model. We suppory any dict[str, Any] as inputs.
        model=lambda inputs: AGENT.predict(inputs),
        data=spark.table(UC_TABLE_NAME),
        model_type="databricks-agent",
        # Enable our custom metrics
        extra_metrics=[grounded_in_tool_outputs, tool_calls_are_logical, is_answer_relevant, latency, cost],

        # Configure which built-in judges are used and customize the guidelines used
        evaluator_config={
            "databricks-agent": {"global_guidelines": guidelines, "metrics": [
                "chunk_relevance", # Check if the retrieved documents are relevant to the user's query
                "guideline_adherence", # Run the global guidelines defined in `guidelines`
                # "groundedness", # Disable the built-in groundness in favor of our custom defined version of groundedness
                # "relevance_to_query", # Check if the LLM's response is relevant to the user's query
                "safety", # Check if the LLM's response has any toxicity
                # We can now enable context_sufficiency & correctness since we have collected labeled ground truth.
                "context_sufficiency", 
                "correctness",
            ],},
            
        },
    )
    display(eval_results.tables["eval_results"])