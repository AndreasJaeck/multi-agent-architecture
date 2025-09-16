# Databricks notebook source
# MAGIC %md
# MAGIC # UC Toolkit: Tool Functions
# MAGIC
# MAGIC ## Introduction
# MAGIC
# MAGIC This notebook demonstrates how to create and use tool functions that can be called by a Large Language Model (LLM) to extend its capabilities. Tool functions allow the LLM to:
# MAGIC  
# MAGIC - Perform vector similarity searches
# MAGIC - Access structured data in your data lakehouse
# MAGIC - Execute calculations and unit conversions
# MAGIC - Run Python code for specialized tasks
# MAGIC - Call other LLMs for specific subtasks
# MAGIC  
# MAGIC  These functions can be registered in your Databricks environment and made available through a Function Calling API, allowing the LLM to use them when needed to answer user questions accurately.
# MAGIC
# MAGIC  **Please use Serverless Cluster to create Functions!**
# MAGIC

# COMMAND ----------

# MAGIC %pip install -q databricks-sdk==0.41.0 langchain-community==0.2.10 langchain-openai==0.1.19 mlflow==2.20.2 faker==37.1.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Use a Databricks managed Workspace
# MAGIC

# COMMAND ----------

# Specify catalog and schema
catalog = "your_catalog"
schema = "your_schema"

# COMMAND ----------

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Store Search Functions
# MAGIC  
# MAGIC Vector search enables `semantic similarity searches` that go beyond exact keyword matching. By converting text to vector embeddings, we can find items that are conceptually similar even if they don't share the same exact words.
# MAGIC  
# MAGIC Databricks Vector Search simplifies this process with the `VECTOR_SEARCH()` SQL function. This function allows you to query a Mosaic AI Vector Search index using SQL. 
# MAGIC
# MAGIC [VECTOR_SEARCH](https://learn.microsoft.com/en-gb/azure/databricks/sql/language-manual/functions/vector_search)
# MAGIC  
# MAGIC And can be used to:
# MAGIC  - Perform semantic search over content descriptions
# MAGIC  - Find relevant documents based on meaning, not just keywords
# MAGIC  - Retrieve the most similar items with relevancy scores

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Find semantically similar content
# MAGIC This function searches in the valona table for semantic similar content.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to find similar news using vector search
# MAGIC CREATE OR REPLACE FUNCTION find_news(
# MAGIC   description STRING COMMENT 'Descriptive text to search for news ticker content about automotive and coatings industry. Can include source, status, header, url, modification date'
# MAGIC )
# MAGIC RETURNS TABLE (results_itemId bigint, title string, created_date timestamp, modified_date timestamp,results_status string, results_source string, item_link string, search_text string, CAT_NAME string, CAT_ID string)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Find content with similarity search based on description of content. This helps users to find latest news about about automotive and coatings industry. Returns results_itemId, title, created_date, modified_date, results_status, results_source, item_link, search_text, CAT_NAME, CAT_ID'
# MAGIC RETURN
# MAGIC   SELECT results_itemId, title, created_date, modified_date,results_status, results_source, item_link, search_text, CAT_NAME, CAT_ID
# MAGIC   FROM VECTOR_SEARCH(
# MAGIC     index => 'your_catalog.your_schema.your_index',
# MAGIC     query => description,
# MAGIC     num_results => 5
# MAGIC   )
# MAGIC   -- WHERE results_source = 'Reuters' -- Example filter
# MAGIC   ORDER BY search_score DESC

# COMMAND ----------

# MAGIC %md
# MAGIC - Determines the maximum number of documents returned in a query, with a max. value of 100. [num_results](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/vector_search)
# MAGIC - Field truncation based on No. of rows or bytes retrieved [SQL client truncation behavioud](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools#uc-function-retriever)

# COMMAND ----------

# %sql
# -- Function to find similar news using vector search
# CREATE OR REPLACE FUNCTION find_news(
#   description STRING COMMENT 'Descriptive text to search for news ticker content about automotive and coatings industry. Can include source, status, header, url, modification date'
# )
# RETURNS TABLE (results_itemId bigint, title string, created_date timestamp, modified_date timestamp,results_status string, results_source string, item_link string, search_text string)
# LANGUAGE SQL
# COMMENT 'Find content with similarity search based on description of content. This helps users to find latest news about about automotive and coatings industry. Returns results_itemId, title, created_date, modified_date, results_status, results_source, item_link, search_text, CAT_NAME, CAT_ID'
# RETURN
#   SELECT results_itemId, title, created_date, modified_date,results_status, results_source, item_link, substring(search_text, 0, 8192)
#   FROM VECTOR_SEARCH(
#     index => 'your_catalog.your_schema.your_index',
#     query => description,
#     num_results => 100
#   )
#   ORDER BY search_score DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's test our function:
# MAGIC SELECT * FROM find_news('What are the latest news about volkswagen?');

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM find_news('What are some of the innovations featured in the new Passat model?');

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM find_news('What were Volkswagen sales and operating profit in the third quarter of 2023?');

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL Functions
# MAGIC
# MAGIC SQL functions provide direct access to structured data within your lakehouse. These functions can:
# MAGIC - Retrieve detailed information about specific products or processes
# MAGIC - Run pre-defined analytical queries
# MAGIC - Return structured data that the LLM can reference in its responses
# MAGIC
# MAGIC The following SQL functions demonstrate how to create structured data access points that an LLM can use to get reliable, up-to-date information.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Information Retrieval
# MAGIC
# MAGIC This function retrieves comprehensive information about a product based on its ID, providing all available details from the products table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to get detailed product information
# MAGIC CREATE OR REPLACE FUNCTION get_product(
# MAGIC   productid STRING COMMENT 'Unique identifier for the product following pattern ^P[0-9]{4}$, e.g., P0001, P0002, etc.'
# MAGIC )
# MAGIC RETURNS TABLE (product_id string, product_name string, category string,
# MAGIC chemical_formula string, molecular_weight double, density double, melting_point double, boiling_point double, description string, application_areas string, storage_conditions string, full_description string, creation_date string, price_per_unit double
# MAGIC )
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Retrieve detailed information about a specific product using its product ID. Returns comprehensive product details including physical and chemical properties, usage recommendations, storage requirements, and pricing information.'
# MAGIC RETURN
# MAGIC   SELECT *
# MAGIC   FROM your_catalog.your_schema.your_table
# MAGIC   WHERE product_id = productid;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_product('P0001')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Safety Protocol Retrieval
# MAGIC
# MAGIC This function retrieves safety protocols, procedures, and research notes for a specific product ID. It's useful for:
# MAGIC - Ensuring proper handling of chemical products
# MAGIC - Accessing the latest safety guidelines
# MAGIC - Reviewing research notes for a particular product

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to get detailed safety protocol information
# MAGIC CREATE OR REPLACE FUNCTION get_safety_protocols(
# MAGIC   productid STRING COMMENT 'Unique identifier for the product following pattern ^P[0-9]{4}$, e.g., P0001, P0002, etc.'
# MAGIC )
# MAGIC RETURNS TABLE (description_id STRING, description_type STRING, product_id STRING, title STRING, content STRING)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Retrieve all safety protocols, handling procedures, and research notes associated with a specific product ID. Returns detailed safety information including protocols, procedures, and documentation needed for proper handling and use of the chemical product.'
# MAGIC RETURN
# MAGIC   SELECT description_id, description_type, product_id, title, content
# MAGIC   FROM your_catalog.your_schema.your_table
# MAGIC   WHERE product_id = productid;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_safety_protocols('P0001')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reaction Details Retrieval
# MAGIC
# MAGIC This function provides detailed information about chemical reactions used to produce a specific product. It's useful for:
# MAGIC - Understanding production processes
# MAGIC - Evaluating reaction conditions
# MAGIC - Assessing safety requirements and hazards

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to get detailed reaction information
# MAGIC CREATE OR REPLACE FUNCTION get_reaction_details(
# MAGIC   productid STRING COMMENT 'Unique identifier for the product following pattern ^P[0-9]{4}$, e.g., P0001, P0002, etc.'
# MAGIC )
# MAGIC RETURNS TABLE (reaction_id STRING, reaction_name STRING, reaction_type STRING, catalyst STRING, solvent STRING, temperature DOUBLE, pressure DOUBLE, reaction_time DOUBLE, energy_consumption DOUBLE, hazards STRING)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Retrieve detailed information about the chemical reactions used in manufacturing a specific product. Returns comprehensive reaction parameters including reaction conditions, catalysts, solvents, environmental requirements, energy usage, and associated hazards.'
# MAGIC RETURN
# MAGIC   SELECT reaction_id, reaction_name, reaction_type, catalyst, solvent, temperature, pressure, reaction_time, energy_consumption, hazards
# MAGIC   FROM your_catalog.your_schema.your_table
# MAGIC   WHERE product_id = productid;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM get_reaction_details('P0001')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Quality Analysis
# MAGIC
# MAGIC This function analyzes quality metrics for a specific product, providing insights into testing results and pass rates. It's useful for:
# MAGIC - Evaluating product reliability
# MAGIC - Identifying quality issues
# MAGIC - Making data-driven decisions about product improvements

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Function to analyze product quality metrics
# MAGIC CREATE OR REPLACE FUNCTION analyze_product_quality(
# MAGIC   productid STRING COMMENT 'Unique identifier for the product following pattern ^P[0-9]{4}$, e.g., P0001, P0002, etc.'
# MAGIC )
# MAGIC RETURNS TABLE (product_name STRING, total_tests INT, passed_tests INT, failed_tests INT, pass_rate DOUBLE)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Analyze quality control metrics for a specific product based on historical test data. Returns aggregated quality statistics including test frequency, pass/fail rates, and overall quality score to help assess product reliability and consistency.'
# MAGIC RETURN
# MAGIC   SELECT 
# MAGIC     p.product_name,
# MAGIC     COUNT(q.test_id) as total_tests,
# MAGIC     SUM(CASE WHEN q.test_result = 'Pass' THEN 1 ELSE 0 END) as passed_tests,
# MAGIC     SUM(CASE WHEN q.test_result = 'Fail' THEN 1 ELSE 0 END) as failed_tests,
# MAGIC     ROUND(SUM(CASE WHEN q.test_result = 'Pass' THEN 1 ELSE 0 END) * 100.0 / COUNT(q.test_id), 2) as pass_rate
# MAGIC   FROM your_catalog.your_schema.your_table q
# MAGIC   JOIN your_catalog.your_schema.your_table b ON q.batch_id = b.batch_id
# MAGIC   JOIN your_catalog.your_schema.your_table p ON b.product_id = p.product_id
# MAGIC   WHERE b.product_id = productid
# MAGIC   GROUP BY p.product_name;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM analyze_product_quality('P0002')

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## Python Functions
# MAGIC
# MAGIC Python functions extend the capabilities of your LLM by allowing it to perform more complex operations that aren't easily expressed in SQL. These functions can:
# MAGIC - Perform specialized calculations and unit conversions
# MAGIC - Execute custom algorithms
# MAGIC - Access external APIs and services
# MAGIC - Run arbitrary Python code for advanced use cases
# MAGIC
# MAGIC The functions below demonstrate different ways Python can be integrated as tools for your LLM.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Unit Conversion Tool
# MAGIC
# MAGIC This function provides chemical unit conversions between common measurement units. It's useful for:
# MAGIC - Converting between different units of measurement (g, kg, mol, L, mL)
# MAGIC - Ensuring consistent units across calculations
# MAGIC - Simplifying unit conversion tasks for users
# MAGIC
# MAGIC Databricks runs Python functions in a safe container. The function below has been designed to prevent prompt injection issues by restricting its functionality to specific unit conversions.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION convert_chemical_unit(
# MAGIC   value DOUBLE COMMENT 'The numeric value to convert from one unit to another',
# MAGIC   from_unit STRING COMMENT 'The source unit of measurement (g, kg, mol, L, mL)',
# MAGIC   to_unit STRING COMMENT 'The target unit of measurement to convert to (g, kg, mol, L, mL)',
# MAGIC   mol_weight DOUBLE COMMENT 'Molecular weight of the substance in g/mol, required for conversions between mass and moles. Use 0 if not applicable.'
# MAGIC )
# MAGIC RETURNS DOUBLE
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Convert between different chemical measurement units including mass (g, kg), volume (L, mL), and molar (mol) units. Molecular weight is required when converting between mass and molar units. Returns the converted value in the specified target unit.'
# MAGIC AS
# MAGIC $$
# MAGIC   unit_conversions = {
# MAGIC     'g_to_kg': lambda x: x / 1000,
# MAGIC     'kg_to_g': lambda x: x * 1000,
# MAGIC     'L_to_mL': lambda x: x * 1000,
# MAGIC     'mL_to_L': lambda x: x / 1000,
# MAGIC     'g_to_mol': lambda x, mw: x / mw if mw else None,
# MAGIC     'mol_to_g': lambda x, mw: x * mw if mw else None
# MAGIC   }
# MAGIC   
# MAGIC   conversion_key = f"{from_unit.lower()}_to_{to_unit.lower()}"
# MAGIC   
# MAGIC   if conversion_key in unit_conversions:
# MAGIC     if conversion_key in ['g_to_mol', 'mol_to_g']:
# MAGIC       if mol_weight is None:
# MAGIC         return f"Molecular weight required for {conversion_key} conversion"
# MAGIC       return unit_conversions[conversion_key](value, mol_weight)
# MAGIC     else:
# MAGIC       return unit_conversions[conversion_key](value)
# MAGIC   else:
# MAGIC     return f"Conversion from {from_unit} to {to_unit} not supported"
# MAGIC $$;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT convert_chemical_unit(1, "g", "mol", 0.58);

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculator Tool
# MAGIC
# MAGIC This function allows an LLM to perform mathematical calculations with high precision. It's useful for:
# MAGIC - Solving complex mathematical expressions
# MAGIC - Performing scientific calculations using the math library
# MAGIC - Ensuring accuracy in numerical responses
# MAGIC
# MAGIC The function has been designed with security in mind, restricting operations to mathematical functions and preventing execution of arbitrary code.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION compute_math(
# MAGIC   expr STRING COMMENT 'A mathematical expression as a string to be evaluated. Supports basic operations (+, -, *, /, **, %) and math module functions (e.g., math.sqrt(13), math.sin(0.5), math.log(10)). Example: "2 + 2" or "math.sqrt(16) + math.pow(2, 3)"'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Run any mathematical function and returns the result as output. Supports python syntax like math.sqrt(13)'
# MAGIC AS
# MAGIC $$
# MAGIC   import ast
# MAGIC   import operator
# MAGIC   import math
# MAGIC   operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.UAdd: operator.pos, ast.USub: operator.neg}
# MAGIC     
# MAGIC   # Supported functions from the math module
# MAGIC   functions = {name: getattr(math, name) for name in dir(math) if callable(getattr(math, name))}
# MAGIC
# MAGIC   def eval_node(node):
# MAGIC     if isinstance(node, ast.Num):  # <number>
# MAGIC       return node.n
# MAGIC     elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
# MAGIC       return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
# MAGIC     elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
# MAGIC       return operators[type(node.op)](eval_node(node.operand))
# MAGIC     elif isinstance(node, ast.Call):  # <func>(<args>)
# MAGIC       func = node.func.id
# MAGIC       if func in functions:
# MAGIC         args = [eval_node(arg) for arg in node.args]
# MAGIC         return functions[func](*args)
# MAGIC       else:
# MAGIC         raise TypeError(f"Unsupported function: {func}")
# MAGIC     else:
# MAGIC       raise TypeError(f"Unsupported type: {type(node)}")  
# MAGIC   try:
# MAGIC     if expr.startswith('```') and expr.endswith('```'):
# MAGIC       expr = expr[3:-3].strip()      
# MAGIC     node = ast.parse(expr, mode='eval').body
# MAGIC     return eval_node(node)
# MAGIC   except Exception as ex:
# MAGIC     return str(ex)
# MAGIC $$;
# MAGIC
# MAGIC -- let's test our function:
# MAGIC SELECT compute_math("(2+2)/3") as result;

# COMMAND ----------

# MAGIC %md
# MAGIC ### External API Integration - Weather Service
# MAGIC
# MAGIC This function demonstrates how to access external APIs from within your tool functions. It retrieves rrent weather data based on latitude and longitude coordinates. This is useful for:
# MAGIC - Incorporating real-time external data into responses
# MAGIC - Enhancing responses with contextual information
# MAGIC - Demonstrating API integration capabilities
# MAGIC
# MAGIC **Note:** This function requires `serverless network egress access` when running on serverless compute. Ensure your networking configuration allows this at the admin account level.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION get_weather(
# MAGIC   latitude DOUBLE COMMENT 'Geographic latitude coordinate in decimal degrees (between -90 and 90)',
# MAGIC   longitude DOUBLE COMMENT 'Geographic longitude coordinate in decimal degrees (between -180 and 180)'
# MAGIC )
# MAGIC RETURNS STRUCT<temperature_in_celsius DOUBLE, rain_in_mm DOUBLE>
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Retrieve current weather information for a specific geographic location using latitude and longitude coordinates. Returns temperature in Celsius and precipitation amount in millimeters from the Open-Meteo API or fallback data if the API is unavailable.'
# MAGIC AS
# MAGIC $$
# MAGIC   try:
# MAGIC     import requests as r
# MAGIC     #Note: this is provided for education only, non commercial - please get a license for real usage: https://api.open-meteo.com. Let s comment it to avoid issues for now
# MAGIC     #weather = r.get(f'https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,rain&forecast_days=1').json()
# MAGIC     return {
# MAGIC       "temperature_in_celsius": weather["current"]["temperature_2m"],
# MAGIC       "rain_in_mm": weather["current"]["rain"]
# MAGIC     }
# MAGIC   except:
# MAGIC     return {"temperature_in_celsius": 22.0, "rain_in_mm": 0.0}
# MAGIC $$;
# MAGIC
# MAGIC -- let's test our function:
# MAGIC SELECT get_weather(52.52, 13.41) as weather;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Python Code Execution
# MAGIC
# MAGIC This function allows an LLM to execute arbitrary Python code and return the results. This is useful for:
# MAGIC - Testing and debugging Python code
# MAGIC - Performing complex data processing tasks
# MAGIC - Creating dynamic responses based on code execution
# MAGIC
# MAGIC **Warning:** This function can execute any Python code, which presents security risks. Only use this in controlled environments with proper security measures in place.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION execute_python_code(
# MAGIC   python_code STRING COMMENT 'Valid Python code as a string to be executed. The code should end with a return statement to provide output. Code can include function definitions, calculations, and standard library imports.'
# MAGIC )
# MAGIC RETURNS STRING
# MAGIC LANGUAGE PYTHON
# MAGIC COMMENT 'Execute arbitrary Python code and return the result as a string. Suitable for custom calculations, data transformations, and other Python-based operations. The code must include a return statement at the end to produce output.'
# MAGIC AS
# MAGIC $$
# MAGIC     import traceback
# MAGIC     try:
# MAGIC         import re
# MAGIC         # Remove code block markers (e.g., ```python) and strip whitespace```
# MAGIC         python_code = re.sub(r"^\s*```(?:python)?|```\s*$", "", python_code).strip()
# MAGIC         # Unescape any escaped newline characters
# MAGIC         python_code = python_code.replace("\\n", "\n")
# MAGIC         # Properly indent the code for wrapping
# MAGIC         indented_code = "\n    ".join(python_code.split("\n"))
# MAGIC         # Define a wrapper function to execute the code
# MAGIC         exec_globals = {}
# MAGIC         exec_locals = {}
# MAGIC         wrapper_code = "def _temp_function():\n    "+indented_code
# MAGIC         exec(wrapper_code, exec_globals, exec_locals)
# MAGIC         # Execute the wrapped function and return its output
# MAGIC         result = exec_locals["_temp_function"]()
# MAGIC         return result
# MAGIC     except Exception as ex:
# MAGIC         return traceback.format_exc()
# MAGIC $$;
# MAGIC
# MAGIC -- let's test our function:
# MAGIC SELECT execute_python_code("return 'Hello! '* 3") as result;

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM-Based Functions
# MAGIC
# MAGIC LLM-based functions use other language models to perform specific tasks at scale. The `ai_query` function allows you to apply LLM prompts to each row of a table, enabling efficient processing of multiple items. This is useful for:
# MAGIC - Processing large datasets with LLM intelligence
# MAGIC - Generating consistent analyses across multiple items
# MAGIC - Creating personalized recommendations at scale

# COMMAND ----------

# MAGIC %md
# MAGIC ### Product Alternative Recommendation
# MAGIC
# MAGIC This function uses an LLM to recommend alternative products based on specific criteria. It processes each product in the database and generates recommendations tailored to the user's needs.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION alternative_prod(
# MAGIC   input_product_id STRING COMMENT 'Unique identifier of the product to find alternatives for, following pattern ^P[0-9]{4}$',
# MAGIC   input_reason STRING COMMENT 'Customer\'s reason for seeking an alternative (e.g., "too expensive", "needs better stability", "requires different storage conditions")'
# MAGIC )
# MAGIC RETURNS TABLE (alternative_option STRING, product_id STRING, product_name STRING)
# MAGIC LANGUAGE SQL
# MAGIC COMMENT 'Compares the specified product with potential alternatives based on customer needs. Uses AI analysis to recommend suitable alternatives specifically addressing the customer\'s stated reason. Only requires product ID and reason for seeking an alternative.'
# MAGIC RETURN 
# MAGIC   WITH input_product AS (
# MAGIC     SELECT * FROM catalog_name.schema_name.table_name 
# MAGIC     WHERE product_id = input_product_id
# MAGIC   )
# MAGIC   SELECT 
# MAGIC     ai_query('databricks-meta-llama-3-3-70b-instruct',
# MAGIC       CONCAT(
# MAGIC         'You are a chemical product specialist. A customer is looking for alternatives to product ', 
# MAGIC         ip.product_name, ' for the following reason: ', input_reason, 
# MAGIC         '. The product has the following specifications: ',
# MAGIC         'Product ID: ', ip.product_id, 
# MAGIC         ', Chemical Formula: ', ip.chemical_formula, 
# MAGIC         ', Molecular Weight: ', ip.molecular_weight, 
# MAGIC         ', Density: ', ip.density,  
# MAGIC         ', Melting Point: ', ip.melting_point, 
# MAGIC         ', Boiling Point: ', ip.boiling_point, 
# MAGIC         ', Application Areas: ', ip.application_areas, 
# MAGIC         ', Storage Conditions: ', ip.storage_conditions, 
# MAGIC         ', Description: ', ip.description, 
# MAGIC         ', Price Per Unit: ', ip.price_per_unit,
# MAGIC         '. Compare the product with the following potential alternative ONLY on the given reason: ',
# MAGIC         'Product ID: ', p.product_id, 
# MAGIC         ', Product Name: ', p.product_name,
# MAGIC         ', Chemical Formula: ', p.chemical_formula, 
# MAGIC         ', Molecular Weight: ', p.molecular_weight, 
# MAGIC         ', Density: ', p.density, 
# MAGIC         ', Melting Point: ', p.melting_point, 
# MAGIC         ', Boiling Point: ', p.boiling_point, 
# MAGIC         ', Application Areas: ', p.application_areas, 
# MAGIC         ', Storage Conditions: ', p.storage_conditions, 
# MAGIC         ', Description: ', p.description, 
# MAGIC         ', Price Per Unit: ', p.price_per_unit
# MAGIC       )
# MAGIC     ) AS alternative_option,
# MAGIC     p.product_id,
# MAGIC     p.product_name
# MAGIC   FROM your_catalog.your_schema.your_table p, input_product ip    
# MAGIC   WHERE p.product_id != input_product_id
# MAGIC   LIMIT 3

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM recommend_product_alternatives('P0001', 'too expensive')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Building an AI system leveraging our Databricks UC functions with Langchain
# MAGIC
# MAGIC These tools can also directly be leveraged on custom model. In this case, you'll be in charge of chaining and calling the functions yourself (the playground does it for you!)
# MAGIC
# MAGIC Langchain makes it easy for you. You can create your own custom AI System using a Langchain model and a list of existing tools (in our case, the tools will be the functions we just created)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable MLflow Tracing
# MAGIC
# MAGIC Enabling MLflow Tracing is required to:
# MAGIC - View the chain's trace visualization in this notebook
# MAGIC - Capture the chain's trace in production via Inference Tables
# MAGIC - Evaluate the chain via the Mosaic AI Evaluation Suite

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start by creating our tools from Unity Catalog
# MAGIC
# MAGIC Let's use UCFunctionToolkit to select which functions we want to use as tool for our demo:

# COMMAND ----------

from databricks.sdk import WorkspaceClient

def get_shared_warehouse(name=None):
    w = WorkspaceClient()
    warehouses = w.warehouses.list()

    # Check for warehouse by exact name (if provided)
    if name:
        for wh in warehouses:
            if wh.name == name:
                return wh

    # Define fallback priorities
    fallback_priorities = [
        lambda wh: wh.name.lower() == "serverless starter warehouse",
        lambda wh: wh.name.lower() == "shared endpoint",
        lambda wh: wh.name.lower() == "dbdemos-shared-endpoint",
        lambda wh: "shared" in wh.name.lower(),
        lambda wh: "dbdemos" in wh.name.lower(),
        lambda wh: wh.num_clusters > 0,
    ]

    # Try each fallback condition in order
    for condition in fallback_priorities:
        for wh in warehouses:
            if condition(wh):
                return wh

    # Raise an exception if no warehouse is found
    raise Exception(
        "Couldn't find any Warehouse to use. Please create one first or pass "
        "a specific name as a parameter to the get_shared_warehouse(name='xxx') function."
    )


def display_tools(tools):
    display(pd.DataFrame([{k: str(v) for k, v in vars(tool).items()} for tool in tools]))

# COMMAND ----------

from langchain_community.tools.databricks import UCFunctionToolkit
import pandas as pd


wh = get_shared_warehouse(name = None) #Get the first shared wh we can. See _resources/01-init for details
print(f'This demo will be using the wg {wh.name} to execute the functions')

def get_tools():
    return (
        UCFunctionToolkit(warehouse_id=wh.id)
        # Include functions as tools using their qualified names.
        # You can use "{catalog_name}.{schema_name}.*" to get all functions in a schema.
        .include(f"{catalog}.{schema}.*")
        .get_tools())

display_tools(get_tools()) #display in a table the tools - see _resource/00-init for details

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's create our langchain agent using the tools we just created

# COMMAND ----------

from langchain_openai import ChatOpenAI
from databricks.sdk import WorkspaceClient

# Note: langchain_community.chat_models.ChatDatabricks doesn't support create_tool_calling_agent yet - it'll soon be availableK. Let's use ChatOpenAI for now
llm = ChatOpenAI(
  base_url=f"{WorkspaceClient().config.host}/serving-endpoints/",
  api_key=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get(),
  model="databricks-meta-llama-3-3-70b-instruct "
)

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks

def get_prompt(history = [], prompt = None):
    if not prompt:
        prompt = """You are an expert chemistry assistant for a chemical manufacturing company. You have access to specialized tools that help you provide accurate, detailed information about chemical products, safety protocols, and processes.

        AVAILABLE TOOLS:
        
        INFORMATION RETRIEVAL:
        - get_product: Retrieve comprehensive details about a specific chemical product using its product ID (format: P####)
        - get_safety_protocols: Access safety guidelines, handling procedures, and precautions for a specific product ID
        - get_reaction_details: Obtain manufacturing process information, reaction conditions, and hazards for a specific product ID
        - analyze_product_quality: Evaluate quality metrics, test results, and reliability statistics for a specific product ID
        
        SEARCH CAPABILITIES:
        - find_similar_products: Discover chemical products similar to a description. Use when users need product recommendations or alternatives
        - find_safety_protocols: Search for safety protocols, handling procedures, or research notes based on description or chemical properties
        - alternative_prod: Find alternative products when given a product ID and specific reason (cost, performance, storage, etc.)
        
        CALCULATION AND CONVERSION:
        - compute_math: Solve mathematical expressions and equations with high precision
        - convert_chemical_unit: Convert between various chemical units (g, kg, mol, L, mL) with molecular weight support for mass/mole conversions
        - execute_python_code: Run custom Python code for specialized calculations or data processing tasks
        
        ENVIRONMENTAL DATA:
        - get_weather: Retrieve current temperature and precipitation data for a specified location using coordinates
        
        RESPONSE GUIDELINES:
        1. Analyze the user query carefully to determine which tool(s) are most appropriate
        2. For product inquiries, first check if a product ID (do not use product name as id) is provided; if not, use search tools
        3. When uncertainty exists between multiple product options, present the most relevant choice
        4. Always provide complete, well-structured responses with clear explanations
        5. For numerical answers, include units and appropriate precision
        6. For safety-related questions, prioritize accurate safety information
        7. Seamlessly integrate information from multiple tools when necessary
        
        Never mention the tools by name to users. Present information as if it comes from your own knowledge. If a question is completely unrelated to chemistry, chemical manufacturing, or the available tools, politely inform the user that you can only assist with chemistry-related inquiries.
        """
    return ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
    ])

# COMMAND ----------

from langchain.agents import AgentExecutor, create_tool_calling_agent
prompt = get_prompt()
tools = get_tools()
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Let's give it a try: Asking to run a simple unit conversion.
# MAGIC Under the hood, we want this to call our Math conversion function:

# COMMAND ----------

#make sure our log is enabled to properly display and trace the call chain 
import mlflow

mlflow.langchain.autolog()
agent_executor.invoke({"input": "what is 1kg in gramms?"})

# COMMAND ----------

agent_executor.invoke({"input": "what is (2+2)*2?"})

# COMMAND ----------

agent_executor.invoke({"input": "get a product similar to Synth for paper industry"})

# COMMAND ----------

agent_executor.invoke({"input": "I need a price for 855g of the product with name SynthChem C402"})

# COMMAND ----------

agent_executor.invoke({"input": "Will i be able to store SynthChem C402 savely outside tomorrow?"})