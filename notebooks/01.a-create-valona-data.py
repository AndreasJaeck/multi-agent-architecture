# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Table pre-procecssing 
# MAGIC Try to clean the table up a bit and concat potential search content.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     results_itemId,
# MAGIC     results_header AS title,
# MAGIC     results_created AS created_date,
# MAGIC     results_modified AS modified_date,
# MAGIC     results_status,
# MAGIC     results_source,
# MAGIC     item_link,
# MAGIC     -- Concatenate text fields for vector search
# MAGIC     CONCAT_WS(' ', 
# MAGIC         results_source,
# MAGIC         results_header,
# MAGIC         results_summary,
# MAGIC         results_body,
# MAGIC         CAT_NAME
# MAGIC     ) AS search_text,
# MAGIC     -- Keep original fields for reference
# MAGIC     CAT_NAME,
# MAGIC     CAT_ID
# MAGIC FROM 
# MAGIC     hong_zhu_demo_catalog.basf_genie_agent.valona
# MAGIC WHERE 
# MAGIC     results_status = 'Published'
# MAGIC ORDER BY 
# MAGIC     results_modified DESC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Create Table **valona_optimized** with revevant columns and concatenated `serach text`

# COMMAND ----------

sql_statement = """
SELECT 
    results_itemId,
    results_header AS title,
    results_created AS created_date,
    results_modified AS modified_date,
    results_status,
    results_source,
    item_link,
    -- Concatenate text fields for vector search
    CONCAT_WS(' ', 
        results_source,
        results_header,
        results_summary,
        results_body,
        CAT_NAME
    ) AS search_text,
    -- Keep original fields for reference
    CAT_NAME,
    CAT_ID
FROM 
    hong_zhu_demo_catalog.basf_genie_agent.valona
WHERE 
    results_status = 'Published'
ORDER BY 
    results_modified DESC
"""

create_table_sql = f"""
CREATE TABLE hong_zhu_demo_catalog.basf_genie_agent.valona_optimized
USING DELTA AS
{sql_statement}
"""

spark.sql(create_table_sql)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Enable **change data capture (CDF)** to unlock vector index suppor. 
# MAGIC
# MAGIC This supports `Delta Sync` to create a vector search index in Databricks. Ensures that the vector search index stays in sync with updates to the underlying Delta table, allowing incremental updates to be identified and processed efficiently

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `hong_zhu_demo_catalog`.`basf_genie_agent`.`valona_optimized` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)