# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Table pre-procecssing 
# MAGIC Try to clean the table up a bit and concat potential search content.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     id AS report_id,
# MAGIC     account,
# MAGIC     country,
# MAGIC     state,
# MAGIC     city,
# MAGIC     visit_date,
# MAGIC     weekday,
# MAGIC     sbu,
# MAGIC     od,
# MAGIC     industry,
# MAGIC     visit_reports,
# MAGIC     -- Concatenate text fields for vector search
# MAGIC     CONCAT_WS(' ', 
# MAGIC         account,
# MAGIC         country,
# MAGIC         visit_date,
# MAGIC         sbu,
# MAGIC         od,
# MAGIC         industry,
# MAGIC         visit_reports
# MAGIC     ) AS search_text
# MAGIC FROM 
# MAGIC     hong_zhu_demo_catalog.basf_genie_agent.visit_reports
# MAGIC WHERE 
# MAGIC     visit_reports IS NOT NULL
# MAGIC     AND industry IS NOT NULL
# MAGIC     AND account IS NOT NULL 
# MAGIC     AND visit_date IS NOT NULL
# MAGIC     AND sbu IS NOT NULL AND od IS NOT NULL
# MAGIC ORDER BY 
# MAGIC     visit_date DESC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Create Table **visit_reports_optimized** with revevant columns and concatenated `serach text`

# COMMAND ----------

sql_statement = """
SELECT 
    id AS report_id,
    account,
    country,
    state,
    city,
    visit_date,
    weekday,
    sbu,
    od,
    industry,
    visit_reports,
    -- Concatenate text fields for vector search
    CONCAT_WS(' ', 
        account,
        country,
        visit_date,
        sbu,
        od,
        industry,
        visit_reports
    ) AS search_text
FROM 
    hong_zhu_demo_catalog.basf_genie_agent.visit_reports
WHERE 
    visit_reports IS NOT NULL
    AND industry IS NOT NULL
    AND account IS NOT NULL 
    AND visit_date IS NOT NULL
    AND sbu IS NOT NULL AND od IS NOT NULL
ORDER BY 
    visit_date DESC
"""

create_table_sql = f"""
CREATE OR REPLACE TABLE hong_zhu_demo_catalog.basf_genie_agent.visit_reports_optimized
USING DELTA AS
{sql_statement}
"""

spark.sql(create_table_sql)

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `hong_zhu_demo_catalog`.`basf_genie_agent`.`visit_reports_optimized`
# MAGIC ALTER COLUMN report_id SET NOT NULL;
# MAGIC
# MAGIC ALTER TABLE `hong_zhu_demo_catalog`.`basf_genie_agent`.`visit_reports_optimized`
# MAGIC ADD CONSTRAINT pk_report_id PRIMARY KEY(report_id);

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Enable **change data capture (CDF)** to unlock vector index suppor. 
# MAGIC
# MAGIC This supports `Delta Sync` to create a vector search index in Databricks. Ensures that the vector search index stays in sync with updates to the underlying Delta table, allowing incremental updates to be identified and processed efficiently

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE `hong_zhu_demo_catalog`.`basf_genie_agent`.`visit_reports_optimized` SET TBLPROPERTIES (delta.enableChangeDataFeed = true)