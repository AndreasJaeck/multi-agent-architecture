# Supply Chain Team - Advanced Genie API Features & Security

## Hackathon Goals

**Core Objectives:**
- Test OBO authentication with Row-Level Security (RLS) and Column-Level Masking (CM) via Genie API. 
- Build robust multi-agent systems for supply chain optimization and risk management


## Key Resources

### Security & Authentication
🔐 **[Databricks Agent Authentication Guide](https://docs.databricks.com/aws/en/generative-ai/agent-framework/agent-authentication#on-behalf-of-user-authentication)** - Comprehensive guide for implementing On-Behalf-Of (OBO) authentication in agent frameworks


## OBO authentifaction Genie Spaces
```python
## OBO for Genie Spaces:
from databricks_langchain.genie import GenieAgent
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials


# Configure a Databricks SDK WorkspaceClient to use on behalf of end
# user authentication
user_client = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())


genie_agent = GenieAgent(
    genie_space_id="space-id",
    genie_agent_name="Genie",
    description="This Genie space has access to sales data in Europe"
    client=user_client
)

# Use the Genie SDK methods available through WorkspaceClient
try:
    response = agent.invoke("Your query here")
except Exception as e:
    _logger.debug("Skipping Genie due to no permissions")
```


### Genie API Orchestrator
🤖 **[LangGraph Multi-Agent Genie Pattern](https://docs.databricks.com/aws/en/notebooks/source/generative-ai/langgraph-multiagent-genie-pat.html)** - Reference implementation for orchestrating multiple agents with Genie spaces using LangGraph



## Security Integration

- **ECT Team**: Cross-domain security patterns
- **Marketing Team**: Data access security for analytics
- **Multi-layered Architecture Team**: Enterprise authentication frameworks


