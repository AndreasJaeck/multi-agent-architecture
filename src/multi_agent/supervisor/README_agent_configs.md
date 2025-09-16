# Agent Configuration

This directory contains agent configuration files for the FMAPI supervisor agent.

## Files

### `agent_configs.py` (SENSITIVE - Not Committed)
Contains the actual agent configurations with real endpoints, Genie space IDs, and other sensitive information.

**⚠️ IMPORTANT:** This file is excluded from version control via `.gitignore` and should never be committed to the repository.

### `agent_configs_example.py` (Safe for Version Control)
Contains anonymized example configurations that can be safely committed and shared.

## Setup

1. Copy `agent_configs_example.py` to `agent_configs.py`:
   ```bash
   cp agent_configs_example.py agent_configs.py
   ```

2. Edit `agent_configs.py` with your actual configuration values:
   - Replace `your_*` placeholders with actual values
   - Update endpoints with real Databricks model serving endpoints
   - Add actual Genie space IDs
   - Configure vector search indices and functions

## Configuration Structure

Each agent is configured using the `AgentConfig` dataclass:

```python
AgentConfig(
    name="AgentName",                    # Human-readable name
    description="Agent description",     # Used for tool routing
    endpoint="endpoint_name",            # Databricks model serving endpoint
    system_prompt="System prompt",       # Agent's system message
    capabilities="What the agent can do", # Optional capabilities description
    domain="domain_category",            # Optional domain categorization
    resources=ResourceConfig(            # Underlying Databricks resources
        genie_spaces=["space_id_1"],
        functions=["catalog.schema.function"],
        vector_search_indices=["catalog.schema.index"]
    )
)
```

## Security

- Never commit `agent_configs.py` to version control
- Keep sensitive information (endpoints, space IDs, etc.) out of version control
- Use the example file as a template for new configurations

## Usage

The supervisor agent will automatically load configurations from `agent_configs.py`. If the file doesn't exist, it will fall back to the example configurations.
