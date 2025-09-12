import os
import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from openai import OpenAI


class DatabricksClientManager:
    _workspace_client = None
    _model_serving_client = None

    @classmethod
    def get_workspace_client(cls) -> WorkspaceClient:
        if cls._workspace_client is None:
            cls._workspace_client = WorkspaceClient()
        return cls._workspace_client

    @classmethod
    def get_model_serving_client(cls) -> OpenAI:
        if cls._model_serving_client is None:
            workspace_client = cls.get_workspace_client()
            cls._model_serving_client = workspace_client.serving_endpoints.get_open_ai_client()
        return cls._model_serving_client


class LLMClient:
    def __init__(self, model_endpoint: str):
        self.model_endpoint = model_endpoint
        self.client = DatabricksClientManager.get_model_serving_client()

    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError))
    @mlflow.trace(span_type=SpanType.LLM)
    def make_llm_call(self, messages, temperature: float, max_tokens: int):
        if os.getenv("DEBUG_PROMPTS", "0") == "1":
            try:
                print("\n=== DEBUG_PROMPTS: LLM CALL ===")
                print(f"Endpoint: {self.model_endpoint}")
                print(f"Temperature: {temperature} | Max tokens: {max_tokens}")
                print("Messages:")
                for m in messages:
                    role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
                    print(f"- {role}: {str(content)[:800]}")
                print("=== END DEBUG_PROMPTS ===\n")
            except Exception:
                pass
        return self.client.chat.completions.create(
            model=self.model_endpoint,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def extract_response_content(response) -> str:
        if not response:
            return "No response received"
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
            return content or "No content in choices"
        if hasattr(response, "messages") and response.messages:
            content = response.messages[0].get("content", "")
            return content or "No content in messages"
        return f"Unexpected response format: {type(response)}"
