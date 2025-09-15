import argparse
import mlflow
from mlflow.types.responses import ResponsesAgentRequest

from multi_agent.supervisor.fmapi_supervisor_agent import AGENT

def print_event(ev) -> None:
    if ev.type in ("response.text.delta", "response.output_text.delta"):
        delta = getattr(ev, "delta", {}) if hasattr(ev, "delta") else {}
        if isinstance(delta, str):
            text = delta
        elif isinstance(delta, dict):
            text = delta.get("content") or delta.get("text") or ""
        else:
            text = getattr(delta, "content", "") or getattr(delta, "text", "") or str(delta)
        if text:
            print(text, end="", flush=True)
    elif ev.type == "response.output_item.done":
        item = getattr(ev, "item", None)
        if item is None:
            return
        # Only show tool call metadata; skip printing final aggregated output_text/message to avoid duplication
        if isinstance(item, dict):
            t = item.get("type")
            if t == "function_call":
                print(f"\n[TOOL CALL] {item.get('name')} args={item.get('arguments')}", flush=True)
            elif t == "function_call_output":
                out = item.get("output", "")
                print(f"\n[TOOL OUT] output={str(out)[:500]}...", flush=True)
        else:
            t = getattr(item, "type", "")
            if t == "function_call":
                name = getattr(item, "name", "")
                args = getattr(item, "arguments", "")
                print(f"\n[TOOL CALL] {name} args={args}", flush=True)
            elif t == "function_call_output":
                out = getattr(item, "output", "")
                print(f"\n[TOOL OUT] output={str(out)[:500]}...", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Stream outputs from the fmapi supervisor agent (Databricks endpoints)")
    parser.add_argument(
        "--question",
        "-q",
        default="Provide a detailed 4-6 bullet summary of current sustainability trends in automotive coatings",
        help="User question to send to the agent",
    )
    parser.add_argument(
        "--supervisor-endpoint",
        "-e",
        default=None,
        help="Databricks model serving endpoint for the supervisor (defaults from SupervisorConfig)",
    )
    args = parser.parse_args()

    req = ResponsesAgentRequest(input=[{"role": "user", "content": args.question}])

    print("=== Streaming ===")
    for ev in AGENT.predict_stream(req):
        print_event(ev)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()