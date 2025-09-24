#!/usr/bin/env python3
"""
Interactive chat interface for the FMAPI Supervisor Agent.
Allows users to have conversations with the supervisor agent through a command-line interface.
"""

import sys
from typing import Any
from mlflow.types.responses import ResponsesAgentRequest, Message
from .fmapi_supervisor_agent import AGENT


def print_welcome():
    """Print welcome message and instructions."""
    print("=" * 60)
    print("🤖 FMAPI Supervisor Agent Interactive Interface")
    print("=" * 60)
    print("This supervisor agent can delegate tasks to specialist agents:")
    tools = AGENT.get_tool_specs()
    for tool in tools:
        name = tool["function"]["name"]
        desc = tool["function"]["description"]
        print(f"• {name}: {desc}")
    print()
    print("Commands:")
    print("• Type your message and press Enter")
    print("• Type 'quit' or 'exit' to end the conversation")
    print("• Type 'help' to see this message again")
    print("• Type 'tools' to see available tools")
    print("• Type 'history' to see conversation history")
    print("• Type 'clear' to clear conversation history")
    print("=" * 60)
    print()


def print_available_tools():
    """Print information about available tools."""
    tools = AGENT.get_tool_specs()

    print("\n📚 Available Specialist Agents:")
    print("-" * 40)
    for i, tool in enumerate(tools, 1):
        func_spec = tool["function"]
        name = func_spec["name"]
        desc = func_spec["description"]
        print(f"{i}. {name}")
        print(f"   {desc}")
        print()
    print()


def print_response_stream(request: ResponsesAgentRequest):
    """Print streaming response from the supervisor agent and return agent messages."""
    accumulated_text = ""
    tool_calls = []
    agent_messages = []

    try:
        for event in AGENT.predict_stream(request):
            if event.type == "response.output_text.delta":
                delta = getattr(event, 'delta', '')
                # Check if this is streaming tool content or LLM content
                if tool_calls:
                    # Streaming tool content - show during tool execution
                    print(delta, end='', flush=True)
                else:
                    # Regular LLM streaming
                    print(delta, end='', flush=True)
                    accumulated_text += delta

            elif event.type == "response.output_item.done":
                if accumulated_text:
                    print()  # New line after complete response
                    accumulated_text = ""

                if hasattr(event, 'item') and event.item:
                    item = event.item
                    agent_messages.append(item)  # Collect agent messages

                    if item.get('type') == 'function_call':
                        tool_name = item.get('name', 'unknown')
                        tool_args = item.get('arguments', 'none')
                        print(f"\n🔧 [Tool Call: {tool_name}]")
                        print(f"📝 [Arguments: {tool_args}]")
                        tool_calls.append((tool_name, tool_args))

                    elif item.get('type') == 'function_call_output':
                        result = item.get('output', '')
                        # Show the tool result
                        if result:
                            print(f"\n✅ [Tool Result: {result[:200]}{'...' if len(result) > 200 else ''}]")
                        else:
                            print("\n✅ [Tool Result: (empty)]")
                        # Clear tool_calls after processing to reset for next interaction
                        tool_calls.clear()

        print()  # Final newline

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

    return agent_messages


def main():
    """Main interactive loop."""
    print("Initializing FMAPI Supervisor Agent...")

    try:
        # The AGENT is already initialized from fmapi_supervisor_agent.py
        tools = AGENT.get_tool_specs()
        print("✅ Supervisor Agent initialized successfully!")
        print(f"📚 Loaded {len(tools)} specialist agents")

    except Exception as e:
        print(f"❌ Failed to initialize supervisor agent: {e}")
        print("Make sure you have:")
        print("1. Activated the virtual environment")
        print("2. Run 'databricks auth login --profile e2-demo-field-eng'")
        print("3. All required dependencies installed")
        print("4. Valid agent_configs.py file")
        sys.exit(1)

    print_welcome()

    # Main conversation loop - maintain full conversation history
    conversation_messages = []

    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == 'help':
                print_welcome()
                continue

            if user_input.lower() == 'tools':
                print_available_tools()
                continue

            if user_input.lower() == 'history':
                print("\n📜 Conversation History:")
                print("Note: The supervisor agent manages conversation state internally.")
                print("Previous interactions are not stored in this interface.")
                print("Each message is processed as a complete conversation turn.")
                print()
                continue

            if user_input.lower() == 'clear':
                print("🧹 Conversation state is managed by the supervisor agent.")
                print("No local history to clear.")
                continue

            # Create user message and add to conversation
            user_message = Message(
                role="user",
                content=user_input
            )
            conversation_messages.append(user_message)

            # Create request with only the current user message
            # The supervisor agent manages conversation state internally and does planning/tool execution
            current_user_messages = [msg for msg in conversation_messages if msg.role == "user"][-1:]  # Only the most recent user message
            request = ResponsesAgentRequest(input=current_user_messages)

            # Get agent response
            print("\n🤖 Supervisor Agent: ", end='', flush=True)
            agent_response_items = print_response_stream(request)

            # Note: The supervisor agent manages its own conversation state internally
            # We don't need to maintain conversation history in the interface since the agent
            # handles planning, tool calls, and responses as a complete conversation turn

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
