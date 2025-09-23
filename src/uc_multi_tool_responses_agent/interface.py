#!/usr/bin/env python3
"""
Interactive chat interface for the ToolCallingAgent.
Allows users to have conversations with the agent through a command-line interface.
"""

import sys
from typing import Any
from mlflow.types.responses import ResponsesAgentRequest, Message
from agent import ToolCallingAgent, AgentManager


def print_welcome():
    """Print welcome message and instructions."""
    print("=" * 60)
    print("🤖 Tool Calling Agent Interactive Interface")
    print("=" * 60)
    print("This agent can help you with:")
    print("• Executing Python code")
    print("• Performing mathematical computations")
    print("• Answering questions using available tools")
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
    manager = AgentManager()
    tools = manager.tools

    print("\n📚 Available Tools:")
    print("-" * 30)
    for i, tool in enumerate(tools, 1):
        tool_name = tool.name.replace("__", ".")
        print(f"{i}. {tool_name}")
        # Try to get description from tool spec
        if hasattr(tool, 'spec') and 'function' in tool.spec:
            desc = tool.spec['function'].get('description', 'No description available')
            print(f"   {desc}")
        print()
    print()


def print_response_stream(agent: ToolCallingAgent, request: ResponsesAgentRequest):
    """Print streaming response from the agent and return agent messages."""
    accumulated_text = ""
    tool_calls = []
    agent_messages = []

    try:
        for event in agent.predict_stream(request):
            if event.type == "response.output_item.add_delta":
                if hasattr(event, 'item') and event.item:
                    item = event.item
                    if item.get('type') == 'response.output_text.delta':
                        delta = item.get('delta', '')
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
                        tool_name = item.get('name', 'unknown').replace("__", ".")
                        tool_args = item.get('arguments', 'none')
                        print(f"\n🔧 [Tool Call: {tool_name}]")
                        print(f"📝 [Arguments: {tool_args}]")
                        tool_calls.append((tool_name, tool_args))

                    elif item.get('type') == 'function_call_output':
                        result = item.get('output', '')
                        # For streaming tools, don't show the final result since it was already streamed
                        # For non-streaming tools, show the result
                        if not tool_calls:  # If no tool calls, this is unexpected, but show result anyway
                            if result:
                                print(f"✅ [Tool Result: {result}]")
                            else:
                                print("✅ [Tool Result: (empty)]")
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
    print("Initializing Tool Calling Agent...")

    try:
        # Get the agent instance from singleton manager
        manager = AgentManager()
        agent = manager.initialize()
        print("✅ Agent initialized successfully!")
        print(f"📚 Loaded {len(manager.tools)} tools")

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Make sure you have:")
        print("1. Activated the virtual environment")
        print("2. Run 'databricks auth login --profile e2-demo-field-eng'")
        print("3. All required dependencies installed")
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
                for i, msg in enumerate(conversation_messages[-10:], 1):  # Show last 10 messages
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if len(content) > 50:
                        content = content[:50] + "..."
                    print(f"{i}. {role}: {content}")
                print(f"Total messages: {len(conversation_messages)}")
                print()
                continue

            if user_input.lower() == 'clear':
                conversation_messages = []
                print("🧹 Conversation history cleared.")
                continue

            # Create user message and add to conversation
            user_message = Message(
                role="user",
                content=user_input
            )
            conversation_messages.append(user_message)

            # Create request with only the current user message to avoid agent responding to previous questions
            # The agent maintains its own internal state for tool calling, but we don't want it to re-answer old questions
            current_user_messages = [msg for msg in conversation_messages if msg.role == "user"][-1:]  # Only the most recent user message
            request = ResponsesAgentRequest(input=current_user_messages)

            # Get agent response
            print("\n🤖 Agent: ", end='', flush=True)
            agent_response_items = print_response_stream(agent, request)

            # Add agent response messages to conversation history
            # Only add the final assistant message to avoid empty content issues
            final_assistant_content = ""
            tool_calls_for_final_message = []

            for item in agent_response_items:
                if item.get('type') == 'response.output_text':
                    final_assistant_content += item.get('text', '')
                elif item.get('type') == 'function_call':
                    # Collect tool calls but don't add intermediate messages
                    tool_calls_for_final_message.append({
                        "id": item.get('call_id', ''),
                        "type": "function",
                        "function": {
                            "name": item.get('name', ''),
                            "arguments": item.get('arguments', '')
                        }
                    })
                # Note: We don't add function_call_output (tool) messages to conversation history
                # as they cause validation errors and aren't needed for conversation context

            # Add final assistant message with accumulated content and tool calls
            if final_assistant_content or tool_calls_for_final_message:
                assistant_message = Message(
                    role="assistant",
                    content=final_assistant_content,
                    tool_calls=tool_calls_for_final_message if tool_calls_for_final_message else None
                )
                conversation_messages.append(assistant_message)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
