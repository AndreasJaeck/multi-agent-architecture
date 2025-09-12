def supervisor_system_prompt(experts_md: str) -> str:
    return (
        "You are a supervisor agent that coordinates domain experts using structured planning.\n"
        "Process: Analyze question → Create execution plan → Consult relevant experts → Synthesize if needed\n"
        f"Available experts:\n{experts_md}".strip()
    )


def planning_prompt(question: str, experts_md: str) -> str:
    return f"""You are a supervisor agent. Create a short markdown plan to answer the user's question.

Question:
{question}

Available experts (refer to them by name exactly as shown):
{experts_md}

Output a concise plan in markdown with bullet points only. Each bullet should start with "- Call {{AgentName}} to ..." and include the focused instruction for that expert. Keep 2–4 bullets max. When planning a verification step, include the precise expression and expected value (if known) so the expert can act without asking for clarification. Optionally add a final bullet "- Consider synthesizing the responses if needed". Do not include any JSON or commentary, only the bullet list.
""".strip()


def synthesis_prompt(question: str, responses_md: str) -> str:
    return f"""Question: {question}

Responses:
{responses_md}

Synthesize into a brief, coherent answer to the original question:""".strip()
