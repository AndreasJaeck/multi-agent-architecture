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

Available experts and tools (refer to them by name exactly as shown):
{experts_md}

Output a concise to-do list in markdown with bullet points only.
""".strip()


def synthesis_prompt(question: str, responses_md: str) -> str:
    return f"""Question: {question}

Responses:
{responses_md}

Synthesize into a brief, coherent answer to the original question:""".strip()
