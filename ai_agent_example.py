"""
AI Agent Example using the Anthropic Claude API

This example demonstrates a research agent that:
- Uses custom tools (web search simulation, calculator, note-taking)
- Runs an agentic loop until Claude finishes the task
- Handles multiple tool calls in a single turn
- Uses Claude Opus 4.6 with adaptive thinking
"""

import json
import math
import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"

# ── Tool implementations ──────────────────────────────────────────────────────

def search_web(query: str) -> str:
    """Simulate a web search (replace with a real search API in production)."""
    mock_results = {
        "climate change": (
            "Climate change refers to long-term shifts in global temperatures and weather patterns. "
            "Since the 1800s, human activities—primarily burning fossil fuels—have been the main driver. "
            "Key impacts include rising sea levels, more frequent extreme weather events, and ecosystem disruption."
        ),
        "python programming": (
            "Python is a high-level, interpreted programming language known for its clear syntax and "
            "readability. Created by Guido van Rossum (1991). Widely used in web development, data science, "
            "AI/ML, automation, and scientific computing."
        ),
        "anthropic claude": (
            "Anthropic is an AI safety company that develops Claude, a family of AI assistants. "
            "Claude is designed to be helpful, harmless, and honest. The latest models include "
            "Claude Opus 4.6, Sonnet 4.6, and Haiku 4.5."
        ),
    }
    query_lower = query.lower()
    for key, result in mock_results.items():
        if key in query_lower:
            return result
    return f"Search results for '{query}': No specific results found. Try a more specific query."


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    allowed = set("0123456789+-*/()., abcdefghijklmnopqrstuvwxyz_")
    if not all(c in allowed for c in expression.lower()):
        return "Error: Expression contains invalid characters."
    try:
        # Expose safe math functions
        safe_env = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        safe_env["abs"] = abs
        result = eval(expression, {"__builtins__": {}}, safe_env)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def take_note(title: str, content: str) -> str:
    """Save a note (in-memory for this example)."""
    notes[title] = content
    return f"Note '{title}' saved successfully."


def list_notes() -> str:
    """List all saved notes."""
    if not notes:
        return "No notes saved yet."
    return "\n".join(f"- {title}: {content}" for title, content in notes.items())


# ── Tool definitions (JSON schema) ───────────────────────────────────────────

TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for information on a given topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'climate change causes'.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression, e.g. 'sqrt(144) + 2**10'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A safe mathematical expression using Python syntax.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "take_note",
        "description": "Save a note with a title and content for later reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "A short title for the note."},
                "content": {"type": "string", "description": "The body of the note."},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "list_notes",
        "description": "List all notes that have been saved in this session.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

TOOL_FUNCTIONS = {
    "search_web": lambda inp: search_web(inp["query"]),
    "calculate": lambda inp: calculate(inp["expression"]),
    "take_note": lambda inp: take_note(inp["title"], inp["content"]),
    "list_notes": lambda _: list_notes(),
}

# ── Agentic loop ──────────────────────────────────────────────────────────────

def run_agent(user_prompt: str) -> str:
    """
    Run the agent loop: call Claude → execute tools → feed results back → repeat
    until Claude signals it is done (stop_reason == "end_turn").
    """
    messages = [{"role": "user", "content": user_prompt}]
    print(f"\n{'='*60}")
    print(f"User: {user_prompt}")
    print(f"{'='*60}")

    while True:
        # Stream the response to avoid HTTP timeouts on long outputs
        with client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            tools=TOOLS,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        # Display non-tool content as it comes in
        for block in response.content:
            if block.type == "thinking":
                print(f"\n[Thinking]\n{block.thinking[:300]}{'...' if len(block.thinking) > 300 else ''}")
            elif block.type == "text" and block.text.strip():
                print(f"\nAssistant: {block.text}")

        # If Claude is done, return the final text
        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if b.type == "text"),
                "(no text response)",
            )
            return final_text

        # Otherwise, execute all tool calls and collect results
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            # Shouldn't happen, but guard against infinite loops
            break

        tool_results = []
        for tool_use in tool_use_blocks:
            print(f"\n[Tool call] {tool_use.name}({json.dumps(tool_use.input)})")
            try:
                result = TOOL_FUNCTIONS[tool_use.name](tool_use.input)
            except Exception as exc:
                result = f"Error: {exc}"
            print(f"[Tool result] {result[:200]}{'...' if len(result) > 200 else ''}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result,
            })

        # Append the assistant turn (with tool_use blocks) and the tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "(agent loop ended without final response)"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    notes: dict[str, str] = {}  # In-memory note store shared across tool calls

    # Patch the note functions to use the local `notes` dict
    TOOL_FUNCTIONS["take_note"] = lambda inp: take_note(inp["title"], inp["content"])
    TOOL_FUNCTIONS["list_notes"] = lambda _: list_notes()

    tasks = [
        # Task 1: multi-tool research task
        (
            "Research what climate change is, calculate how many years until 2100 "
            "from 2024, and save a brief summary note titled 'Climate Research'."
        ),
        # Task 2: math-heavy task
        (
            "Calculate the area of a circle with radius 7 (use pi from math), "
            "then calculate 2 to the power of 16, and list any notes I have saved."
        ),
    ]

    for task in tasks:
        # Reset notes only between top-level demo runs if desired
        result = run_agent(task)
        print(f"\n{'─'*60}")
        print(f"Final answer: {result}")
        print(f"{'─'*60}\n")
