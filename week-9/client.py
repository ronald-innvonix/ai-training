import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Reads GROQ_API_KEY from a .env file in this folder
load_dotenv()

# Absolute path to server.py so it works from any working directory
SERVER_PATH = str(Path(__file__).parent / "server.py")

# Groq model with good tool-calling support. Change if you like.
GROQ_MODEL = "openai/gpt-oss-120b"


async def show_resources(client: MultiServerMCPClient) -> None:
    """Demo: read a RESOURCE directly (resources are app-driven, not LLM-driven)."""
    try:
        blobs = await client.get_resources(
            "employee_directory", uris=["directory://departments"]
        )
        print("Resource directory://departments ->")
        print(blobs[0].as_string(), "\n")
    except Exception as exc:  # older adapter versions may lack get_resources
        print(f"(Skipping resource demo: {exc})\n")


async def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Set GROQ_API_KEY in your environment or a .env file first.")

    # 1) Describe how to start/connect to each MCP server.
    #    You can add more servers to this dict later (e.g. a "vendors" server).
    client = MultiServerMCPClient(
        {
            "employee_directory": {
                "transport": "stdio",
                "command": sys.executable,  # use the same Python as this venv
                "args": [SERVER_PATH],
            }
        }
    )

    # 2) Discover tools dynamically from the server.
    tools = await client.get_tools()
    print("Tools discovered from MCP server:")
    for t in tools:
        print(f"  - {t.name}: {t.description.splitlines()[0]}")
    print()

    # Optional: show how resources are read
    await show_resources(client)

    # 3) Build the agent: Groq LLM + discovered tools.
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    agent = create_react_agent(llm, tools)

    # 4) Chat loop
    print("Ask about the directory (type 'exit' to quit).")
    print("Try: 'Who in Engineering knows python?' or 'Who reports to Priya Patel?'\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break

        # ainvoke runs the full loop: LLM -> tool call(s) via MCP -> final answer
        result = await agent.ainvoke({"messages": [("user", question)]})

        # Print which tools the LLM chose (helps you learn how it works)
        for msg in result["messages"]:
            for call in getattr(msg, "tool_calls", None) or []:
                print(f"  [tool call] {call['name']}({call['args']})")

        # The last message is the LLM's final answer
        print(f"Assistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(main())