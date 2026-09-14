"""
=============================================================================
WEEK 5 SUBMISSION — LangChain: Tools, Tool Calling & ReAct Agents
=============================================================================
Topics covered in this file:
    1. Using a LangChain pre-built tool with an LLM
    2. Creating custom tools (stock price + weather) and binding them to an LLM
    3. Tool calling (LLM decides which tool to call, with what arguments)
    4. Building a ReAct agent with 2-4 tools

Design goal:
    The LLM provider is fully pluggable. Switch between OpenAI, Groq, or
    Mistral (mistral-large-latest) just by changing MODEL_PROVIDER below —
    no other code needs to change, because every task calls the same
    `get_llm()` factory function.

Install (uncomment what you need):
    pip install langchain langchain-community langchain-openai \
                langchain-groq langchain-mistralai langgraph \
                yfinance requests duckduckgo-search

Set the relevant API key as an environment variable before running, e.g.:
    export OPENAI_API_KEY="..."
    export GROQ_API_KEY="..."
    export MISTRAL_API_KEY="..."
=============================================================================
"""

import os
import logging
import argparse
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool

# Load environment variables from a .env file when present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional but recommended; we'll continue if it's not installed
    pass


# -----------------------------------------------------------------------
# 0. MODEL FACTORY — this is what makes the whole script "swap-in-swap-out"
# -----------------------------------------------------------------------
# Just change this one variable to change which LLM provider powers
# every task below (pre-built tool demo, custom tools, and the agent).
MODEL_PROVIDER = "groq"   # options: "openai" | "groq" | "mistral"


def get_llm(provider: str = MODEL_PROVIDER, temperature: float = 0.2):
    """
    Factory function that returns a chat model instance for the requested
    provider. Keeping this logic in one place means every other function
    in the file is provider-agnostic — they just call get_llm() and use
    whatever comes back.
    """
    provider = provider.lower()

    try:
        if provider == "openai":
            # Requires: pip install langchain-openai, OPENAI_API_KEY env var
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

        elif provider == "groq":
            # Requires: pip install langchain-groq, GROQ_API_KEY env var
            from langchain_groq import ChatGroq

            return ChatGroq(model="openai/gpt-oss-20b", temperature=temperature)

        elif provider == "mistral":
            # Requires: pip install langchain-mistralai, MISTRAL_API_KEY env var
            from langchain_mistralai import ChatMistralAI

            return ChatMistralAI(model="mistral-large-latest", temperature=temperature)

        else:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose from: openai, groq, mistral."
            )
    except ImportError as e:
        raise ImportError(
            f"Failed to import provider client for '{provider}': {e}.\n"
            "Install the provider package (see README) or set a different MODEL_PROVIDER."
        )


# =========================================================================
# TASK 1 — Provide a LangChain pre-built tool to an LLM
# =========================================================================
def task1_prebuilt_tool_demo():
    """
    Demonstrates using an out-of-the-box LangChain tool: DuckDuckGoSearchRun.
    Step 1: use the tool directly (no LLM involved).
    Step 2: bind it to an LLM so the LLM can decide to call it.
    """
    logging.info("=== TASK 1: Pre-built LangChain Tool (DuckDuckGo Search) ===")

    # LangChain community tools live in `langchain_community.tools`.
    # This one wraps the duckduckgo-search package for free web search
    # (no API key needed).
    from langchain_community.tools import DuckDuckGoSearchRun

    search_tool = DuckDuckGoSearchRun()

    # --- Step 1: call the tool directly, just like a normal function ---
    # Use the tool directly with a short timeout where possible
    try:
        raw_result = search_tool.invoke("Who won the latest ICC Cricket World Cup?")
    except Exception as e:
        logging.error("Direct tool call failed: %s", e)
        raw_result = ""

    logging.info("Direct tool call result (truncated): %s...", raw_result[:300])

    # --- Step 2: bind the tool to an LLM so the LLM decides when to use it ---
    llm = get_llm()
    llm_with_tools = llm.bind_tools([search_tool])

    try:
        response = llm_with_tools.invoke(
            "Search the web and tell me who won the latest ICC Cricket World Cup."
        )
        # If the LLM decided to call the tool, the call details show up here.
        logging.info("LLM tool_calls (LLM decided which tool + args to use): %s", response.tool_calls)
        # The final answer from the LLM is in response.content
        logging.info("LLM final answer (truncated): %s...", response.content[:300])
    except Exception as e:
        logging.error("LLM invocation failed: %s", e)


# =========================================================================
# TASK 2 — Create custom tools and bind them to an LLM
# =========================================================================
@tool
def get_current_stock_price(ticker: str) -> str:
    """
    Get the current/latest stock price for a given ticker symbol
    (e.g. 'AAPL', 'TSLA', 'INFY.NS').
    """
    # yfinance gives free, no-API-key stock data.
    try:
        import yfinance as yf
    except Exception:
        return "yfinance is not installed. Install dependencies via requirements.txt"

    stock = yf.Ticker(ticker)
    price = getattr(stock, "fast_info", {}) and stock.fast_info.get("last_price")
    if price is None:
        return f"Could not fetch price for '{ticker}'. Check the ticker symbol."
    return f"The current price of {ticker} is approximately {price:.2f}."


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather (temperature and conditions) for a given city name.
    """
    # We use Open-Meteo (free, no API key) in two steps:
    #   1. Geocode the city name -> latitude/longitude
    #   2. Fetch current weather for that lat/lon
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        geo_resp = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10).json()
    except Exception as e:
        logging.error("Geocoding request failed: %s", e)
        return f"Could not fetch location data for '{city}': {e}"

    if not geo_resp.get("results"):
        return f"Could not find location data for '{city}'."

    location = geo_resp["results"][0]
    lat, lon = location["latitude"], location["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    try:
        weather_resp = requests.get(
            weather_url,
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=10,
        ).json()
    except Exception as e:
        logging.error("Weather request failed: %s", e)
        return f"Could not fetch weather for '{city}': {e}"

    current = weather_resp.get("current_weather", {})
    temp = current.get("temperature")
    wind = current.get("windspeed")

    if temp is None:
        return f"Could not fetch weather for '{city}'."

    return (
        f"Current weather in {city}: {temp}°C, wind speed {wind} km/h "
        f"(source: Open-Meteo)."
    )


def task2_custom_tools_demo():
    """
    Binds our two custom tools (stock price + weather) to an LLM and lets
    the LLM decide which one to call, with which arguments, based on the
    user's question.
    """
    logging.info("=== TASK 2: Custom Tools (stock price + weather), tool calling ===")

    llm = get_llm()
    custom_tools = [get_current_stock_price, get_weather]
    llm_with_tools = llm.bind_tools(custom_tools)

    # A query that should trigger the weather tool
    try:
        response_weather = llm_with_tools.invoke("What's the weather like in Ahmedabad?")
        logging.info("Weather query -> tool_calls: %s", response_weather.tool_calls)
    except Exception as e:
        logging.error("Weather LLM invocation failed: %s", e)

    # A query that should trigger the stock price tool
    try:
        response_stock = llm_with_tools.invoke("What is Tesla's current stock price?")
        logging.info("Stock query -> tool_calls: %s", response_stock.tool_calls)
    except Exception as e:
        logging.error("Stock LLM invocation failed: %s", e)

    # Manually execute whichever tool the LLM asked for, and show the result.
    # (In a full pipeline, a ToolNode/agent executor normally does this step
    # automatically — see Task 3 below.)
    # If tool calls exist, execute them to show observations (agent executors normally do this)
    for resp in (locals().get("response_stock"), locals().get("response_weather")):
        if not resp:
            continue
        for call in getattr(resp, "tool_calls", []):
            try:
                if call["name"] == "get_current_stock_price":
                    result = get_current_stock_price.invoke(call["args"])
                    logging.info("Executed tool -> %s", result)
                if call["name"] == "get_weather":
                    result = get_weather.invoke(call["args"])
                    logging.info("Executed tool -> %s", result)
            except Exception as e:
                logging.error("Executing tool call failed: %s", e)


# =========================================================================
# TASK 3 — Build a ReAct agent with 2-4 tools
# =========================================================================
def task3_react_agent_demo():
    """
    Wires up a ReAct-style agent (Reason + Act loop) using LangGraph's
    prebuilt create_react_agent helper. The agent is given 3 tools:
        1. DuckDuckGoSearchRun   (pre-built LangChain tool, Task 1)
        2. get_current_stock_price (custom tool, Task 2)
        3. get_weather              (custom tool, Task 2)

    The agent will reason about the user's question, decide which tool(s)
    to call, call them, observe the results, and produce a final answer —
    the classic ReAct "Thought -> Action -> Observation -> ... -> Answer"
    loop, all handled internally by langgraph.
    """
    logging.info("=== TASK 3: ReAct Agent with multiple tools ===")

    from langchain_community.tools import DuckDuckGoSearchRun
    from langgraph.prebuilt import create_react_agent

    llm = get_llm()

    tools = [
        DuckDuckGoSearchRun(),      # pre-built tool
        get_current_stock_price,   # custom tool
        get_weather,                # custom tool
    ]

    # create_react_agent wires the LLM + tools into a runnable ReAct agent
    # graph. It handles the reasoning loop for us: the LLM decides an
    # action, the graph executes the matching tool, feeds the observation
    # back to the LLM, and repeats until the LLM is ready to answer.
    agent = create_react_agent(llm, tools)

    user_query = (
        "What is the current weather in Ahmedabad and the current "
        "stock price of Apple (AAPL)? Give me both answers clearly."
    )

    try:
        result = agent.invoke({"messages": [("user", user_query)]})
        # The final assistant message is the agent's answer after reasoning
        # through however many tool calls it needed.
        final_message = result["messages"][-1]
        logging.info("Agent final answer:\n%s", final_message.content)
    except Exception as e:
        logging.error("Agent invocation failed: %s", e)


# =========================================================================
# MAIN — run all three tasks together (as requested: combine into one file)
# =========================================================================
def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point. Select which task to run via `--task`.

    Examples:
        python week-5/tool_agent.py --task task1
        python week-5/tool_agent.py --task all
    """
    parser = argparse.ArgumentParser(description="Week-5 LangChain tool agent demo")
    parser.add_argument("--task", choices=["task1", "task2", "task3", "all"], default="all")
    parser.add_argument("--provider", default=MODEL_PROVIDER, help="LLM provider to use")
    args = parser.parse_args(argv)

    # configure logging for nicer output
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # allow runtime override of provider
    # global MODEL_PROVIDER
    # MODEL_PROVIDER = args.provider

    # logging.info("Running Week-5 submission with provider: %s", MODEL_PROVIDER)

    if args.task in ("task1", "all"):
        task1_prebuilt_tool_demo()
    if args.task in ("task2", "all"):
        task2_custom_tools_demo()
    if args.task in ("task3", "all"):
        task3_react_agent_demo()


if __name__ == "__main__":
    main()