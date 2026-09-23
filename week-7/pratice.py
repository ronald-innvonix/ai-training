"""
LangGraph Fundamentals — one annotated example
================================================
Run with:  python langgraph_fundamentals_example.py

Covers everything from the practice plan, in one graph:
  1. State management      -> the State TypedDict + reducers
  2. Nodes and edges        -> plain functions + add_node / add_edge
  3. Conditional routing    -> add_conditional_edges
  4. Retries                -> a self-looping conditional edge (a cycle)
  5. Parallel execution     -> dynamic fan-out with Send(), then fan-in
  6. Multi-agent collab.    -> three "specialist" agents merged by an
                               aggregator node

No API key needed. The "agents" are simulated with plain functions so you
can run this immediately and watch the mechanics. Once it makes sense,
swap simulate_llm_call() for a real chat model call.
"""

import operator
from typing import Annotated, List, Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ---------------------------------------------------------------------------
# STEP 1: STATE MANAGEMENT
# ---------------------------------------------------------------------------
# State is just a typed dict shared across every node. Each node receives
# the *current* state and returns a *partial* dict of updates — LangGraph
# merges it in for you; you never mutate state directly.
#
# For fields that get written from MULTIPLE places (e.g. every node logs
# something, and 3 parallel agents each write a result), you need a
# *reducer* via Annotated[type, fn]. The fn tells LangGraph how to combine
# the old value with the new one instead of overwriting it.
# operator.add on lists means "concatenate" (so results accumulate instead
# of the last writer wiping out the others).
#
# Fields WITHOUT Annotated (route, retry_count, final_answer) are simply
# overwritten by whichever node last returns them.
class State(TypedDict):
    query: str                                          # the user's input
    log: Annotated[List[str], operator.add]              # accumulates across nodes
    route: str                                           # decided by classify()
    retry_count: int                                     # used by the retry loop
    final_answer: str                                    # the end result
    agent_results: Annotated[List[str], operator.add]     # filled in parallel
    summary: str                                          # set by the aggregator


# ---------------------------------------------------------------------------
# STEP 2: NODES AND EDGES
# ---------------------------------------------------------------------------
# A node is any function State -> dict. How they CONNECT (add_edge /
# add_conditional_edges) is set up later in build_graph() — a node itself
# doesn't know what runs before or after it.

def intake(state: State) -> dict:
    """Entry node. Initializes bookkeeping fields."""
    return {
        "log": [f"intake: received query = {state['query']!r}"],
        "retry_count": 0,
    }


def classify(state: State) -> dict:
    """
    Decides which path the query should take. This node does NOT branch —
    it only writes `route` into state. The actual branching happens in the
    conditional edge (route_after_classify) that reads it next.
    """
    query = state["query"].lower()
    route = "multi_agent" if ("team" in query or "compare" in query) else "single_agent"
    return {"route": route, "log": [f"classify: chose route = {route}"]}


# ---------------------------------------------------------------------------
# STEP 3: CONDITIONAL ROUTING  +  STEP 5: PARALLEL EXECUTION (fan-out)
# ---------------------------------------------------------------------------
# This one router function demonstrates both:
#   - plain conditional routing: return a node-name string
#   - dynamic parallel fan-out: return a LIST of Send objects instead
#
# Each Send("agent_worker", {...}) spawns an independent, concurrent
# execution of `agent_worker` with its own small input dict. LangGraph
# runs all of them in parallel and waits for every one to finish before
# continuing to whatever node they all connect to (the fan-in step below).
AGENT_ROLES = ["researcher", "critic", "summarizer"]

def route_after_classify(state: State):
    if state["route"] == "single_agent":
        return "call_single_agent"                       # -> one node
    return [                                              # -> N parallel nodes
        Send("agent_worker", {"query": state["query"], "role": role})
        for role in AGENT_ROLES
    ]


# ---------------------------------------------------------------------------
# STEP 4: RETRIES
# ---------------------------------------------------------------------------
# call_single_agent simulates a flaky call (e.g. an LLM request that
# sometimes fails). should_retry is a conditional edge that inspects
# retry_count: if the call failed and we haven't hit MAX_RETRIES, it routes
# BACK to call_single_agent itself — a cycle. Otherwise it moves on.
MAX_RETRIES = 2

def call_single_agent(state: State) -> dict:
    attempt = state["retry_count"] + 1
    # Deterministic "failure" so the retry loop is easy to observe: fails
    # on attempt 1 only, then succeeds. Swap this for a real API call +
    # try/except in a production graph.
    succeeded = attempt > 1
    log_line = f"call_single_agent: attempt {attempt} " + ("succeeded" if succeeded else "FAILED")
    update = {"retry_count": attempt, "log": [log_line]}
    if succeeded:
        update["final_answer"] = f"Answer for {state['query']!r} (after {attempt} attempt(s))"
    return update


def should_retry(state: State) -> Literal["retry", "give_up", "done"]:
    if state.get("final_answer"):
        return "done"
    if state["retry_count"] < MAX_RETRIES:
        return "retry"
    return "give_up"


def fallback(state: State) -> dict:
    return {
        "final_answer": "Could not get an answer after retries — fallback response.",
        "log": ["fallback: giving up and returning a default answer"],
    }


# ---------------------------------------------------------------------------
# STEP 6: MULTI-AGENT COLLABORATION
# ---------------------------------------------------------------------------
# Each parallel branch runs this SAME node but with a different `role`,
# simulating specialized agents that each handle one part of the task. In a
# real app, swap simulate_llm_call() for an actual chat model call using a
# role-specific system prompt.
def simulate_llm_call(role: str, query: str) -> str:
    responses = {
        "researcher": f"[researcher] key facts about: {query}",
        "critic": f"[critic] potential issues with: {query}",
        "summarizer": f"[summarizer] tl;dr of: {query}",
    }
    return responses.get(role, f"[{role}] generic response to: {query}")


def agent_worker(state: dict) -> dict:
    # NOTE: this node's *input* is the small dict passed via Send(), not
    # the full graph State — but its *return* still merges into the shared
    # State using the reducers defined above. agent_results uses
    # operator.add, so each worker's single-item list gets APPENDED rather
    # than overwriting the other workers' results.
    role = state["role"]
    result = simulate_llm_call(role, state["query"])
    return {
        "agent_results": [result],
        "log": [f"agent_worker({role}): produced a result"],
    }


def aggregate(state: State) -> dict:
    """Fan-in node: only runs once ALL parallel agent_worker branches finish."""
    combined = " | ".join(state["agent_results"])
    return {
        "summary": combined,
        "final_answer": f"Combined answer: {combined}",
        "log": ["aggregate: merged all agent results"],
    }


# ---------------------------------------------------------------------------
# BUILD THE GRAPH — wiring every node above together
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(State)

    graph.add_node("intake", intake)
    graph.add_node("classify", classify)
    graph.add_node("call_single_agent", call_single_agent)
    graph.add_node("fallback", fallback)
    graph.add_node("agent_worker", agent_worker)
    graph.add_node("aggregate", aggregate)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "classify")

    # CONDITIONAL ROUTING + PARALLEL FAN-OUT in one edge: the router
    # function above returns either a node name (single_agent path) or a
    # list of Send objects (multi_agent path -> 3 parallel agent_worker runs)
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        ["call_single_agent", "agent_worker"],   # possible destinations, for graph drawing
    )

    # RETRY LOOP: call_single_agent points back to itself on failure
    graph.add_conditional_edges(
        "call_single_agent",
        should_retry,
        {"retry": "call_single_agent", "give_up": "fallback", "done": END},
    )
    graph.add_edge("fallback", END)

    # FAN-IN: LangGraph waits for every parallel agent_worker Send to finish
    # before running aggregate, since they all point to the same next node.
    graph.add_edge("agent_worker", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# RUN IT — two example invocations, one per path
# ---------------------------------------------------------------------------
def run_example(app, query: str, label: str):
    print(f"\n{'=' * 70}\n{label}: {query!r}\n{'=' * 70}")
    result = app.invoke({
        "query": query,
        "log": [],
        "route": "",
        "retry_count": 0,
        "final_answer": "",
        "agent_results": [],
        "summary": "",
    })
    print("\n-- execution log (order nodes ran in) --")
    for line in result["log"]:
        print(" ", line)
    print("\n-- final answer --")
    print(" ", result["final_answer"])


if __name__ == "__main__":
    app = build_graph()

    # Optional: uncomment to see the graph structure as text
    # print(app.get_graph().draw_ascii())

    # Triggers: single_agent path -> retry loop (fails once, then succeeds)
    run_example(app, "What is the capital of France?", "SINGLE-AGENT PATH (with retry)")

    # Triggers: multi_agent path -> parallel fan-out to 3 agents -> aggregate
    run_example(app, "Compare our team's options for a database", "MULTI-AGENT PATH (parallel)")