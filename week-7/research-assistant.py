import json
import os
import operator
import sys
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq          # <-- replaces init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

# Any model from https://console.groq.com/docs/models works here.
MODEL = os.environ.get("RESEARCH_ASSISTANT_MODEL", "openai/gpt-oss-120b")
llm = ChatGroq(
    model=MODEL,
    temperature=0,
    max_retries=4,   # research branches run in parallel, so retry on 429 rate limits
)

MAX_PLAN_RETRIES = 2
RESULTS_PER_QUESTION = 4


# ---------------------------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------------------------
# One field worth calling out: `research` is written by MULTIPLE parallel
# branches (one per sub-question), so it MUST have a reducer (operator.add
# = append) or LangGraph will raise a conflicting-concurrent-write error
# when two branches try to update it in the same step. `sub_questions`,
# `final_answer` etc. are only ever written by a single node, so they
# don't need one — plain overwrite is fine.
class ResearchItem(TypedDict):
    question: str
    sources: List[str]
    summary: str


class State(TypedDict):
    query: str
    sub_questions: List[str]
    plan_retry_count: int
    research: Annotated[List[ResearchItem], operator.add]
    final_answer: str
    log: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# AGENT 1 — PLANNER
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You are a research planner. Break the user's question into 2-4 focused, \
independently-searchable sub-questions that together would let someone fully answer it.

Respond with ONLY a JSON array of strings — nothing else. Example:
["sub-question one", "sub-question two"]

User's question: {query}
"""

def planner(state: State) -> dict:
    attempt = state.get("plan_retry_count", 0) + 1
    raw = llm.invoke(PLANNER_PROMPT.format(query=state["query"])).content.strip()
    # Models sometimes wrap JSON in a code fence even when told not to — strip it.
    for fence in ("```json", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
    raw = raw.strip().removesuffix("```").strip()

    try:
        sub_questions = json.loads(raw)
        assert isinstance(sub_questions, list) and sub_questions
        assert all(isinstance(q, str) for q in sub_questions)
    except Exception:
        # Parsing failed. Leave sub_questions unset so should_replan can
        # decide whether to retry or fall back.
        return {
            "plan_retry_count": attempt,
            "log": [f"planner: attempt {attempt} failed to produce valid JSON"],
        }

    return {
        "sub_questions": sub_questions,
        "plan_retry_count": attempt,
        "log": [f"planner: attempt {attempt} -> {len(sub_questions)} sub-questions: {sub_questions}"],
    }


def should_replan(state: State):
    """
    CONDITIONAL ROUTING + PARALLEL FAN-OUT in one router function:
      - valid plan                    -> fan out one research branch per sub-question
      - invalid plan, retries left     -> loop back to planner (a cycle)
      - invalid plan, out of retries   -> fall back to researching the raw query directly
    """
    if state.get("sub_questions"):
        return [Send("research_branch", {"question": q}) for q in state["sub_questions"]]
    if state["plan_retry_count"] < MAX_PLAN_RETRIES:
        return "planner"
    return [Send("research_branch", {"question": state["query"]})]


# ---------------------------------------------------------------------------
# AGENT 2 — RETRIEVAL   (no LLM call: pure web search)
# ---------------------------------------------------------------------------
def retrieval_agent(question: str, max_results: int = RESULTS_PER_QUESTION) -> List[str]:
    """Real web search via DuckDuckGo — no API key required. Falls back to a
    placeholder if the library isn't installed or the network call fails, so
    the graph still runs end-to-end while you're setting things up."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(question, max_results=max_results))
        return [f"{h['title']}: {h['body']}" for h in hits] or ["No results found."]
    except Exception as e:
        return [f"(search unavailable — {e}) Answering from general knowledge instead."]


# ---------------------------------------------------------------------------
# AGENT 3 — SUMMARIZER   (LLM call, grounded in the retrieved sources)
# ---------------------------------------------------------------------------
SUMMARIZER_PROMPT = """Using ONLY the sources below, write a 2-4 sentence answer to the question. \
If the sources don't contain enough information, say what's missing.

Question: {question}

Sources:
{sources}
"""

def summarizer_agent(question: str, sources: List[str]) -> str:
    prompt = SUMMARIZER_PROMPT.format(question=question, sources="\n".join(f"- {s}" for s in sources))
    return llm.invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# retrieval_agent + summarizer_agent are combined into ONE graph node
# ---------------------------------------------------------------------------
# Why combine them, if they're supposed to be "separate agents"? They ARE
# separate — separate functions, separate responsibilities, one does search
# and one does an LLM call. What they can't be is separate GRAPH NODES
# connected by a fixed edge in this design, because each parallel branch
# would then need to pass its own private (question, sources) pair through
# *shared* graph state to get from one node to the next — and two branches
# writing to the same unannotated field in the same step is exactly the
# conflict reducers exist to prevent. Keeping the two calls sequential
# inside one node keeps each branch's intermediate data private to itself,
# and the node still only writes to `research`, which IS a reducer field.
def research_branch(branch_input: dict) -> dict:
    # branch_input is just {"question": ...} — the payload from Send().
    question = branch_input["question"]
    sources = retrieval_agent(question)
    summary = summarizer_agent(question, sources)
    return {
        "research": [{"question": question, "sources": sources, "summary": summary}],
        "log": [f"research_branch: completed {question!r}"],
    }


# ---------------------------------------------------------------------------
# AGENT 4 — FINAL ANSWER   (fan-in: runs once every branch has finished)
# ---------------------------------------------------------------------------
FINAL_ANSWER_PROMPT = """You are writing the final answer to a user's research question, using \
research notes gathered by your team. Synthesize them into one clear, well-organized answer. \
Mention which sub-questions were covered.

Original question: {query}

Research notes:
{notes}
"""

def final_answer_agent(state: State) -> dict:
    notes = "\n\n".join(
        f"Sub-question: {item['question']}\nFindings: {item['summary']}"
        for item in state["research"]
    )
    answer = llm.invoke(FINAL_ANSWER_PROMPT.format(query=state["query"], notes=notes)).content.strip()
    return {
        "final_answer": answer,
        "log": ["final_answer_agent: synthesized final answer from all research branches"],
    }


# ---------------------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(State)

    graph.add_node("planner", planner)
    graph.add_node("research_branch", research_branch)
    graph.add_node("final_answer_agent", final_answer_agent)

    graph.add_edge(START, "planner")

    # conditional routing + dynamic parallel fan-out, from one router
    graph.add_conditional_edges(
        "planner",
        should_replan,
        ["planner", "research_branch"],   # possible destinations, for graph drawing
    )

    # fan-in: LangGraph waits for every research_branch (however many
    # sub-questions there were) to finish before running final_answer_agent
    graph.add_edge("research_branch", "final_answer_agent")
    graph.add_edge("final_answer_agent", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What are the main approaches to reducing urban traffic congestion?"

    app = build_graph()
    result = app.invoke(
        {
            "query": query,
            "sub_questions": [],
            "plan_retry_count": 0,
            "research": [],
            "final_answer": "",
            "log": [],
        },
        config={"recursion_limit": 50},
    )

    print(f"\n{'=' * 70}\nQUESTION: {query}\n{'=' * 70}")

    print("\n-- execution log --")
    for line in result["log"]:
        print(" ", line)

    print("\n-- per sub-question research --")
    for item in result["research"]:
        print(f"\n  Q: {item['question']}")
        print(f"  A: {item['summary']}")

    print(f"\n{'=' * 70}\nFINAL ANSWER\n{'=' * 70}")
    print(result["final_answer"])