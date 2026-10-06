import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


# ===========================================================================
# SUBGRAPH 1 — "Research team"  (PATTERN 1: shared state keys)
# ===========================================================================
class ResearchState(TypedDict):
    topic: str                                  # SHARED with parent
    notes: Annotated[list[str], operator.add]   # SHARED with parent
    queries: list[str]                          # PRIVATE to this subgraph


def make_queries(state: ResearchState):
    print("    (research) make_queries")
    t = state["topic"]
    return {"queries": [f"what is {t}", f"{t} use cases", f"{t} pitfalls"]}


def run_searches(state: ResearchState):
    print("    (research) run_searches")
    # In real life: call a search tool / LLM per query.

    return {"notes": [f"note about '{q}'" for q in state["queries"]]}


def build_research_subgraph():
    b = StateGraph(ResearchState)
    b.add_node("make_queries", make_queries)
    b.add_node("run_searches", run_searches)
    b.add_edge(START, "make_queries")
    b.add_edge("make_queries", "run_searches")
    b.add_edge("run_searches", END)
    return b.compile()  # NOTE: no checkpointer here — the parent supplies it


# ===========================================================================
# SUBGRAPH 2 — "Review team"  (PATTERN 2: DIFFERENT schema, needs a wrapper)
# ===========================================================================
class ReviewState(TypedDict):
    text: str       # NOT a parent key — parent calls it "report"
    issues: list[str]
    verdict: str


def find_issues(state: ReviewState):
    print("    (review) find_issues")
    issues = [] if len(state["text"]) > 40 else ["too short"]
    return {"issues": issues}


def decide(state: ReviewState):
    print("    (review) decide")
    return {"verdict": "PASS" if not state["issues"] else "NEEDS WORK"}


def build_review_subgraph():
    b = StateGraph(ReviewState)
    b.add_node("find_issues", find_issues)
    b.add_node("decide", decide)
    b.add_edge(START, "find_issues")
    b.add_edge("find_issues", "decide")
    b.add_edge("decide", END)
    return b.compile()


review_subgraph = build_review_subgraph()


# ===========================================================================
# PARENT GRAPH
# ===========================================================================
class ParentState(TypedDict):
    topic: str
    notes: Annotated[list[str], operator.add]
    report: str
    verdict: str


def planner(state: ParentState):
    print("  [parent] planner")
    return {"topic": state["topic"].strip().lower()}


def writer(state: ParentState):
    print("  [parent] writer")
    body = "; ".join(state["notes"])
    return {"report": f"Report on {state['topic']}: {body}"}


# ---- PATTERN 2 wrapper: translate between parent and subgraph schemas ------
def review_node(state: ParentState):
    print("  [parent] review_node (wrapper)")
    sub_result = review_subgraph.invoke(
        {"text": state["report"], "issues": [], "verdict": ""}  # parent -> sub
    )
    return {"verdict": sub_result["verdict"]}                    # sub -> parent


def build_parent():
    research_subgraph = build_research_subgraph()

    b = StateGraph(ParentState)
    b.add_node("planner", planner)
    b.add_node("research", research_subgraph)  # PATTERN 1: compiled graph as a node
    b.add_node("writer", writer)
    b.add_node("review", review_node)          # PATTERN 2: wrapper function

    b.add_edge(START, "planner")
    b.add_edge("planner", "research")
    b.add_edge("research", "writer")
    b.add_edge("writer", "review")
    b.add_edge("review", END)

    return b.compile(checkpointer=InMemorySaver())  # parent-only checkpointer


# ===========================================================================
# RUN
# ===========================================================================
if __name__ == "__main__":
    graph = build_parent()
    config = {"configurable": {"thread_id": "pipeline-1"}}

    # --- 1) Normal run -----------------------------------------------------
    print("\n=== Run ===")
    out = graph.invoke({"topic": "  LangGraph  ", "notes": [], "report": "", "verdict": ""}, config)
    print("\nverdict:", out["verdict"])
    print("report :", out["report"])
    # Note: "queries" (private to the research subgraph) is NOT in the parent output.
    print("parent keys:", list(out.keys()))

    # xray=True expands subgraphs so you can see their internals (Mermaid text;
    # paste it into https://mermaid.live to view).
    print("\n=== Mermaid diagram (xray) ===")
    print(graph.get_graph(xray=True).draw_mermaid())

