import operator
import sys
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# ---------------------------------------------------------------------------
# 1) STATE
# ---------------------------------------------------------------------------
class State(TypedDict):
    request: str
    amount: float
    status: str
    reason: str
    log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2) NODES
# ---------------------------------------------------------------------------
def analyze_request(state: State):
    """Pretend an LLM parsed the customer message and decided on an amount."""
    print("  [node] analyze_request")
    return {
        "amount": 120.0,
        "status": "pending_approval",
        "log": [f"analyzed: '{state['request']}' -> refund 120.0"],
    }


def approval_gate(state: State) -> Command[Literal["execute_refund", "cancel_refund"]]:
    """
    THE HUMAN-IN-THE-LOOP STEP.

    interrupt() does two things:
      - First time: STOPS the graph and hands the dict below to the caller.
      - After resume: RETURNS whatever the human sent via Command(resume=...).
    """
    print("  [node] approval_gate (about to ask human)")

    decision = interrupt(
        {
            "question": "Approve this refund?",
            "request": state["request"],
            "amount": state["amount"],
            "options": "approved: bool, amount: float (optional override)",
        }
    )
    print("  [node] approval_gate resumed with:", decision)
    # ---- Code below only runs AFTER the human has answered -----------------
    reason = decision.get("reason", state.get("reason", ""))
    if decision.get("approved"):
        # Human may also EDIT the amount — "approve with changes" pattern.
        new_amount = decision.get("amount", state["amount"])
        return Command(
            update={
                "amount": new_amount,
                "status": "approved",
                "log": [f"human approved amount={new_amount}"],
                "reason": reason,
            },
            goto="execute_refund",  # route to the next node
        )

    return Command(
        update={"status": "rejected", "log": ["human rejected"], "reason": reason},
        goto="cancel_refund",
    )


def execute_refund(state: State):
    # The critical action — only reachable through an approval.
    print(f"  [node] execute_refund  --> REFUNDING {state['amount']}")
    return {"status": "refunded", "log": [f"refund of {state['amount']} executed"], "reason": state.get("reason", "")}


def cancel_refund(state: State):
    print("  [node] cancel_refund")
    return {"status": "cancelled", "log": ["no refund issued"], "reason": state.get("reason", "")}


# ---------------------------------------------------------------------------
# 3) BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph():
    b = StateGraph(State)
    b.add_node("analyze_request", analyze_request)
    b.add_node("approval_gate", approval_gate)
    b.add_node("execute_refund", execute_refund)
    b.add_node("cancel_refund", cancel_refund)

    b.add_edge(START, "analyze_request")
    b.add_edge("analyze_request", "approval_gate")
    # No edges out of approval_gate needed: it routes itself via Command(goto=...)
    b.add_edge("execute_refund", END)
    b.add_edge("cancel_refund", END)

    return b.compile(checkpointer=InMemorySaver())  # checkpointer is REQUIRED


# ---------------------------------------------------------------------------
# 4) HELPER: run until pause, show the question
# ---------------------------------------------------------------------------
def start_run(graph, thread_id: str, request: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"request": request, "amount": 0.0, "status": "", "log": [], "reason": ""}, config
    )

    # When paused, the result contains "__interrupt__" with the payload we sent.
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("  >>> PAUSED. Human is asked:", payload)
    return config, result


# ---------------------------------------------------------------------------
# 5) DEMOS
# ---------------------------------------------------------------------------
def scripted_demo():
    graph = build_graph()

    # print("\n=== Scenario A: human APPROVES ===")
    # cfg, _ = start_run(graph, "ticket-A", "Item arrived broken")
    # print("  state while paused -> next:", graph.get_state(cfg).next)
    # final = graph.invoke(Command(resume={"approved": True}), cfg)
    # print("  FINAL:", final["status"], final["log"])

    # print("\n=== Scenario B: human REJECTS ===")
    # cfg, _ = start_run(graph, "ticket-B", "Changed my mind")
    # final = graph.invoke(Command(resume={"approved": False}), cfg)
    # print("  FINAL:", final["status"], final["log"])

    print("\n=== Scenario C: human APPROVES WITH EDIT (amount 120 -> 50) ===")
    cfg, _ = start_run(graph, "ticket-C", "Partial damage")
    final = graph.invoke(Command(resume={"approved": True, "amount": 50.0}, update={"log": ["edited amount"]}), cfg)
    print("  FINAL:", final["status"], final["log"], "Reason:", final.get("reason", ""))


def interactive_demo():
    graph = build_graph()
    cfg, _ = start_run(graph, "ticket-live", "Item arrived broken")
    answer = input("\nApprove refund? [y/n/amount]: ").strip().lower()
    reason = input("\nReason for decision: ").strip().upper()

    if answer == "y":
        resume = {"approved": True}
    elif answer == "n":
        resume = {"approved": False}
    else:
        resume = {"approved": True, "amount": float(answer)}

    final = graph.invoke(Command(resume={**resume, "reason": reason}), cfg)
    print("FINAL:", final["status"], final["log"], "Reason:", final.get("reason", ""))


if __name__ == "__main__":
    interactive_demo() if "interactive" in sys.argv else scripted_demo()

