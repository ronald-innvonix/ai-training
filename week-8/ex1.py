import operator
import sys
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------------
# 1) STATE
# ---------------------------------------------------------------------------
# `Annotated[list[str], operator.add]` is a REDUCER: when a node returns
# {"log": ["x"]}, LangGraph APPENDS it to the existing list instead of
# overwriting it. Without a reducer, the latest value simply replaces the old one.
class State(TypedDict):
    log: Annotated[list[str], operator.add]
    data: str


# ---------------------------------------------------------------------------
# 2) NODES — plain functions: take state, return a partial state update
# ---------------------------------------------------------------------------
def fetch_data(state: State):
    print("  [node] fetch_data running")
    return {"data": "raw numbers", "log": ["fetched"]}



def process_data(state: State):
    print("  [node] process_data running")
    return {"data": state["data"].upper(), "log": ["processed"]}


def publish_report(state: State):
    # Imagine this is the expensive / irreversible step we want to gate.
    print("  [node] publish_report running  <-- the critical step")
    return {"log": [f"published: {state['data']}"]}


# ---------------------------------------------------------------------------
# 3) BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("process_data", process_data)
    builder.add_node("publish_report", publish_report)

    builder.add_edge(START, "fetch_data")
    builder.add_edge("fetch_data", "process_data")
    builder.add_edge("process_data", "publish_report")
    builder.add_edge("publish_report", END)

    # compile() with a checkpointer = persistence ON.
    # interrupt_before = static breakpoint: stop BEFORE publish_report executes.
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["publish_report"],
    )


# ---------------------------------------------------------------------------
# PART A — In-memory demo (everything in one process)
# ---------------------------------------------------------------------------
def demo_in_memory():
    graph = build_graph(InMemorySaver())

    # The config tells LangGraph WHICH thread's checkpoints to use.
    config = {"configurable": {"thread_id": "run-1"}}

    print("\n--- Step 1: first run (should pause before publish_report) ---")
    graph.invoke({"log": [], "data": ""}, config)

    # get_state() returns the latest checkpoint (a StateSnapshot).
    snap = graph.get_state(config)
    print("  values :", snap.values)
    print("  next   :", snap.next, " <-- nodes still waiting to run")

    # --- Optional: a human edits the state while the graph is paused --------
    # update_state() writes a NEW checkpoint, as if a node had produced it.
    print("\n--- Step 2: human tweaks the state while paused ---")
    graph.update_state(config, {"log": ["reviewed by human"]})
    print("  values :", graph.get_state(config).values)

    print("\n--- Step 3: resume (input=None means 'continue where you stopped') ---")
    final = graph.invoke(None, config)
    print("  final  :", final)

    # --- Inspect every checkpoint ever saved for this thread ---------------
    print("\n--- Step 4: checkpoint history (newest first) ---")
    history = list(graph.get_state_history(config))
    for s in history:
        print(f"  next={s.next!s:22} log={s.values.get('log')}")

    # --- TIME TRAVEL: replay from an older checkpoint ----------------------
    # Find the checkpoint taken right before process_data ran and re-run from it.
    print("\n--- Step 5: time travel — replay from before process_data ---")
    target = next(s for s in history if s.next == ("process_data",))
    graph.invoke(None, target.config)  # target.config pins the checkpoint_id
    

    # --- A different thread_id has completely separate state ---------------
    print("\n--- Step 6: new thread = fresh, independent run ---")
    graph.invoke({"log": [], "data": ""}, {"configurable": {"thread_id": "run-2"}})
    print("  run-2 next:", graph.get_state({"configurable": {"thread_id": "run-2"}}).next)


# ---------------------------------------------------------------------------
# PART B — SQLite: survives process restarts (the REAL point of persistence)
# ---------------------------------------------------------------------------
# InMemorySaver dies with the process. SqliteSaver writes to a file, so you can
# stop the script entirely and resume it later from a different process.
DB_PATH = "checkpoints.db"
SQLITE_CONFIG = {"configurable": {"thread_id": "persistent-run"}}


def sqlite_start():
    with SqliteSaver.from_conn_string(DB_PATH) as saver:
        graph = build_graph(saver)
        graph.invoke({"log": [], "data": ""}, SQLITE_CONFIG)
        print("Paused. State saved to", DB_PATH)
        print("Next nodes:", graph.get_state(SQLITE_CONFIG).next)
        print("Now run: resume")


def sqlite_resume():
    with SqliteSaver.from_conn_string(DB_PATH) as saver:
        graph = build_graph(saver)  # brand-new process, brand-new graph object
        print("Loaded state:", graph.get_state(SQLITE_CONFIG).values)
        final = graph.invoke({"log": ["test"], "data": "boom"}, SQLITE_CONFIG)
        print("Final:", final)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "memory"
    {"memory": demo_in_memory, "start": sqlite_start, "resume": sqlite_resume}[mode]()

