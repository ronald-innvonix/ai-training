"""Practical 2 - Model Invocation Methods.

For each configured provider (Groq, Mistral, Google) we exercise the four
invocation methods -- ``invoke``, ``stream``, ``ainvoke`` and ``astream`` --
and time them.  Results are printed as a comparison table, saved to
``outputs/comparison_results.json``, and summarised in a markdown report.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AsyncIterator, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import OUTPUTS_DIR, get_settings
from .models import build_configured_models

# A fixed, representative prompt so latency comparisons are measured against
# the same workload for every provider and method.
PROMPT_TEXT = "Describe a smart water bottle in exactly two sentences."


@dataclass
class MethodResult:
    """Timing/output metrics captured for one (provider, method) run."""

    provider: str
    model: str
    method: str
    total_latency_s: float
    first_token_latency_s: float | None  # only populated for streaming methods
    output_length_chars: int


def _messages(prompt_text: str = PROMPT_TEXT) -> list[SystemMessage | HumanMessage]:
    """Build the message list sent to every model."""
    return [
        SystemMessage(content="You are a terse, accurate product copywriter."),
        HumanMessage(content=prompt_text),
    ]


def _measure_invoke(model: BaseChatModel) -> MethodResult:
    """Time ``model.invoke(...)`` (blocking, single response)."""
    start = time.perf_counter()
    response: AIMessage = model.invoke(_messages())
    elapsed = time.perf_counter() - start
    return MethodResult(
        provider=model.__class__.__name__,
        model=getattr(model, "model_name", None) or getattr(model, "model", "?"),
        method="invoke",
        total_latency_s=elapsed,
        first_token_latency_s=None,
        output_length_chars=len(response.content),
    )


def _measure_stream(model: BaseChatModel) -> MethodResult:
    """Time ``model.stream(...)``, printing tokens as they arrive."""
    chunks: Iterator[AIMessage] = model.stream(_messages())
    started = time.perf_counter()
    first_token = None
    collected: list[str] = []
    printed_chars = 0
    for chunk in chunks:
        if first_token is None:
            first_token = time.perf_counter() - started
        token = chunk.content
        collected.append(token)
        # Print each token chunk as it arrives (streaming behaviour).
        if token:
            print(token, end="", flush=True)
            printed_chars += len(token)
    elapsed = time.perf_counter() - started
    print()  # newline after streamed output
    return MethodResult(
        provider=model.__class__.__name__,
        model=getattr(model, "model_name", None) or getattr(model, "model", "?"),
        method="stream",
        total_latency_s=elapsed,
        first_token_latency_s=first_token,
        output_length_chars=sum(len(c) for c in collected),
    )


async def _measure_ainvoke(model: BaseChatModel) -> MethodResult:
    """Time ``await model.ainvoke(...)`` (async, single response)."""
    start = time.perf_counter()
    response: AIMessage = await model.ainvoke(_messages())
    elapsed = time.perf_counter() - start
    return MethodResult(
        provider=model.__class__.__name__,
        model=getattr(model, "model_name", None) or getattr(model, "model", "?"),
        method="ainvoke",
        total_latency_s=elapsed,
        first_token_latency_s=None,
        output_length_chars=len(response.content),
    )


async def _measure_astream(model: BaseChatModel) -> MethodResult:
    """Time ``model.astream(...)``, printing tokens as they arrive."""
    chunks: AsyncIterator[AIMessage] = model.astream(_messages())
    started = time.perf_counter()
    first_token = None
    collected: list[str] = []
    async for chunk in chunks:
        if first_token is None:
            first_token = time.perf_counter() - started
        token = chunk.content
        if token:
            print(token, end="", flush=True)
            collected.append(token)
    elapsed = time.perf_counter() - started
    print()  # newline when done
    return MethodResult(
        provider=model.__class__.__name__,
        model=getattr(model, "model_name", None) or getattr(model, "model", "?"),
        method="astream",
        total_latency_s=elapsed,
        first_token_latency_s=first_token,
        output_length_chars=sum(len(c) for c in collected),
    )


async def _run_async_methods(model: BaseChatModel) -> list[MethodResult]:
    """Run the two async methods for one model sequentially."""
    return [await _measure_ainvoke(model), await _measure_astream(model)]


def _format_table(rows: list[MethodResult]) -> str:
    """Render the raw metric rows as a simple fixed-width text table."""
    header = f"{'Provider':<18}{'Method':<10}{'Total (s)':<12}{'First-token (s)':<18}{'Chars':<8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        first = f"{r.first_token_latency_s:.3f}" if r.first_token_latency_s is not None else "-"
        lines.append(
            f"{r.provider:<18}{r.method:<10}{r.total_latency_s:<12.3f}{first:<18}{r.output_length_chars:<8}"
        )
    return "\n".join(lines)


def _build_summary(rows: list[MethodResult]) -> str:
    """Produce a human-readable comparison / findings text from the metrics."""
    lines: list[str] = []
    lines.append("# Latency Comparison Across Providers\n")

    # Group methods for cross-provider view.
    methods = ["invoke", "stream", "ainvoke", "astream"]
    lines.append("## Per-method average latency (across providers)")
    lines.append(f"{'Method':<10}{'Avg total (s)':<16}{'Avg first-token (s)':<20}{'Avg chars':<10}")
    avg_by_method: dict[str, list[MethodResult]] = {}
    for r in rows:
        avg_by_method.setdefault(r.method, []).append(r)
    for m in methods:
        group = avg_by_method.get(m, [])
        if not group:
            continue
        total = sum(r.total_latency_s for r in group) / len(group)
        firsts = [r.first_token_latency_s for r in group if r.first_token_latency_s is not None]
        avg_first = sum(firsts) / len(firsts) if firsts else float("nan")
        chars = sum(r.output_length_chars for r in group) / len(group)
        f_str = f"{avg_first:.3f}" if firsts else "-"
        lines.append(f"{m:<10}{total:<16.3f}{f_str:<20}{chars:<10.1f}")

    lines.append("\n## Provider ranking by average total latency")
    providers: dict[str, list[MethodResult]] = {}
    for r in rows:
        providers.setdefault(r.provider, []).append(r)
    ranking = sorted(
        ((p, sum(x.total_latency_s for x in g) / len(g)) for p, g in providers.items()),
        key=lambda item: item[1],
    )
    for provider, avg in ranking:
        lines.append(f"- {provider}: {avg:.3f} s (avg)")

    lines.append("\n## Findings")
    lines.append(
        "Streaming methods (stream / astream) return their *first token* earlier "
        "than the total latency of invoke/ainvoke, so they feel more responsive "
        "even when total completion time is similar."
    )
    if len(rows) >= 2:
        # Compare a matching sync/async pair if we have both for any provider.
        pairs = [r for r in rows if r.method in ("invoke", "ainvoke")]
        sync = next((r for r in pairs if r.method == "invoke"), None)
        a_sync = next((r for r in pairs if r.method == "ainvoke"), None)
        if sync and a_sync:
            ratio = a_sync.total_latency_s / sync.total_latency_s if sync.total_latency_s else float("inf")
            lines.append(
                f"Async vs sync single-call latency: ainvoke/invoke ratio = {ratio:.2f}x "
                "(values close to 1x indicate async has ~no per-call advantage for a "
                "single request; its real benefit shows under concurrent workloads)."
            )

    lines.append(
        "\nCaveat: measured on one request per method against cloud APIs; latency "
        "varies with provider load, region and time of day."
    )
    return "\n".join(lines)


def run() -> None:
    """Entry point for practical 2."""
    print("#" * 60)
    print("# Practical 2 - Model Invocation Methods")
    print("#" * 60)

    providers = build_configured_models()
    if not providers:
        print(
            "No providers configured (no API keys in .env). "
            "Add at least one key to GROQ_API_KEY / MISTRAL_API_KEY / GOOGLE_API_KEY."
        )
        return

    all_rows: list[MethodResult] = []
    for name, model in providers:
        print(f"\n=== {name} ({model.__class__.__name__}) ===")
        rows: list[MethodResult] = []
        rows.append(_measure_invoke(model))
        rows.append(_measure_stream(model))
        rows.extend(asyncio.run(_run_async_methods(model)))
        all_rows.extend(rows)

    table = _format_table(all_rows)
    print("\n" + table)

    summary = _build_summary(all_rows)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "comparison_results.json").write_text(
        json.dumps([asdict(r) for r in all_rows], indent=2), encoding="utf-8"
    )
    (OUTPUTS_DIR / "latency_summary.md").write_text(summary, encoding="utf-8")

    print(f"\nSaved metrics -> {OUTPUTS_DIR / 'comparison_results.json'}")
    print(f"Saved summary  -> {OUTPUTS_DIR / 'latency_summary.md'}")


if __name__ == "__main__":
    run()