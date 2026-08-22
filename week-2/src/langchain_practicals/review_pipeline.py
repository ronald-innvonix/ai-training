"""Practical 3 - Review Analysis Pipeline.

Loads 10 real human-written reviews from ``data/reviews.json`` and runs them
through a structured-output pipeline (``prompt | model | parser``) using
``PydanticOutputParser`` against a Pydantic v2 schema.  Results are saved to
``outputs/review_analysis_<provider>.json``.  Parse failures are retried with
``OutputFixingParser``.

The pipeline is exercised against every configured provider so the output
quality, behaviour and latency can be compared across providers.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from .config import OUTPUTS_DIR, REVIEWS_FILE
from .models import build_configured_models


# --------------------------------------------------------------------------
# 1. Pydantic schema for structured review output.
# --------------------------------------------------------------------------
class ReviewAnalysis(BaseModel):
    """Structured output produced for a single customer review."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review: positive, negative or neutral."
    )
    sentiment_score: float = Field(
        description="Sentiment magnitude on a scale from -1.0 (very negative) to +1.0 (very positive).",
    )
    key_issues: list[str] = Field(
        description="Short list of problems or complaints mentioned, empty if there are none."
    )
    key_positives: list[str] = Field(
        description="Short list of good points / praise mentioned, empty if there are none."
    )
    summary: str = Field(description="One-sentence factual summary of the review.")
    recommended_action: str = Field(
        description="Suggested action for the company based on this review."
    )


# --------------------------------------------------------------------------
# 2. Building the Runnable pipeline.
# --------------------------------------------------------------------------
def build_chains(model: BaseChatModel) -> tuple[ChatPromptTemplate, Runnable, Runnable, PydanticOutputParser]:
    """Create the prompt, the raw string chain and the parsed-output chain.

    Returns ``(prompt, raw_chain, parsed_chain, parser)`` where
    ``raw_chain = prompt | model | StrOutputParser`` (returns a ``str``) and
    ``parsed_chain = prompt | model | parser`` (returns ``ReviewAnalysis``).
    """
    parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a meticulous customer-experience analyst. Analyse the "
                "review below and return a strict JSON object matching this schema:\n"
                "{format_instructions}",
            ),
            ("human", "Product: {product}\n\nReview text:\n{review_text}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    raw_chain: Runnable = prompt | model | StrOutputParser()
    parsed_chain: Runnable = prompt | model | parser
    return prompt, raw_chain, parsed_chain, parser


def _fix_parse(
    raw: str,
    model: BaseChatModel,
    parser: PydanticOutputParser,
    error: OutputParserException | None = None,
) -> ReviewAnalysis:
    """Repair a malformed/model output by re-prompting the model.

    ``OutputFixingParser`` was removed in LangChain v1, so we implement the same
    idea as a tiny Runnable chain (``prompt | model | parser``): we resend the
    schema instructions, the invalid output and the parser error, then re-parse
    the corrected model output into a ``ReviewAnalysis``.
    """
    fix_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You output strict JSON matching a schema, nothing else.",
            ),
            (
                "human",
                "Schema instructions:\n{instructions}\n\n"
                "Your previous output was invalid:\n{invalid_output}\n\n"
                "Parser error:\n{error}\n\n"
                "Return ONLY corrected JSON matching the schema.",
            ),
        ]
    )
    fix_chain = fix_prompt | model | parser
    return fix_chain.invoke(
        {
            "instructions": parser.get_format_instructions(),
            "invalid_output": raw,
            "error": str(error or "unable to parse"),
        }
    )


# --------------------------------------------------------------------------
# 3. Running the pipeline over all reviews.
# --------------------------------------------------------------------------
@dataclass
class AnalysisRun:
    """Results + metrics for one provider run over all reviews."""

    provider: str            # display name of the provider
    model_name: str          # underlying model identifier
    results: list[dict]      # one entry per review
    total_latency_s: float   # wall-clock time for the whole batch of raw calls
    parse_failures: int      # number of outputs that needed OutputFixingParser


def analyze_reviews(model: BaseChatModel, reviews: list[dict]) -> AnalysisRun:
    """Run the pipeline over ``reviews`` for a single model.

    Uses ``raw_chain.batch(...)`` to fetch all raw model outputs (demonstrating
    LangChain batching), then parses each one with retry/fixing on failure.
    """
    prompt, raw_chain, parsed_chain, parser = build_chains(model)

    inputs = [
        {"product": r["product"], "review_text": r["text"]}
        for r in reviews
    ]

    start = time.perf_counter()
    raw_outputs: list[str] = raw_chain.batch(inputs)
    total_latency = time.perf_counter() - start

    results: list[dict] = []
    parse_failures = 0
    for raw, review in zip(raw_outputs, reviews):
        try:
            analysis: ReviewAnalysis = parser.parse(raw)
        except OutputParserException as exc:
            # First parse failed: attempt to repair the output by re-prompting.
            parse_failures += 1
            analysis = _fix_parse(raw, model, parser, error=exc)

        results.append(
            {
                "id": review["id"],
                "product": review["product"],
                "rating": review["rating"],
                "raw_model_output": raw,
                "analysis": analysis.model_dump(mode="json"),
            }
        )

    model_name = getattr(model, "model_name", None) or getattr(model, "model", "?")
    return AnalysisRun(
        provider=model.__class__.__name__,
        model_name=model_name,
        results=results,
        total_latency_s=total_latency,
        parse_failures=parse_failures,
    )


def load_reviews() -> list[dict]:
    """Load the 10 review records from the JSON data file."""
    with REVIEWS_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# 4. Cross-provider comparison + report.
# --------------------------------------------------------------------------
def _build_comparison_report(runs: list[AnalysisRun]) -> str:
    """Write a markdown report comparing schema validity and latency."""
    lines: list[str] = []
    lines.append("# Review Analysis - Cross-Provider Comparison\n")

    lines.append("## Schema validity / parse failures")
    lines.append(f"{'Provider':<14}{'Outputs':<10}{'Parse failures':<16}{'Total latency (s)':<18}{'Avg latency/review (s)':<24}")
    for run in runs:
        avg = run.total_latency_s / len(run.results)
        lines.append(
            f"{run.provider:<14}{len(run.results):<10}{run.parse_failures:<16}"
            f"{run.total_latency_s:<18.3f}{avg:<24.3f}"
        )

    lines.append("\n## Behavioural observations")
    lines.append(
        "PydanticOutputParser enforces the schema, so any valid parse returns "
        "exactly one ReviewAnalysis; differences show up in parse-failure counts "
        "(models that follow the JSON contract more reliably), and in the "
        "consistency of sentiment labels and sentiment_score values."
    )

    # Label consistency: how often "positive/negative/neutral" matches the star rating.
    lines.append("\n## Label consistency vs star rating (1-5)")
    rows = []
    for run in runs:
        for item in run.results:
            sentiment = item["analysis"]["sentiment"]
            rating = item["rating"]
            expected = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
            rows.append((run.provider, sentiment == expected, rating, sentiment, item["id"]))
    lines.append(f"{'Provider':<12}{'rating':<8}{'predicted':<12}{'matches rating':<16}{'review id':<10}")
    for provider, ok, rating, sentiment, rid in rows:
        lines.append(f"{provider:<12}{rating:<8}{sentiment:<12}{str(ok):<16}{rid:<10}")

    lines.append("\n## Findings")
    lines.append(
        "Lower parse-failure counts indicate a provider model that more reliably "
        "emits well-formed JSON matching the Pydantic schema.  Higher label "
        "consistency means the model agrees with the human star rating.  Per-review "
        "average latency drives DX: slower / higher-variance providers make batch "
        "processing of large review sets more expensive."
    )
    return "\n".join(lines)


def run(providers: list[str] | None = None) -> None:
    """Entry point for practical 3.

    ``providers`` optionally filters to ``groq`` / ``mistral`` / ``google``.
    Runs against every provider that has an API key configured.
    """
    print("#" * 60)
    print("# Practical 3 - Review Analysis Pipeline")
    print("#" * 60)

    reviews = load_reviews()
    print(f"Loaded {len(reviews)} reviews from {REVIEWS_FILE}\n")

    configured = build_configured_models()
    if not configured:
        print("No providers configured (no API keys in .env). Nothing to run.")
        return

    # Map display name -> key used for the CLI --providers filter.
    name_to_attr = {
        "Groq": "groq",
        "Mistral": "mistral",
        "Google": "google",
    }
    selected = [
        (name, model)
        for name, model in configured
        if providers is None or name_to_attr[name] in providers
    ]
    if len(selected) < 2:
        print("Need at least two configured providers to compare. Configure another key in .env.")
        return

    runs: list[AnalysisRun] = []
    for name, model in selected:
        print(f"\n=== Analysing 10 reviews with {name} ({model.__class__.__name__}) ===")
        run_data = analyze_reviews(model, reviews)
        runs.append(run_data)

        out_file = OUTPUTS_DIR / f"review_analysis_{name_to_attr[name].lower()}.json"
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(run_data.results, fh, indent=2, ensure_ascii=False)
        print(f"  Scaffold: {run_data.parse_failures} parse failure(s), "
              f"total batch latency {run_data.total_latency_s:.3f}s"
              f" -> saved {out_file.name}")

    report = _build_comparison_report(runs)
    report_file = OUTPUTS_DIR / "review_provider_comparison.md"
    report_file.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nSaved comparison -> {report_file}")


if __name__ == "__main__":
    run()