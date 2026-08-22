"""Command-line entry point for the Week 2 LangChain practicals.

Provides a ``langchain-practicals`` console script exposing subcommands:

    langchain-practicals prompts    # Practical 1 - prompt templates
    langchain-practicals models     # Practical 2 - model invocation methods
    langchain-practicals reviews [--providers groq mistral google]
                                    # Practical 3 - review analysis pipeline
    langchain-practicals all        # run every practical in order
"""

from __future__ import annotations

import argparse
import sys

from . import model_invocation, prompts, review_pipeline

_KNOWN_PROVIDERS = ("groq", "mistral", "google")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langchain-practicals",
        description="Run the Week 2 LangChain practicals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prompts", help="Practical 1 - reusable prompt templates + LCEL.")

    sub.add_parser("models", help="Practical 2 - invoke/stream/ainvoke/astream with timing.")

    p_reviews = sub.add_parser(
        "reviews",
        help="Practical 3 - structured review analysis across providers.",
    )
    p_reviews.add_argument(
        "--providers",
        nargs="*",
        choices=list(_KNOWN_PROVIDERS),
        default=None,
        help="Only test these providers (default: all configured).",
    )

    sub.add_parser("all", help="Run practicals 1, 2 and 3 in order.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the requested practical."""
    args = _build_parser().parse_args(argv)

    if args.command in ("prompts", "all"):
        prompts.run()
        print("\n")

    if args.command in ("models", "all"):
        model_invocation.run()
        print("\n")

    if args.command in ("reviews", "all"):
        review_pipeline.run(providers=args.providers)

    return 0


if __name__ == "__main__":
    sys.exit(main())