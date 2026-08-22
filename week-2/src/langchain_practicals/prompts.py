"""Practical 1 - Prompt Templates.

Demonstrates building reusable prompts with ``ChatPromptTemplate.from_messages``
(system + human messages), including templates with named input variables, and
inspects both ``.format_messages()`` (plan old message formatting) and LCEL
composition (``prompt | model | parser``).
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .models import get_first_configured_model


def build_templates() -> dict[str, ChatPromptTemplate]:
    """Return the two reusable prompt templates used by this practical.

    - ``expert`` : a system message + a human message containing the
      ``{topic}`` and ``{tone}`` input variables.
    - ``reviewer``: a system message + a human message, also parameterised.
    """
    expert = ChatPromptTemplate.from_messages(
        [
            # System message: no variables, sets the assistant's role.
            (
                "system",
                "You are an expert technology writer. You always answer clearly, "
                "honestly, and in plain language.",
            ),
            # Human message: uses {topic} and {tone} as input variables.
            (
                "human",
                "Please explain the following topic in a {tone} tone, "
                "in four sentences max: {topic}",
            ),
        ]
    )

    travel = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise travel advisor. Your advice is always practical "
                "and budget-conscious.",
            ),
            # Multiple input variables in a single human message.
            (
                "human",
                "Suggest a {days}-day itinerary for {destination} focused on "
                "{interest}. Keep it short and actionable.",
            ),
        ]
    )

    return {"expert": expert, "travel": travel}


def demo_format_messages() -> None:
    """Show ``.format_messages()`` for both templates (no model required)."""
    templates = build_templates()

    expert_ml = templates["expert"].format_messages(topic="vector databases", tone="friendly")
    print("=== format_messages() - expert (system + human) ===")
    for message in expert_ml:
        print(f"  [{message.type}] {message.content}")

    travel_ml = templates["travel"].format_messages(
        days="3", destination="Kyoto", interest="temples and food"
    )
    print("\n=== format_messages() - travel ===")
    for message in travel_ml:
        print(f"  [{message.type}] {message.content}")


def demo_lcel() -> None:
    """Demonstrate LCEL composition ``prompt | model | parser``."""
    model = get_first_configured_model()
    template = build_templates()["expert"]

    if model is None:
        print("\n=== LCEL | chain (no provider configured; showing composition only) ===")
        print("  chain = template | model | StrOutputParser()")
        print("  (Set at least one API key in .env to run it.)")
        return

    print(f"\n=== LCEL | chain using {model.__class__.__name__} ===")
    chain = template | model | StrOutputParser()
    result = chain.invoke({"topic": "quantum computing", "tone": "fun"})
    print(result)


def run() -> None:
    """Entry point for practical 1."""
    print("#" * 60)
    print("# Practical 1 - Prompt Templates")
    print("#" * 60)
    demo_format_messages()
    demo_lcel()


if __name__ == "__main__":
    run()