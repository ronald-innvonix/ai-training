"""Shared chat-model factory used across the practicals.

Providers are instantiated only when their API key is present in the
environment, so every part degrades gracefully when a key is missing.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

from .config import get_settings

# A configured provider entry: (display name, chat-model instance).
Provider = tuple[str, BaseChatModel]


def build_configured_models() -> list[Provider]:
    """Build and return a chat model for every provider that has an API key.

    Providers without a key are skipped so that commands can still run
    (against the providers that are configured) without hardcoding keys.
    """
    settings = get_settings()
    builders: list[tuple[str, Callable[[], BaseChatModel]]] = []

    if settings.groq_api_key:
        builders.append(
            (
                "Groq",
                lambda: ChatGroq(
                    api_key=settings.groq_api_key,
                    model=settings.groq_model,
                    temperature=0.2,
                ),
            )
        )

    if settings.mistral_api_key:
        builders.append(
            (
                "Mistral",
                lambda: ChatMistralAI(
                    api_key=settings.mistral_api_key,
                    model=settings.mistral_model,
                    temperature=0.2,
                ),
            )
        )

    if settings.google_api_key:
        builders.append(
            (
                "Google",
                lambda: ChatGoogleGenerativeAI(
                    api_key=settings.google_api_key,
                    model=settings.google_model,
                    temperature=0.2,
                ),
            )
        )

    return [(name, builder()) for name, builder in builders]


def get_first_configured_model() -> BaseChatModel | None:
    """Return the first configured chat model (used by small demo chains)."""
    models = build_configured_models()
    return models[0][1] if models else None