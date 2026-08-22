"""Configuration and environment handling for the project.

All API keys are read from the environment (a ``.env`` file loaded via
python-dotenv, or real environment variables).  No key is ever hardcoded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Root of the project = parent of src/langchain_practicals (two levels up).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Convenience paths used across the project.
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REVIEWS_FILE: Path = DATA_DIR / "reviews.json"


def _env(name: str) -> str | None:
    """Return ``os.getenv(name)`` stripping empties to ``None``."""
    value = os.getenv(name)
    return value if value else None


@dataclass(frozen=True)
class Settings:
    """Read-only snapshot of provider credentials and model names."""

    groq_api_key: str | None
    groq_model: str
    mistral_api_key: str | None
    mistral_model: str
    google_api_key: str | None
    google_model: str

    @property
    def configured_providers(self) -> tuple[str, ...]:
        """Names of providers that have an API key present."""
        providers: list[str] = []
        if self.groq_api_key:
            providers.append("groq")
        if self.mistral_api_key:
            providers.append("mistral")
        if self.google_api_key:
            providers.append("google")
        return tuple(providers)


# Default model used when the .env does not pin one.
_DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "mistral": "mistral-large-latest",
    "google": "gemini-2.0-flash",
}


def _load_settings() -> Settings:
    """Load ``.env`` if present, then read the relevant variables."""
    if (PROJECT_ROOT / ".env").exists():
        load_dotenv(PROJECT_ROOT / ".env")

    return Settings(
        groq_api_key=_env("GROQ_API_KEY"),
        groq_model=_env("GROQ_MODEL") or _DEFAULT_MODELS["groq"],
        mistral_api_key=_env("MISTRAL_API_KEY"),
        mistral_model=_env("MISTRAL_MODEL") or _DEFAULT_MODELS["mistral"],
        google_api_key=_env("GOOGLE_API_KEY"),
        google_model=_env("GOOGLE_MODEL") or _DEFAULT_MODELS["google"],
    )


# Lazily built singleton so nothing reads the environment at import time.
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the (cached) project settings, loading them on first use."""
    global _settings
    if _settings is None:
        _settings = _load_settings()
    return _settings