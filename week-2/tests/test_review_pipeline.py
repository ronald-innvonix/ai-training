"""Tests for the Week 2 LangChain practicals.

The pipeline tests use a fake chat model (no network, no API keys) so they are
hermetic and fast.  They validate that the Runnable pipeline produces output
that conforms to the ReviewAnalysis Pydantic schema, and that parse failures
are repaired by OutputFixingParser.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from langchain_practicals.config import REVIEWS_FILE
from langchain_practicals.review_pipeline import (
    ReviewAnalysis,
    build_chains,
    _fix_parse,
)

VALID_JSON = (
    '{"sentiment": "positive", "sentiment_score": 0.8, "key_issues": [], '
    '"key_positives": ["battery life"], "summary": "Battery life is excellent.", '
    '"recommended_action": "Keep highlighting battery endurance in marketing."}'
)

BAD_JSON = "I am sorry, I cannot do that."


def make_fake_model(contents: list[str]) -> FakeMessagesListChatModel:
    """Return a fake chat model that emits the given response contents in order."""
    return FakeMessagesListChatModel(responses=[AIMessage(content=c) for c in contents])


# --------------------------------------------------------------------------
# Schema validation.
# --------------------------------------------------------------------------
def test_schema_accepts_valid_dict() -> None:
    parsed = ReviewAnalysis.model_validate_json(VALID_JSON)
    assert parsed.sentiment == "positive"
    assert isinstance(parsed.sentiment_score, float)
    assert isinstance(parsed.key_issues, list)
    assert isinstance(parsed.summary, str)
    assert isinstance(parsed.recommended_action, str)


def test_schema_rejects_invalid_sentiment() -> None:
    with pytest.raises(Exception):
        ReviewAnalysis.model_validate_json(
            json.dumps({"sentiment": "enraged", "sentiment_score": 1.0})
        )


# --------------------------------------------------------------------------
# Data file.
# --------------------------------------------------------------------------
def test_reviews_data_has_ten_entries() -> None:
    with REVIEWS_FILE.open("r", encoding="utf-8") as fh:
        reviews = json.load(fh)
    assert len(reviews) == 10
    for review in reviews:
        assert {"id", "product", "rating", "text"} <= set(review)


# --------------------------------------------------------------------------
# Runnable pipeline (raw string output, then batch demo).
# --------------------------------------------------------------------------
def test_raw_chain_returns_strings_via_batch() -> None:
    model = make_fake_model([VALID_JSON] * 10)
    _, raw_chain, _, _ = build_chains(model)
    outputs = raw_chain.batch([{"product": "X", "review_text": "y"} for _ in range(10)])
    assert len(outputs) == 10
    assert all(isinstance(o, str) for o in outputs)


def test_parsed_chain_returns_schema_valid_objects() -> None:
    model = make_fake_model([VALID_JSON])
    _, _, parsed_chain, _ = build_chains(model)
    result = parsed_chain.invoke({"product": "Phone", "review_text": "Great battery."})
    assert isinstance(result, ReviewAnalysis)
    assert result.sentiment == "positive"


def test_failed_parse_raises_output_parser_exception() -> None:
    model = make_fake_model([BAD_JSON])
    _, _, parsed_chain, _ = build_chains(model)
    with pytest.raises(OutputParserException):
        parsed_chain.invoke({"product": "Phone", "review_text": "Meh."})


# --------------------------------------------------------------------------
# Retry / fix behaviour (OutputFixingParser).
# --------------------------------------------------------------------------
def test_fix_parse_repairs_bad_output() -> None:
    bad_model = make_fake_model([BAD_JSON])
    _, _, _, parser = build_chains(bad_model)
    # The fixing model returns valid JSON in response to the original bad output.
    fix_model = make_fake_model([VALID_JSON])
    fixed = _fix_parse(BAD_JSON, fix_model, parser)
    assert isinstance(fixed, ReviewAnalysis)
    assert fixed.sentiment == "positive"


def test_prompt_includes_format_instructions() -> None:
    model = make_fake_model([VALID_JSON])
    prompt, _, _, _ = build_chains(model)
    formatted_msgs = prompt.format_messages(product="A", review_text="B")
    joined = " ".join(m.content for m in formatted_msgs)
    assert "sentiment_score" in joined
    assert "key_issues" in joined