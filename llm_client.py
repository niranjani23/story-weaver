"""
OpenAI GPT-4o-mini wrapper for Story Weaver.

Two public functions:
  generate_text(prompt, temperature) → str
  generate_json(prompt)             → dict

Each call has a hard 30-second timeout. Specific error types (rate limit /
bad key / timeout) surface immediately with clear messages.
"""

import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()

_MODEL   = "gpt-4o-mini"
_TIMEOUT = 30  # seconds per API call


class RateLimitError(Exception):
    """Raised on 429 / quota exhaustion."""


def _get_api_key() -> str:
    """Read key from Streamlit secrets (cloud) or .env file (local)."""
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")


def _make_client() -> openai.OpenAI:
    key = _get_api_key()
    if not key or key == "your_openai_api_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to .env (local) or Streamlit Cloud secrets (deployed)."
        )
    return openai.OpenAI(api_key=key, timeout=_TIMEOUT)


def generate_text(prompt: str, temperature: float = 0.7) -> str:
    """
    Generate plain narrative text from GPT-4o-mini.
    Raises RateLimitError on 429; raises Exception with a clear message on all
    other failures. No silent retry delays.
    """
    try:
        client   = _make_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    except openai.RateLimitError as exc:
        if "insufficient_quota" in str(exc).lower():
            raise RateLimitError(
                "OpenAI quota exceeded — your account has no remaining credits. "
                "Add a payment method at platform.openai.com/settings/billing, "
                "then try again."
            ) from exc
        raise RateLimitError(
            "Rate limit reached — too many requests. Wait a moment and try again."
        ) from exc
    except openai.AuthenticationError as exc:
        raise Exception(
            "API key rejected. Check that OPENAI_API_KEY in your .env file is correct."
        ) from exc
    except openai.APITimeoutError as exc:
        raise Exception(
            f"OpenAI didn't respond within {_TIMEOUT}s. "
            "Check your internet connection and try again."
        ) from exc
    except openai.APIError as exc:
        raise Exception(f"OpenAI API error: {exc}") from exc


def generate_json(prompt: str) -> dict:
    """
    Generate a JSON object using OpenAI's native JSON mode.
    Returns {} on any failure — character extraction is non-critical.
    """
    try:
        client   = _make_client()
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, Exception):
        return {}
