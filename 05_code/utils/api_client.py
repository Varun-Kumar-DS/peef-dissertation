"""
API Client utility — Claude (Anthropic) wrapper for PEEF.
Loads the API key from .env and exposes a simple call() function
used by all PEEF modules.

Setup:
    1. Go to https://console.anthropic.com and get an API key
    2. Add to .env:  ANTHROPIC_API_KEY=your-key-here
    3. Run: pip install anthropic python-dotenv
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env from project root (three levels up from this file)
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def get_client() -> anthropic.Anthropic:
    """Return an authenticated Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. "
            "Add ANTHROPIC_API_KEY=your-key to your .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


def call(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[str, int, int]:
    """
    Single Claude API call.

    Returns
    -------
    (response_text, input_tokens, output_tokens)
    """
    client = get_client()
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text
    return text, message.usage.input_tokens, message.usage.output_tokens


def test_connection() -> None:
    """Quick smoke-test — prints model response and token count."""
    print("Testing Claude API connection...")
    response, in_tok, out_tok = call(
        "Say 'API connection successful' and nothing else.",
        max_tokens=20,
    )
    print(f"Response  : {response.strip()}")
    print(f"Tokens    — input: {in_tok}, output: {out_tok}")
    print("Connection OK.")


if __name__ == "__main__":
    test_connection()
