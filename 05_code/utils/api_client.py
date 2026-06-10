"""
API Client utility — thin wrapper around the Anthropic SDK.
Loads the API key from environment and exposes a simple call() function.

Setup:
    1. Copy .env.example → .env
    2. Add your ANTHROPIC_API_KEY
    3. Run: pip install python-dotenv anthropic
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


def get_client() -> anthropic.Anthropic:
    """Return an authenticated Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found. "
            "Copy .env.example → .env and add your key."
        )
    return anthropic.Anthropic(api_key=api_key)


def call(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> tuple[str, int, int]:
    """
    Single API call.

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
    response, in_tok, out_tok = call("Say 'API connection successful' and nothing else.")
    print(f"Response: {response}")
    print(f"Tokens — input: {in_tok}, output: {out_tok}")
    print("Connection OK.")


if __name__ == "__main__":
    test_connection()
