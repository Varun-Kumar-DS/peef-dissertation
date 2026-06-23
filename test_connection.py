"""
Run this first to confirm your Claude API key is working.

    python test_connection.py

Expected output:
    Testing Claude API connection...
    Response  : API connection successful
    Tokens    — input: X, output: X
    All good! You are ready to run experiments.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from this folder
load_dotenv(Path(__file__).parent / ".env")

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key or api_key == "your-key-here":
    print("ERROR: ANTHROPIC_API_KEY not found in .env file.")
    print("1. Go to https://console.anthropic.com")
    print("2. Click 'API Keys' and create a new key")
    print("3. Open .env and set ANTHROPIC_API_KEY=your-key")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic library not installed.")
    print("Run: pip install anthropic python-dotenv")
    sys.exit(1)

print("Testing Claude API connection...")

client = anthropic.Anthropic(api_key=api_key)
message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=20,
    messages=[{"role": "user", "content": "Say 'API connection successful' and nothing else."}],
)

text    = message.content[0].text.strip()
in_tok  = message.usage.input_tokens
out_tok = message.usage.output_tokens

print(f"Response  : {text}")
print(f"Tokens    — input: {in_tok}, output: {out_tok}")
print("All good! You are ready to run experiments.")
