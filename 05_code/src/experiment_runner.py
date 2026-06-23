"""
Experiment Runner — PEEF Module 2
===================================
Handles all Claude API calls. Manages rate limiting with exponential back-off,
deduplication (never re-sends a completed prompt), result persistence, and
optional concurrency.

Usage:
    runner = ExperimentRunner(config_path="configs/qa_few_shot_4shot.yaml")
    results = runner.run()
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import anthropic


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """One row in the results file."""
    run_id: str
    config_name: str
    task: str
    technique: str
    n_shots: int
    model: str
    temperature: float
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: str
    error: str | None = None
    score: float | None = None          # filled in by Evaluator
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRunner:
    """
    Runs a batch of prompts against the Claude API.

    Parameters
    ----------
    config : dict
        Experiment configuration (loaded from YAML).
    output_dir : str | Path
        Directory to write raw results JSON to.
    cache_dir : str | Path
        Directory to store the deduplication cache.
    max_retries : int
        Maximum retries on rate-limit / transient errors.
    base_delay : float
        Base delay (seconds) for exponential back-off.
    """

    config: dict[str, Any]
    output_dir: str | Path = Path("../../03_results/raw")
    cache_dir: str | Path = Path(".cache")
    max_retries: int = 5
    base_delay: float = 1.0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from dotenv import load_dotenv
        from pathlib import Path as _Path
        load_dotenv(_Path(__file__).parent.parent.parent / ".env")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not found in .env file.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._cache: set[str] = self._load_cache()

    # ── public API ──────────────────────────────────────────────────────────

    def run(self, prompts: list[dict[str, Any]]) -> list[ExperimentResult]:
        """
        Run all prompts, skipping any that are already cached.

        Each item in `prompts` must have at least:
            {"run_id": str, "prompt": str, ...metadata fields...}
        """
        results: list[ExperimentResult] = []
        for item in prompts:
            if item["run_id"] in self._cache:
                print(f"[SKIP] {item['run_id']} already completed.")
                continue
            result = self._run_one(item)
            results.append(result)
            self._save_result(result)
            self._cache.add(item["run_id"])
            self._persist_cache()
        return results

    # ── internals ───────────────────────────────────────────────────────────

    def _run_one(self, item: dict[str, Any]) -> ExperimentResult:
        import datetime

        prompt = item["prompt"]
        model = self.config.get("model", "claude-haiku-4-5-20251001")
        temperature = self.config.get("temperature", 0.0)
        max_tokens = self.config.get("max_tokens", 512)

        for attempt in range(self.max_retries):
            try:
                message = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = message.content[0].text
                in_tok  = message.usage.input_tokens
                out_tok = message.usage.output_tokens
                return ExperimentResult(
                    run_id=item["run_id"],
                    config_name=self.config.get("name", "unnamed"),
                    task=self.config["task"],
                    technique=self.config["technique"],
                    n_shots=self.config.get("n_shots", 0),
                    model=model,
                    temperature=temperature,
                    prompt=prompt,
                    response=response_text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    total_tokens=in_tok + out_tok,
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    metadata=item.get("metadata", {}),
                )
            except Exception as exc:
                if "quota" in str(exc).lower() or "rate" in str(exc).lower():
                    delay = self.base_delay * (2 ** attempt)
                    print(f"[RATE LIMIT] Attempt {attempt + 1}/{self.max_retries}. Waiting {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"[API ERROR] {exc}")
                    return ExperimentResult(
                        run_id=item["run_id"],
                        config_name=self.config.get("name", "unnamed"),
                        task=self.config["task"],
                        technique=self.config["technique"],
                        n_shots=self.config.get("n_shots", 0),
                        model=model,
                        temperature=temperature,
                        prompt=prompt,
                        response="",
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        timestamp="",
                        error=str(exc),
                    )
        raise RuntimeError(f"Max retries exceeded for run_id={item['run_id']}")

    def _save_result(self, result: ExperimentResult) -> None:
        path = self.output_dir / f"{result.run_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)

    def _load_cache(self) -> set[str]:
        path = self.cache_dir / "completed.txt"
        if path.exists():
            return set(path.read_text().splitlines())
        return set()

    def _persist_cache(self) -> None:
        path = self.cache_dir / "completed.txt"
        path.write_text("\n".join(sorted(self._cache)))

    @staticmethod
    def make_run_id(config_name: str, example_id: str) -> str:
        """Deterministic run ID from config + example."""
        raw = f"{config_name}::{example_id}"
        return hashlib.md5(raw.encode()).hexdigest()[:12] + "_" + example_id[:20]
