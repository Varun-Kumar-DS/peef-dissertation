"""
Adaptive Cascade — PEEF Module 6
==================================
Per-instance technique selection under a token budget.

The dissertation's Modules 1-4 answer "which technique is best *on average*?"
This module asks the sharper question: "which technique does *this question*
need?" — and shows that most questions don't need the expensive one.

Mechanism (a two-tier confidence-gated cascade):

    Tier 1 (cheap)   — zero-shot prompt, sampled k times at temperature > 0.
                       The agreement ratio among the k answers is a
                       self-consistency confidence estimate
                       (cf. Wang et al. 2022, Self-Consistency).
    Gate             — if confidence ≥ threshold, accept the majority answer.
    Tier 2 (costly)  — otherwise escalate to chain-of-thought and take its
                       answer.

The headline result this enables for the dissertation: CoT-level accuracy at
a fraction of CoT's token cost, with the threshold as a tunable
accuracy/cost dial — a finding none of the three static techniques can
produce on their own.

Usage:
    from src.adaptive_cascade import AdaptiveCascade
    from src.prompt_builder import PromptBuilder

    cascade = AdaptiveCascade(
        task="reasoning",
        cheap_builder=PromptBuilder(task="reasoning", technique="zero_shot"),
        strong_builder=PromptBuilder(task="reasoning", technique="zero_shot_cot"),
        evaluator=Evaluator(),
        confidence_threshold=0.7,
    )
    results = cascade.run(items)        # items: [{"problem": ..., "reference": ...}]
    print(cascade.summary(results))

Offline testing: pass call_fn=<your mock> — no API key needed.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

CallFn = Callable[..., tuple[str, int, int]]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CascadeDecision:
    """The cascade's full decision record for one item."""
    item_id: str
    tier_used: str                    # "cheap" or "strong"
    confidence: float                 # tier-1 self-consistency agreement
    answer: str
    tier1_answers: list[str] = field(default_factory=list)
    tokens_spent: int = 0
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Answer normalisation (for vote counting)
# ---------------------------------------------------------------------------

def _normalise_answer(text: str, task: str) -> str:
    """Reduce a response to a canonical voteable form."""
    if task == "reasoning":
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return nums[-1] if nums else text.strip().lower()
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# AdaptiveCascade
# ---------------------------------------------------------------------------

class AdaptiveCascade:
    """
    Confidence-gated two-tier prompting cascade.

    Parameters
    ----------
    task : str
        "qa", "summarisation", or "reasoning" (drives answer normalisation
        and scoring metric).
    cheap_builder / strong_builder : PromptBuilder
        Tier-1 and tier-2 prompt builders (Module 1 instances).
    evaluator : Evaluator | None
        If given, each decision is scored against the item's "reference".
    confidence_threshold : float
        Minimum tier-1 agreement ratio required to skip tier 2.
        1.0 = escalate unless unanimous; 0.0 = never escalate.
    k_samples : int
        Number of tier-1 samples used for the self-consistency vote.
    sample_temperature : float
        Temperature for tier-1 sampling (must be > 0 for diversity).
    call_fn : CallFn | None
        API function with the utils.api_client.call signature. Defaults to
        the real Claude client; inject a fake for offline tests.
    """

    def __init__(
        self,
        task: str,
        cheap_builder: Any,
        strong_builder: Any,
        evaluator: Any | None = None,
        confidence_threshold: float = 0.7,
        k_samples: int = 3,
        sample_temperature: float = 0.7,
        model: str = "claude-haiku-4-5-20251001",
        cheap_max_tokens: int = 64,
        strong_max_tokens: int = 1024,
        call_fn: CallFn | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if k_samples < 1:
            raise ValueError("k_samples must be >= 1")
        self.task = task
        self.cheap_builder = cheap_builder
        self.strong_builder = strong_builder
        self.evaluator = evaluator
        self.threshold = confidence_threshold
        self.k = k_samples
        self.sample_temperature = sample_temperature
        self.model = model
        self.cheap_max_tokens = cheap_max_tokens
        self.strong_max_tokens = strong_max_tokens
        if call_fn is None:
            from utils.api_client import call as call_fn  # lazy: needs API key
        self.call_fn = call_fn

    # ── public API ──────────────────────────────────────────────────────────

    def run(self, items: list[dict[str, Any]]) -> list[CascadeDecision]:
        """
        Route each item through the cascade.

        Each item needs the builder's input fields, optionally a "reference"
        (for scoring) and an "id".
        """
        decisions = []
        for i, item in enumerate(items):
            decisions.append(self._route_one(item, item.get("id", f"item_{i:04d}")))
        return decisions

    def summary(self, decisions: list[CascadeDecision]) -> dict[str, Any]:
        """
        Headline numbers for the dissertation: accuracy, escalation rate,
        and token savings versus an always-CoT baseline.
        """
        n = len(decisions)
        escalated = [d for d in decisions if d.tier_used == "strong"]
        scored = [d for d in decisions if d.score is not None]
        cascade_tokens = sum(d.tokens_spent for d in decisions)

        # Always-CoT baseline cost estimate: the mean tier-2 spend applied
        # to every item (tier-2 spends are observed on escalated items).
        if escalated:
            mean_strong = sum(
                d.metadata.get("strong_tokens", 0) for d in escalated
            ) / len(escalated)
            always_cot_tokens = int(mean_strong * n)
        else:
            always_cot_tokens = None  # nothing escalated → no CoT cost sample

        return {
            "n_items": n,
            "escalation_rate": round(len(escalated) / n, 4) if n else 0.0,
            "mean_confidence": round(sum(d.confidence for d in decisions) / n, 4) if n else 0.0,
            "accuracy": round(sum(d.score for d in scored) / len(scored), 4) if scored else None,
            "cascade_tokens": cascade_tokens,
            "always_cot_tokens_est": always_cot_tokens,
            "token_savings_pct": (
                round(100 * (1 - cascade_tokens / always_cot_tokens), 2)
                if always_cot_tokens else None
            ),
        }

    # ── internals ───────────────────────────────────────────────────────────

    def _route_one(self, item: dict[str, Any], item_id: str) -> CascadeDecision:
        fields = {k: v for k, v in item.items() if k not in ("reference", "id")}

        # Tier 1: k cheap samples → majority vote + agreement confidence
        cheap_prompt = self.cheap_builder.build(**fields)
        answers, tier1_tokens = [], 0
        for _ in range(self.k):
            text, in_tok, out_tok = self.call_fn(
                cheap_prompt, model=self.model,
                max_tokens=self.cheap_max_tokens,
                temperature=self.sample_temperature,
            )
            answers.append(text.strip())
            tier1_tokens += in_tok + out_tok

        votes = Counter(_normalise_answer(a, self.task) for a in answers)
        majority_key, majority_count = votes.most_common(1)[0]
        confidence = majority_count / self.k
        # Recover a raw answer matching the winning normalised form
        majority_answer = next(
            a for a in answers if _normalise_answer(a, self.task) == majority_key
        )

        decision = CascadeDecision(
            item_id=item_id,
            tier_used="cheap",
            confidence=round(confidence, 4),
            answer=majority_answer,
            tier1_answers=answers,
            tokens_spent=tier1_tokens,
        )

        # Gate: escalate to CoT when tier 1 disagrees with itself
        if confidence < self.threshold:
            strong_prompt = self.strong_builder.build(**fields)
            text, in_tok, out_tok = self.call_fn(
                strong_prompt, model=self.model,
                max_tokens=self.strong_max_tokens,
                temperature=0.0,
            )
            strong_tokens = in_tok + out_tok
            decision.tier_used = "strong"
            decision.answer = text.strip()
            decision.tokens_spent += strong_tokens
            decision.metadata["strong_tokens"] = strong_tokens

        if self.evaluator is not None and "reference" in item:
            metrics = self.evaluator.score(
                task=self.task, prediction=decision.answer, reference=item["reference"]
            )
            if isinstance(metrics, list):
                metrics = metrics[0]
            primary = "exact_match" if self.task in ("qa", "reasoning") else "rougeL_f"
            decision.score = float(metrics[primary])

        return decision

    # ── threshold sweep (for the accuracy/cost trade-off figure) ────────────

    @staticmethod
    def replay_threshold_sweep(
        decisions: list[CascadeDecision],
        thresholds: list[float],
    ) -> list[dict[str, Any]]:
        """
        Re-derive escalation decisions at different thresholds WITHOUT new
        API calls, using the recorded tier-1 confidences. Only items that
        actually escalated have tier-2 answers, so this gives the exact
        curve for thresholds ≤ the original one and is the basis for the
        cost/accuracy trade-off figure.
        """
        rows = []
        for t in thresholds:
            would_escalate = sum(1 for d in decisions if d.confidence < t)
            rows.append({
                "threshold": t,
                "escalation_rate": round(would_escalate / len(decisions), 4),
            })
        return rows
