"""
Reflexive Prompt Optimizer — PEEF Module 5
============================================
Closed-loop, failure-driven prompt evolution.

Where Modules 1-4 *compare* hand-written prompts, this module makes the
prompt improve itself:

    1. EVALUATE  — run the current champion template on a dev set
    2. REFLECT   — the LLM diagnoses its own failure cases and names the
                   recurring failure modes (the "critique")
    3. MUTATE    — the LLM rewrites the template into K diverse candidates,
                   each targeting the diagnosed failure modes
    4. SELECT    — a candidate replaces the champion ONLY if it beats it
                   under a paired bootstrap significance test (p < alpha),
                   so improvements cannot be promoted by luck
    5. repeat for G generations or until the token budget / patience runs out

Novelty relative to the dissertation baseline: this converts the static
zero-shot / few-shot / CoT comparison into an *automatic prompt engineering*
experiment (cf. APE, Zhou et al. 2022; OPRO, Yang et al. 2023) with a
statistical promotion gate — a guard against dev-set noise that prior
automatic-prompting work typically omits.

Every API call, critique, candidate, score, and promotion decision is
written to an audit-trail JSON so the full optimization lineage is
reproducible in the dissertation write-up.

Usage:
    from src.prompt_optimizer import ReflexivePromptOptimizer, OptimizerConfig
    from src.evaluator import Evaluator

    opt = ReflexivePromptOptimizer(
        task="reasoning",
        evaluator=Evaluator(),
        config=OptimizerConfig(generations=4, candidates_per_gen=3),
    )
    best = opt.optimize(seed_template=SEED, dev_set=dev_items)
    print(best.template)          # use via PromptBuilder(custom_template=...)

Offline testing: pass call_fn=<your mock> — no API key needed.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from jinja2 import Environment, BaseLoader

# call_fn signature: (prompt, model=..., max_tokens=..., temperature=...) -> (text, in_tok, out_tok)
CallFn = Callable[..., tuple[str, int, int]]

# Primary scalar metric per task (Evaluator returns a dict of metrics).
_PRIMARY_METRIC: dict[str, str] = {
    "qa": "exact_match",
    "reasoning": "exact_match",
    "summarisation": "rougeL_f",
}

# USD per million tokens (input, output) — used for the cost ledger only.
# Claude Haiku pricing as of 2025.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """Hyper-parameters for the optimization loop."""
    generations: int = 4              # max reflect→mutate→select cycles
    candidates_per_gen: int = 3       # mutations proposed each generation
    n_failures_shown: int = 6         # worst dev items shown to the critic
    alpha: float = 0.05               # promotion significance threshold
    bootstrap_iters: int = 10_000     # paired bootstrap resamples
    patience: int = 2                 # stop after this many gens w/o promotion
    model: str = "claude-haiku-4-5-20251001"
    eval_temperature: float = 0.0     # deterministic dev-set evaluation
    mutate_temperature: float = 0.9   # diverse candidate generation
    max_eval_tokens: int = 512
    max_mutate_tokens: int = 2000
    token_budget: int = 500_000       # hard stop on total tokens spent
    seed: int = 42


@dataclass
class PromptGenome:
    """One prompt template plus its optimization lineage."""
    genome_id: str
    template: str
    generation: int
    parent_id: str | None = None
    critique: str | None = None       # diagnosis that produced this mutation
    mean_score: float | None = None
    item_scores: list[float] = field(default_factory=list)
    promoted: bool = False
    p_value: float | None = None      # vs champion at time of evaluation


@dataclass
class CostLedger:
    """Running token and dollar spend across the whole optimization."""
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0

    def add(self, in_tok: int, out_tok: int) -> None:
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.api_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def usd(self, model: str) -> float:
        in_price, out_price = _PRICES.get(model, (3.0, 15.0))
        return round(
            self.input_tokens * in_price / 1e6 + self.output_tokens * out_price / 1e6, 4
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_bootstrap(
    champion: list[float],
    challenger: list[float],
    iters: int = 10_000,
    seed: int = 42,
) -> float:
    """
    One-sided paired bootstrap test of H1: mean(challenger) > mean(champion).

    Both score lists must be aligned on the same dev items. Returns the
    p-value: the fraction of bootstrap resamples in which the challenger
    does NOT outperform the champion. Small p ⇒ improvement is unlikely
    to be dev-set noise.
    """
    if len(champion) != len(challenger):
        raise ValueError("Score lists must be paired (same dev items, same order).")
    n = len(champion)
    deltas = [c - a for a, c in zip(champion, challenger)]
    rng = random.Random(seed)
    not_better = 0
    for _ in range(iters):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        if sum(sample) / n <= 0:
            not_better += 1
    return not_better / iters


# ---------------------------------------------------------------------------
# Meta-prompts (the prompts that improve the prompts)
# ---------------------------------------------------------------------------

_CRITIC_PROMPT = """\
You are auditing a prompt template used with a large language model.

TASK TYPE: {task}

CURRENT TEMPLATE:
<template>
{template}
</template>

The template scored {mean_score:.3f} (metric: {metric}) on a dev set.
Below are its worst-scoring cases:

{failures}

Diagnose the failure modes. Be concrete and mechanistic: what about the
TEMPLATE (wording, structure, missing constraints, output-format ambiguity)
plausibly caused these errors? Do not blame the model or the data.
List at most 3 distinct failure modes, each in 1-2 sentences."""

_MUTATE_PROMPT = """\
You are a prompt engineer improving a template for a {task} task.

CURRENT TEMPLATE (Jinja2 — placeholders in double braces MUST be preserved):
<template>
{template}
</template>

DIAGNOSED FAILURE MODES:
{critique}

Write {k} alternative templates that fix these failure modes. Rules:
- Each template MUST contain exactly these placeholders: {placeholders}
- Each must take a meaningfully DIFFERENT approach (do not produce paraphrases)
- Keep instructions concise; do not add few-shot examples
- The final line of each template should cue the model's answer the way the
  original does

Return ONLY a JSON array of {k} strings, e.g. ["template one...", "template two..."]."""


# ---------------------------------------------------------------------------
# ReflexivePromptOptimizer
# ---------------------------------------------------------------------------

class ReflexivePromptOptimizer:
    """
    Evolves a prompt template against a dev set via reflect→mutate→select.

    Parameters
    ----------
    task : str
        One of "qa", "summarisation", "reasoning" (drives metric choice
        and how dev items are rendered/scored).
    evaluator : Evaluator
        The PEEF Module 3 evaluator; its primary metric is the fitness.
    config : OptimizerConfig
    call_fn : CallFn | None
        API function with the utils.api_client.call signature. Defaults to
        the real Claude client; inject a fake for offline tests.
    trace_dir : str | Path
        Where the audit-trail JSON is written.
    """

    def __init__(
        self,
        task: str,
        evaluator: Any,
        config: OptimizerConfig | None = None,
        call_fn: CallFn | None = None,
        trace_dir: str | Path = "../../03_results/optimizer_traces",
    ) -> None:
        self.task = task
        self.evaluator = evaluator
        self.config = config or OptimizerConfig()
        self.metric = _PRIMARY_METRIC[task]
        if call_fn is None:
            from utils.api_client import call as call_fn  # lazy: needs API key
        self.call_fn = call_fn
        self.trace_dir = Path(trace_dir)
        self.ledger = CostLedger()
        self.genomes: list[PromptGenome] = []
        self._eval_cache: dict[str, tuple[str, int, int]] = {}
        self._env = Environment(loader=BaseLoader())
        self._rng = random.Random(self.config.seed)

    # ── public API ──────────────────────────────────────────────────────────

    def optimize(
        self,
        seed_template: str,
        dev_set: list[dict[str, Any]],
    ) -> PromptGenome:
        """
        Run the optimization loop and return the champion genome.

        Each dev item must contain the template's input fields plus a
        "reference" key with the ground-truth answer, e.g.
            {"problem": "...", "reference": "42"}
        """
        cfg = self.config
        champion = self._make_genome(seed_template, generation=0, parent=None)
        champion.item_scores, champion.mean_score = self._evaluate(champion, dev_set)
        champion.promoted = True
        self.genomes.append(champion)
        self._log(f"gen 0 │ seed score = {champion.mean_score:.3f}")

        stale = 0
        for gen in range(1, cfg.generations + 1):
            if self.ledger.total_tokens >= cfg.token_budget:
                self._log(f"stopping: token budget exhausted ({self.ledger.total_tokens:,})")
                break
            if stale >= cfg.patience:
                self._log(f"stopping: no promotion for {stale} generations")
                break

            critique = self._reflect(champion, dev_set)
            candidates = self._mutate(champion, critique, generation=gen)

            promoted_this_gen = False
            for cand in candidates:
                cand.item_scores, cand.mean_score = self._evaluate(cand, dev_set)
                cand.p_value = paired_bootstrap(
                    champion.item_scores, cand.item_scores,
                    iters=cfg.bootstrap_iters, seed=cfg.seed,
                )
                self.genomes.append(cand)
                verdict = "PROMOTED" if cand.p_value < cfg.alpha else "rejected"
                self._log(
                    f"gen {gen} │ candidate {cand.genome_id[:8]} "
                    f"score={cand.mean_score:.3f} vs {champion.mean_score:.3f} "
                    f"p={cand.p_value:.4f} → {verdict}"
                )
                if cand.p_value < cfg.alpha:
                    cand.promoted = True
                    champion = cand
                    promoted_this_gen = True

            stale = 0 if promoted_this_gen else stale + 1

        self._write_trace(champion)
        return champion

    # ── phase 1: evaluate ───────────────────────────────────────────────────

    def _evaluate(
        self, genome: PromptGenome, dev_set: list[dict[str, Any]]
    ) -> tuple[list[float], float]:
        template = self._env.from_string(genome.template)
        scores: list[float] = []
        for item in dev_set:
            fields = {k: v for k, v in item.items() if k != "reference"}
            prompt = template.render(**fields).strip()
            response = self._cached_call(
                prompt,
                temperature=self.config.eval_temperature,
                max_tokens=self.config.max_eval_tokens,
            )
            metrics = self.evaluator.score(
                task=self.task, prediction=response, reference=item["reference"]
            )
            # summarisation's batch scorer returns a list of one dict
            if isinstance(metrics, list):
                metrics = metrics[0]
            scores.append(float(metrics[self.metric]))
        return scores, sum(scores) / len(scores)

    # ── phase 2: reflect ────────────────────────────────────────────────────

    def _reflect(self, champion: PromptGenome, dev_set: list[dict[str, Any]]) -> str:
        worst = sorted(
            zip(dev_set, champion.item_scores), key=lambda pair: pair[1]
        )[: self.config.n_failures_shown]

        blocks = []
        template = self._env.from_string(champion.template)
        for item, score in worst:
            fields = {k: v for k, v in item.items() if k != "reference"}
            prompt = template.render(**fields).strip()
            response = self._cached_call(
                prompt,
                temperature=self.config.eval_temperature,
                max_tokens=self.config.max_eval_tokens,
            )
            blocks.append(
                f"--- case (score {score:.2f}) ---\n"
                f"INPUT: {json.dumps(fields)[:600]}\n"
                f"MODEL OUTPUT: {response[:600]}\n"
                f"EXPECTED: {item['reference'][:300]}"
            )

        critic_prompt = _CRITIC_PROMPT.format(
            task=self.task,
            template=champion.template,
            mean_score=champion.mean_score,
            metric=self.metric,
            failures="\n\n".join(blocks),
        )
        critique, in_tok, out_tok = self.call_fn(
            critic_prompt,
            model=self.config.model,
            max_tokens=self.config.max_mutate_tokens,
            temperature=0.3,
        )
        self.ledger.add(in_tok, out_tok)
        return critique.strip()

    # ── phase 3: mutate ─────────────────────────────────────────────────────

    def _mutate(
        self, champion: PromptGenome, critique: str, generation: int
    ) -> list[PromptGenome]:
        placeholders = self._placeholders(champion.template)
        mutate_prompt = _MUTATE_PROMPT.format(
            task=self.task,
            template=champion.template,
            critique=critique,
            k=self.config.candidates_per_gen,
            placeholders=", ".join(f"{{{{ {p} }}}}" for p in sorted(placeholders)),
        )
        raw, in_tok, out_tok = self.call_fn(
            mutate_prompt,
            model=self.config.model,
            max_tokens=self.config.max_mutate_tokens,
            temperature=self.config.mutate_temperature,
        )
        self.ledger.add(in_tok, out_tok)

        candidates = []
        for text in self._parse_json_array(raw):
            if self._placeholders(text) != placeholders:
                self._log("  discarded a candidate: placeholder mismatch")
                continue
            candidates.append(
                self._make_genome(
                    text, generation=generation,
                    parent=champion.genome_id, critique=critique,
                )
            )
        return candidates

    # ── helpers ─────────────────────────────────────────────────────────────

    def _cached_call(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Never pay twice for the same (prompt, temperature) at temp 0."""
        key = hashlib.md5(f"{prompt}::{temperature}".encode()).hexdigest()
        if key in self._eval_cache:
            return self._eval_cache[key][0]
        text, in_tok, out_tok = self.call_fn(
            prompt, model=self.config.model,
            max_tokens=max_tokens, temperature=temperature,
        )
        self.ledger.add(in_tok, out_tok)
        if temperature == 0.0:
            self._eval_cache[key] = (text, in_tok, out_tok)
        return text

    @staticmethod
    def _placeholders(template: str) -> frozenset[str]:
        return frozenset(re.findall(r"\{\{\s*(\w+)(?:\.\w+)?\s*\}\}", template))

    @staticmethod
    def _parse_json_array(raw: str) -> list[str]:
        """Extract the first JSON array of strings from a model response."""
        match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return [item for item in parsed if isinstance(item, str) and item.strip()]

    def _make_genome(
        self, template: str, generation: int,
        parent: str | None, critique: str | None = None,
    ) -> PromptGenome:
        gid = hashlib.md5(f"{template}::{generation}".encode()).hexdigest()[:12]
        return PromptGenome(
            genome_id=gid, template=template.strip(),
            generation=generation, parent_id=parent, critique=critique,
        )

    def _write_trace(self, champion: PromptGenome) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"rpo_{self.task}_{int(time.time())}.json"
        trace = {
            "task": self.task,
            "metric": self.metric,
            "config": asdict(self.config),
            "champion_id": champion.genome_id,
            "champion_score": champion.mean_score,
            "cost": {
                "api_calls": self.ledger.api_calls,
                "total_tokens": self.ledger.total_tokens,
                "estimated_usd": self.ledger.usd(self.config.model),
            },
            "genomes": [asdict(g) for g in self.genomes],
        }
        path.write_text(json.dumps(trace, indent=2))
        self._log(f"audit trail → {path}")

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[RPO] {msg}")
