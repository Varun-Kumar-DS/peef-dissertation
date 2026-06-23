"""
Adaptive experiments — Modules 5 & 6 demo run
==============================================
Runs both novel PEEF modules against the live Claude API on a small
GSM8K-style reasoning set:

  Part A — Reflexive Prompt Optimizer: evolves the zero-shot reasoning
           template from its own failures (audit trail saved to
           03_results/optimizer_traces/).
  Part B — Adaptive Cascade: routes each item zero-shot→CoT based on
           self-consistency confidence and reports token savings vs an
           always-CoT baseline.

Usage (from 05_code/, with ANTHROPIC_API_KEY in .env):
    python ../02_experiments/combined_techniques/run_adaptive_experiments.py

Swap DEV_SET for a real GSM8K subset (04_evaluation/benchmarks/) for the
dissertation runs — this file is the end-to-end smoke test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make 05_code/ importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "05_code"))

from src.adaptive_cascade import AdaptiveCascade
from src.evaluator import Evaluator
from src.prompt_builder import PromptBuilder, _TEMPLATES
from src.prompt_optimizer import OptimizerConfig, ReflexivePromptOptimizer

# Small built-in dev set (replace with a GSM8K subset for real runs)
DEV_SET = [
    {"problem": "Natalia sold clips to 48 of her friends in April, and then she "
                "sold half as many clips in May. How many clips did Natalia sell "
                "altogether in April and May?", "reference": "72"},
    {"problem": "Weng earns $12 an hour for babysitting. Yesterday, she just did "
                "50 minutes of babysitting. How much did she earn?", "reference": "10"},
    {"problem": "Betty has $50. She needs $100 for a wallet. Her parents give her "
                "$15 and her grandparents give twice as much as her parents. How "
                "much more money does Betty need?", "reference": "5"},
    {"problem": "James writes a 3-page letter to 2 different friends twice a week. "
                "How many pages does he write a year?", "reference": "624"},
    {"problem": "A robe takes 2 bolts of blue fiber and half that much white fiber. "
                "How many bolts in total does it take?", "reference": "3"},
    {"problem": "Tim has 30 less apples than Martha, and Harry has half as many "
                "apples as Tim. If Martha has 68 apples, how many apples does "
                "Harry have?", "reference": "19"},
]

RESULTS_DIR = Path(__file__).resolve().parents[2] / "03_results"


def part_a_optimizer() -> None:
    print("=" * 70)
    print("PART A — Reflexive Prompt Optimizer (Module 5)")
    print("=" * 70)
    opt = ReflexivePromptOptimizer(
        task="reasoning",
        evaluator=Evaluator(),
        config=OptimizerConfig(generations=3, candidates_per_gen=3, token_budget=200_000),
        trace_dir=RESULTS_DIR / "optimizer_traces",
    )
    champion = opt.optimize(
        seed_template=_TEMPLATES["reasoning_zero_shot"],
        dev_set=DEV_SET,
    )
    print(f"\nChampion (generation {champion.generation}, "
          f"score {champion.mean_score:.3f}):\n")
    print(champion.template)
    print(f"\nSpend: {opt.ledger.api_calls} calls, "
          f"{opt.ledger.total_tokens:,} tokens, "
          f"~${opt.ledger.usd(opt.config.model)}")


def part_b_cascade() -> None:
    print("\n" + "=" * 70)
    print("PART B — Adaptive Cascade (Module 6)")
    print("=" * 70)
    cascade = AdaptiveCascade(
        task="reasoning",
        cheap_builder=PromptBuilder(task="reasoning", technique="zero_shot"),
        strong_builder=PromptBuilder(task="reasoning", technique="zero_shot_cot"),
        evaluator=Evaluator(),
        confidence_threshold=0.7,
        k_samples=3,
    )
    decisions = cascade.run(DEV_SET)
    for d in decisions:
        print(f"  {d.item_id}: tier={d.tier_used:6s} conf={d.confidence:.2f} "
              f"score={d.score} tokens={d.tokens_spent}")
    print("\nSummary:")
    print(json.dumps(cascade.summary(decisions), indent=2))


if __name__ == "__main__":
    part_a_optimizer()
    part_b_cascade()
