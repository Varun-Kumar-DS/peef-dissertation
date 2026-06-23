"""
PEEF — Statistical Analysis Script
=====================================
Runs Wilcoxon signed-rank tests and computes Cohen's d effect sizes
across all experiment pairs within each task.

Usage:
    python analyse.py

Output:
    - Prints a pairwise significance table per task
    - Saves full results to 03_results/analysis/statistical_summary.json
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
from scipy import stats

SCORED_DIR   = Path("03_results/scored")
ANALYSIS_DIR = Path("03_results/analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Primary metric per task
METRIC = {
    "qa":             "exact_match",
    "summarisation":  "rougeL_f",
    "reasoning":      "exact_match",
}

TASK_MAP = {
    "qa":            "qa",
    "summarisation": "summarisation",
    "reasoning":     "reasoning",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled Cohen's d (positive = a > b)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return round((np.mean(a) - np.mean(b)) / pooled_std, 4)


def effect_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


def load_experiment(fpath: Path, task: str) -> dict[str, float]:
    """Returns {item_id: score} for the primary metric."""
    metric = METRIC[task]
    scores = {}
    for line in fpath.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = row.get("metadata", {}).get("item_id", row.get("run_id", ""))
        score   = row.get("scores", {}).get(metric, None)
        if score is not None:
            scores[item_id] = float(score)
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Group files by task
    task_experiments: dict[str, dict[str, dict]] = {
        "qa": {}, "summarisation": {}, "reasoning": {}
    }

    for fpath in sorted(SCORED_DIR.glob("*.jsonl")):
        if "pilot" in fpath.name:
            continue
        for prefix, task in TASK_MAP.items():
            if fpath.name.startswith(prefix):
                task_experiments[task][fpath.stem] = load_experiment(fpath, task)
                break

    all_results = {}

    for task, experiments in task_experiments.items():
        if len(experiments) < 2:
            continue

        metric = METRIC[task]
        print(f"\n{'='*70}")
        print(f"  TASK: {task.upper()}   |   Metric: {metric}")
        print(f"{'='*70}")

        # Mean scores
        means = {}
        for name, scores in experiments.items():
            vals = list(scores.values())
            means[name] = round(np.mean(vals), 4)
            print(f"  {name:<35} mean={means[name]:.4f}  n={len(vals)}")

        print(f"\n  {'PAIR':<55} {'p-value':>8} {'sig':>5} {'Cohen d':>9} {'effect':>12}")
        print(f"  {'-'*90}")

        pairs_results = []
        for (name_a, scores_a), (name_b, scores_b) in itertools.combinations(
            experiments.items(), 2
        ):
            # Align on shared item_ids
            common = sorted(set(scores_a) & set(scores_b))
            if len(common) < 10:
                print(f"  WARNING: only {len(common)} shared items for {name_a} vs {name_b}")
                continue

            a = [scores_a[k] for k in common]
            b = [scores_b[k] for k in common]

            # Wilcoxon signed-rank test
            try:
                stat, p = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                # All differences are zero
                p = 1.0
                stat = 0.0

            d     = cohens_d(a, b)
            sig   = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            label = effect_label(d)

            pair_label = f"{name_a}  vs  {name_b}"
            print(f"  {pair_label:<55} {p:>8.4f} {sig:>5} {d:>9.4f} {label:>12}")

            pairs_results.append({
                "experiment_a": name_a,
                "experiment_b": name_b,
                "mean_a": means[name_a],
                "mean_b": means[name_b],
                "n_common": len(common),
                "wilcoxon_stat": round(float(stat), 4),
                "p_value": round(float(p), 6),
                "significant": bool(p < 0.05),
                "cohens_d": d,
                "effect_size": label,
            })

        all_results[task] = {
            "metric": metric,
            "means": means,
            "pairwise": pairs_results,
        }

    # Save full results
    out = ANALYSIS_DIR / "statistical_summary.json"
    out.write_text(json.dumps(all_results, indent=2))

    print(f"\n\nSignificance legend: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print(f"Effect size (Cohen's d): negligible <0.2 | small 0.2-0.5 | medium 0.5-0.8 | large >0.8")
    print(f"\nFull results saved to: {out}")


if __name__ == "__main__":
    main()
