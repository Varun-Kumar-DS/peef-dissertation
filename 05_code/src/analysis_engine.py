"""
Analysis Engine — PEEF Module 4
=================================
Takes a collection of scored ExperimentResults and produces:
  - Summary statistics (mean, std, CI)
  - Wilcoxon signed-rank tests between technique pairs
  - Cohen's d effect sizes
  - Cost-efficiency data (accuracy vs. mean token usage)
  - Charts saved to 03_results/figures/

Usage:
    engine = AnalysisEngine(results_dir="../../03_results/raw")
    df = engine.load()
    summary = engine.summarise(df)
    engine.plot_comparison(df, output_dir="../../03_results/figures")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# AnalysisEngine
# ---------------------------------------------------------------------------

class AnalysisEngine:

    def __init__(self, results_dir: str | Path = "../../03_results/raw") -> None:
        self.results_dir = Path(results_dir)

    # ── data loading ────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load all JSON result files into a DataFrame."""
        records: list[dict[str, Any]] = []
        for path in self.results_dir.glob("*.json"):
            with open(path) as f:
                records.append(json.load(f))
        if not records:
            raise FileNotFoundError(f"No result files found in {self.results_dir}")
        return pd.DataFrame(records)

    # ── summary statistics ──────────────────────────────────────────────────

    def summarise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean, std, 95% CI, and mean tokens per technique × task.
        Returns a tidy summary DataFrame.
        """
        rows = []
        for (task, technique, n_shots), group in df.groupby(["task", "technique", "n_shots"]):
            scores = group["score"].dropna()
            n = len(scores)
            mean = scores.mean()
            std = scores.std()
            ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            rows.append({
                "task": task,
                "technique": technique,
                "n_shots": n_shots,
                "n": n,
                "mean_score": round(mean, 4),
                "std": round(std, 4),
                "ci_95": round(ci, 4),
                "mean_tokens": round(group["total_tokens"].mean(), 1),
            })
        return pd.DataFrame(rows).sort_values(["task", "technique", "n_shots"])

    # ── statistical tests ───────────────────────────────────────────────────

    def wilcoxon_test(
        self,
        df: pd.DataFrame,
        task: str,
        technique_a: str,
        technique_b: str,
        n_shots_a: int = 0,
        n_shots_b: int = 0,
    ) -> dict[str, Any]:
        """
        Wilcoxon signed-rank test comparing two techniques on the same task.
        Returns statistic, p-value, and Cohen's d effect size.
        """
        from scipy.stats import wilcoxon

        a = df[(df.task == task) & (df.technique == technique_a) & (df.n_shots == n_shots_a)]["score"].dropna().values
        b = df[(df.task == task) & (df.technique == technique_b) & (df.n_shots == n_shots_b)]["score"].dropna().values

        min_len = min(len(a), len(b))
        if min_len < 10:
            return {"error": f"Not enough samples (n={min_len}) for a reliable test."}

        a, b = a[:min_len], b[:min_len]
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        d = self._cohens_d(a, b)

        return {
            "task": task,
            "technique_a": f"{technique_a}({n_shots_a}-shot)",
            "technique_b": f"{technique_b}({n_shots_b}-shot)",
            "n": min_len,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 6),
            "significant": p < 0.05,
            "cohens_d": round(d, 4),
            "effect_size": self._effect_label(d),
        }

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
        return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0

    @staticmethod
    def _effect_label(d: float) -> str:
        d = abs(d)
        if d < 0.2:  return "negligible"
        if d < 0.5:  return "small"
        if d < 0.8:  return "medium"
        return "large"

    # ── plotting ─────────────────────────────────────────────────────────────

    def plot_comparison(
        self,
        df: pd.DataFrame,
        output_dir: str | Path = "../../03_results/figures",
        metric: str = "score",
    ) -> None:
        """
        Save three charts:
          1. Bar chart — mean score per technique per task
          2. Cost-efficiency scatter — accuracy vs. mean tokens
          3. Box plot — score distributions
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("Run: pip install matplotlib seaborn")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summarise(df)

        sns.set_theme(style="whitegrid", palette="muted")

        # ── 1. Bar chart ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
        for ax, task in zip(axes, ["qa", "summarisation", "reasoning"]):
            task_df = summary[summary.task == task]
            labels = task_df.apply(
                lambda r: f"{r['technique']}\n({int(r['n_shots'])}-shot)" if r["technique"] in ("few_shot", "cot") else r["technique"],
                axis=1,
            )
            ax.bar(labels, task_df.mean_score, yerr=task_df.ci_95, capsize=4, color=sns.color_palette()[0])
            ax.set_title(task.capitalize())
            ax.set_ylabel("Mean Score")
            ax.set_ylim(0, 1)
        fig.suptitle("Technique Comparison by Task", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / "bar_comparison.png", dpi=150)
        plt.close(fig)

        # ── 2. Cost-efficiency scatter ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        for task, grp in summary.groupby("task"):
            ax.scatter(grp.mean_tokens, grp.mean_score, label=task, s=80)
            for _, row in grp.iterrows():
                label = f"{row['technique']}({int(row['n_shots'])})" if row["n_shots"] else row["technique"]
                ax.annotate(label, (row.mean_tokens, row.mean_score), textcoords="offset points", xytext=(5, 3), fontsize=7)
        ax.set_xlabel("Mean Tokens per Run")
        ax.set_ylabel("Mean Score")
        ax.set_title("Cost-Efficiency: Accuracy vs Token Usage")
        ax.legend(title="Task")
        fig.tight_layout()
        fig.savefig(output_dir / "cost_efficiency.png", dpi=150)
        plt.close(fig)

        # ── 3. Box plot ───────────────────────────────────────────────────
        if "score" in df.columns:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, task in zip(axes, ["qa", "summarisation", "reasoning"]):
                task_df = df[df.task == task].copy()
                task_df["label"] = task_df.apply(
                    lambda r: f"{r['technique']}({int(r['n_shots'])})" if r["technique"] in ("few_shot", "cot") else r["technique"],
                    axis=1,
                )
                sns.boxplot(data=task_df, x="label", y="score", ax=ax)
                ax.set_title(task.capitalize())
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=30)
            fig.suptitle("Score Distributions by Technique", fontsize=14, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_dir / "score_distributions.png", dpi=150)
            plt.close(fig)

        print(f"Charts saved to {output_dir}")
