"""
PEEF — Professional Results Visualisation
==========================================
Generates publication-quality charts for the dissertation.

Charts produced:
    1. fig1_task_results.png       — Bar charts per task with significance markers
    2. fig2_combined_overview.png  — Grouped bar chart across all tasks
    3. fig3_effect_sizes.png       — Cohen's d effect size heatmaps

Usage:
    python plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCORED_DIR   = Path("03_results/scored")
ANALYSIS_DIR = Path("03_results/analysis")
FIGURES_DIR  = Path("06_writeup/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#E0E0E0",
    "grid.linewidth":     0.8,
    "xtick.bottom":       False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

# Colourblind-friendly palette (Wong 2011)
COLOURS = {
    "Zero-Shot":     "#0072B2",   # blue
    "Few-Shot (4)":  "#009E73",   # green
    "CoT":           "#D55E00",   # vermillion
    "Zero-Shot CoT": "#CC79A7",   # pink/purple
}

TECH_ORDER  = ["Zero-Shot", "Few-Shot (4)", "CoT", "Zero-Shot CoT"]
TECH_LABELS = {
    "qa_zero_shot":        "Zero-Shot",
    "qa_few_shot_4shot":   "Few-Shot (4)",
    "qa_cot":              "CoT",
    "qa_zero_shot_cot":    "Zero-Shot CoT",
    "summarisation_zero_shot":        "Zero-Shot",
    "summarisation_few_shot_4shot":   "Few-Shot (4)",
    "summarisation_cot":              "CoT",
    "summarisation_zero_shot_cot":    "Zero-Shot CoT",
    "reasoning_zero_shot":        "Zero-Shot",
    "reasoning_few_shot_4shot":   "Few-Shot (4)",
    "reasoning_cot":              "CoT",
    "reasoning_zero_shot_cot":    "Zero-Shot CoT",
}

METRIC_LABEL = {
    "qa":            "Exact Match Accuracy",
    "summarisation": "ROUGE-L F1 Score",
    "reasoning":     "Exact Match Accuracy",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(task_prefix: str, metric: str) -> dict[str, list[float]]:
    """Load item-level scores for all experiments of a given task."""
    result = {}
    for fpath in sorted(SCORED_DIR.glob(f"{task_prefix}*.jsonl")):
        tech = TECH_LABELS.get(fpath.stem)
        if tech is None:
            continue
        scores = []
        for line in fpath.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            s = row.get("scores", {}).get(metric)
            if s is not None:
                scores.append(float(s))
        result[tech] = scores
    return result


def ci95(values: list[float]) -> float:
    """95% confidence interval half-width."""
    return 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------

def get_sig_pairs(task: str, stats: dict) -> list[tuple[str, str, str]]:
    """Return (tech_a, tech_b, stars) for significant pairs only."""
    pairs = []
    for p in stats[task]["pairwise"]:
        if not p["significant"]:
            continue
        a = TECH_LABELS.get(p["experiment_a"])
        b = TECH_LABELS.get(p["experiment_b"])
        pv = p["p_value"]
        stars = "***" if pv < 0.001 else ("**" if pv < 0.01 else "*")
        if a and b:
            pairs.append((a, b, stars))
    return pairs


def add_significance_bar(ax, x1, x2, y, stars, bar_gap=0.01):
    """Draw a significance bracket between two bars."""
    ax.annotate(
        "", xy=(x1, y), xytext=(x2, y),
        arrowprops=dict(arrowstyle="-", color="black", lw=1.2),
    )
    ax.plot([x1, x1], [y - bar_gap, y], color="black", lw=1.2)
    ax.plot([x2, x2], [y - bar_gap, y], color="black", lw=1.2)
    ax.text((x1 + x2) / 2, y + bar_gap * 0.5, stars,
            ha="center", va="bottom", fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# Figure 1 — Task-specific bar charts (3 panels)
# ---------------------------------------------------------------------------

def plot_task_results(stats: dict) -> None:
    tasks = [
        ("qa",            "Question Answering\n(TriviaQA)",    "exact_match",  "qa"),
        ("summarisation", "Summarisation\n(CNN/DailyMail)",    "rougeL_f",     "summarisation"),
        ("reasoning",     "Mathematical Reasoning\n(GSM8K)",   "exact_match",  "reasoning"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Prompting Technique Performance Across Tasks",
        fontsize=15, fontweight="bold", y=1.02
    )

    for ax, (task_key, title, metric, prefix) in zip(axes, tasks):
        scores_map = load_scores(prefix, metric)

        # Order bars consistently
        techs  = [t for t in TECH_ORDER if t in scores_map]
        means  = [np.mean(scores_map[t]) for t in techs]
        errors = [ci95(scores_map[t]) for t in techs]
        colours = [COLOURS[t] for t in techs]
        x = np.arange(len(techs))

        bars = ax.bar(
            x, means, width=0.55,
            color=colours, edgecolor="white", linewidth=0.8,
            yerr=errors, capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#555555", "capthick": 1.2},
            zorder=3,
        )

        # Value labels on top of bars
        for bar, mean, err in zip(bars, means, errors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + err + 0.008,
                f"{mean:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333333",
            )

        # Axes formatting
        ax.set_title(title, pad=12)
        ax.set_ylabel(METRIC_LABEL[task_key])
        ax.set_xticks(x)
        ax.set_xticklabels(techs, rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, min(max(means) + max(errors) + 0.15, 1.05))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

        # Highlight best bar with a star
        best_idx = int(np.argmax(means))
        ax.text(
            bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
            means[best_idx] + errors[best_idx] + 0.045,
            "★ Best",
            ha="center", va="bottom",
            fontsize=8.5, color="#B8860B", fontweight="bold",
        )

        # Note for QA Zero-Shot CoT
        if task_key == "qa":
            ax.text(
                0.98, 0.04,
                "† Zero-Shot CoT score reflects\nexact-match extraction difficulty",
                transform=ax.transAxes, fontsize=7.5,
                ha="right", va="bottom", color="#888888",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.8),
            )

    # Shared legend
    handles = [mpatches.Patch(color=COLOURS[t], label=t) for t in TECH_ORDER]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=4, bbox_to_anchor=(0.5, -0.08),
        fontsize=10, frameon=True,
        edgecolor="#CCCCCC", facecolor="white",
    )

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_task_results.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Combined grouped bar chart
# ---------------------------------------------------------------------------

def plot_combined_overview(stats: dict) -> None:
    # Normalise summarisation ROUGE-L to % for readability alongside accuracy
    data = {
        "QA\n(Exact Match %)": {
            "Zero-Shot":     stats["qa"]["means"]["qa_zero_shot"] * 100,
            "Few-Shot (4)":  stats["qa"]["means"]["qa_few_shot_4shot"] * 100,
            "CoT":           stats["qa"]["means"]["qa_cot"] * 100,
            "Zero-Shot CoT": stats["qa"]["means"]["qa_zero_shot_cot"] * 100,
        },
        "Summarisation\n(ROUGE-L × 100)": {
            "Zero-Shot":     stats["summarisation"]["means"]["summarisation_zero_shot"] * 100,
            "Few-Shot (4)":  stats["summarisation"]["means"]["summarisation_few_shot_4shot"] * 100,
            "CoT":           stats["summarisation"]["means"]["summarisation_cot"] * 100,
            "Zero-Shot CoT": stats["summarisation"]["means"]["summarisation_zero_shot_cot"] * 100,
        },
        "Reasoning\n(Exact Match %)": {
            "Zero-Shot":     stats["reasoning"]["means"]["reasoning_zero_shot"] * 100,
            "Few-Shot (4)":  stats["reasoning"]["means"]["reasoning_few_shot_4shot"] * 100,
            "CoT":           stats["reasoning"]["means"]["reasoning_cot"] * 100,
            "Zero-Shot CoT": stats["reasoning"]["means"]["reasoning_zero_shot_cot"] * 100,
        },
    }

    tasks   = list(data.keys())
    n_tasks = len(tasks)
    n_techs = len(TECH_ORDER)
    width   = 0.18
    x       = np.arange(n_tasks)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, tech in enumerate(TECH_ORDER):
        offsets = x + (i - n_techs / 2 + 0.5) * width
        vals    = [data[task][tech] for task in tasks]
        bars    = ax.bar(
            offsets, vals, width=width * 0.9,
            color=COLOURS[tech], label=tech,
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=8, color="#333333",
            )

    ax.set_title(
        "Overview: Prompting Technique Performance Across All Tasks",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(
        loc="upper right", fontsize=10,
        frameon=True, edgecolor="#CCCCCC",
    )
    ax.text(
        0.01, 0.97,
        "Scores shown as percentages. Summarisation ROUGE-L scaled ×100 for comparison.",
        transform=ax.transAxes, fontsize=8, color="#888888",
        va="top", style="italic",
    )

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_combined_overview.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Effect size heatmaps (Cohen's d)
# ---------------------------------------------------------------------------

def plot_effect_sizes(stats: dict) -> None:
    task_configs = [
        ("qa",            "QA — Cohen's d\n(positive = row better than col)"),
        ("summarisation", "Summarisation — Cohen's d"),
        ("reasoning",     "Reasoning — Cohen's d"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Effect Sizes (Cohen's d) Between Prompting Techniques",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (task_key, title) in zip(axes, task_configs):
        pairwise = stats[task_key]["pairwise"]

        # Get unique technique labels
        techs = TECH_ORDER[:]
        n = len(techs)
        matrix = np.full((n, n), np.nan)

        for p in pairwise:
            a = TECH_LABELS.get(p["experiment_a"])
            b = TECH_LABELS.get(p["experiment_b"])
            if a in techs and b in techs:
                i, j = techs.index(a), techs.index(b)
                matrix[i][j] =  p["cohens_d"]
                matrix[j][i] = -p["cohens_d"]

        # Fill diagonal with 0
        np.fill_diagonal(matrix, 0)

        im = ax.imshow(matrix, cmap="RdBu", vmin=-1.0, vmax=1.0, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short = ["ZS", "FS(4)", "CoT", "ZS-CoT"]
        ax.set_xticklabels(short, fontsize=9)
        ax.set_yticklabels(short, fontsize=9)
        ax.set_title(title, fontsize=10, pad=10)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = matrix[i][j]
                if np.isnan(val):
                    continue
                color = "white" if abs(val) > 0.5 else "black"
                sig_mark = ""
                for p in pairwise:
                    a = TECH_LABELS.get(p["experiment_a"])
                    b = TECH_LABELS.get(p["experiment_b"])
                    if (a == techs[i] and b == techs[j]) or \
                       (b == techs[i] and a == techs[j]):
                        if p["significant"]:
                            sig_mark = "*"
                ax.text(j, i, f"{val:.2f}{sig_mark}",
                        ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Cohen's d")

    fig.text(0.5, -0.04,
             "* = statistically significant (p < 0.05, Wilcoxon signed-rank test). "
             "ZS = Zero-Shot, FS(4) = Few-Shot 4-shot, ZS-CoT = Zero-Shot CoT.",
             ha="center", fontsize=9, color="#555555", style="italic")

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_effect_sizes.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    stats = json.loads((ANALYSIS_DIR / "statistical_summary.json").read_text())

    print("Generating figures...\n")
    plot_task_results(stats)
    plot_combined_overview(stats)
    plot_effect_sizes(stats)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Files:")
    for f in sorted(FIGURES_DIR.glob("fig*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
