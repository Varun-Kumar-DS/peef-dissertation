"""
PEEF — Evaluation Script
=========================
Scores all experiment result files and prints a summary table.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "05_code"))
from src.evaluator import Evaluator

RESULTS_DIR  = Path("03_results/raw")
SCORED_DIR   = Path("03_results/scored")
SCORED_DIR.mkdir(parents=True, exist_ok=True)

evaluator = Evaluator()


def extract_final_answer(response: str, task: str) -> str:
    """
    For CoT responses, extract just the final answer.
    Tries multiple patterns before falling back to last line.
    """
    import re
    text = response.strip()

    # Pattern 1: explicit "Answer: X" on its own line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if re.match(r'^answer\s*:', line, re.IGNORECASE):
            return line.split(':', 1)[1].strip()

    # Pattern 2: bold markdown with "= answer" e.g. **Giugno = June**
    bold_eq = re.findall(r'\*\*[^*]+=\s*([^*]+)\*\*', text)
    if bold_eq:
        return bold_eq[-1].strip()

    # Pattern 3: any bold markdown **answer** — take the last one
    bold = re.findall(r'\*\*([^*]+)\*\*', text)
    if bold:
        candidate = bold[-1].strip()
        if len(candidate.split()) <= 8:
            return candidate

    # Pattern 4: "the answer is X" / "final answer is X"
    matches = re.findall(
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([^\.\n,]+)',
        text, re.IGNORECASE
    )
    if matches:
        return matches[-1].strip()

    # Pattern 5: "therefore, X" where X is short (≤6 words)
    match = re.search(
        r'therefore[,:]?\s+(?:it\s+is\s+|the\s+answer\s+is\s+)?([^\.\n]+)',
        text, re.IGNORECASE
    )
    if match:
        ans = match.group(1).strip()
        if len(ans.split()) <= 6:
            return ans

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


# Map filename prefixes to tasks
TASK_MAP = {
    "qa":             "qa",
    "summarisation":  "summarisation",
    "reasoning":      "reasoning",
}

summary_rows = []

result_files = sorted(RESULTS_DIR.glob("*.jsonl"))
if not result_files:
    print("No result files found in 03_results/raw/")
    sys.exit(1)

for fpath in result_files:
    # Skip pilot file
    if "pilot" in fpath.name:
        continue

    # Detect task from filename
    task = None
    for prefix, t in TASK_MAP.items():
        if fpath.name.startswith(prefix):
            task = t
            break
    if task is None:
        print(f"Skipping {fpath.name} — could not detect task.")
        continue

    print(f"Evaluating {fpath.name} ({task})...")

    rows = [json.loads(line) for line in fpath.read_text().splitlines() if line.strip()]

    if not rows:
        print(f"  WARNING: {fpath.name} is empty, skipping.")
        continue

    # For CoT techniques, extract just the final answer from the response
    technique = rows[0].get("technique", "")
    is_cot = technique in ("cot", "zero_shot_cot")

    # For QA CoT only — extract final answer from markdown/reasoning text
    # Reasoning CoT: pass full response; the evaluator's number extractor handles it
    if is_cot and task == "qa":
        predictions = [extract_final_answer(r["response"], task) for r in rows]
    else:
        predictions = [r["response"] for r in rows]

    references  = [r["metadata"]["reference"] for r in rows]

    if not predictions:
        print(f"  WARNING: No predictions found in {fpath.name}, skipping.")
        continue

    scores = evaluator.score_batch(task=task, predictions=predictions, references=references)

    if not scores:
        print(f"  WARNING: No scores computed for {fpath.name}, skipping.")
        continue

    # Attach scores to rows and save
    scored_rows = []
    for row, score in zip(rows, scores):
        row["scores"] = score
        scored_rows.append(row)

    out_path = SCORED_DIR / fpath.name
    with open(out_path, "w") as f:
        for row in scored_rows:
            f.write(json.dumps(row) + "\n")

    # Compute mean scores for summary
    if task in ("qa", "reasoning"):
        mean_score = sum(s["exact_match"] for s in scores) / len(scores)
        summary_rows.append({
            "experiment": fpath.stem,
            "task": task,
            "n": len(rows),
            "exact_match": round(mean_score, 4),
            "rougeL_f": "-",
            "bertscore_f1": "-",
        })
    else:
        mean_rouge = sum(s["rougeL_f"] for s in scores) / len(scores)
        mean_bert  = sum(s["bertscore_f1"] for s in scores) / len(scores)
        summary_rows.append({
            "experiment": fpath.stem,
            "task": task,
            "n": len(rows),
            "exact_match": "-",
            "rougeL_f": round(mean_rouge, 4),
            "bertscore_f1": round(mean_bert, 4),
        })

# Print summary table
print("\n" + "=" * 80)
print(f"{'EXPERIMENT':<35} {'TASK':<15} {'N':>5} {'EXACT MATCH':>12} {'ROUGE-L':>9} {'BERTSCORE':>10}")
print("=" * 80)
for row in summary_rows:
    print(
        f"{row['experiment']:<35} "
        f"{row['task']:<15} "
        f"{row['n']:>5} "
        f"{str(row['exact_match']):>12} "
        f"{str(row['rougeL_f']):>9} "
        f"{str(row['bertscore_f1']):>10}"
    )
print("=" * 80)
print(f"\nScored files saved to: {SCORED_DIR}")
