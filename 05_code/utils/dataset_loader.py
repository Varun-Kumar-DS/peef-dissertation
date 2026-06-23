"""
Dataset Loader — PEEF utility
==============================
Loads TriviaQA, CNN/DailyMail, and GSM8K from HuggingFace datasets
and returns a standardised list of dicts ready for PromptBuilder.

Each item always has:
    - the input field(s) the template needs  (question / article / problem)
    - a "reference" key with the ground-truth answer
    - an "id" key

Usage:
    from utils.dataset_loader import load_dataset_for_task
    items = load_dataset_for_task("qa", split="validation", n=50)
"""

from __future__ import annotations

import re
from typing import Literal

Task = Literal["qa", "summarisation", "reasoning"]


def load_dataset_for_task(
    task: Task,
    split: str = "validation",
    n: int = 200,
    seed: int = 42,
) -> list[dict]:
    """
    Load n items for the given task.

    Parameters
    ----------
    task    : "qa" → TriviaQA  |  "summarisation" → CNN/DailyMail  |  "reasoning" → GSM8K
    split   : HuggingFace split name ("train", "validation", "test")
    n       : number of items to return
    seed    : shuffle seed for reproducibility

    Returns
    -------
    List of dicts with keys: id, reference, + task-specific input fields
    """
    if task == "qa":
        return _load_triviaqa(split=split, n=n, seed=seed)
    elif task == "summarisation":
        return _load_cnn_dailymail(split=split, n=n, seed=seed)
    elif task == "reasoning":
        return _load_gsm8k(split=split, n=n, seed=seed)
    else:
        raise ValueError(f"Unknown task '{task}'. Choose from: qa, summarisation, reasoning")


# ---------------------------------------------------------------------------
# TriviaQA
# ---------------------------------------------------------------------------

def _load_triviaqa(split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    # rc.nocontext = question + answer only (no passage needed)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # answer.value is the canonical answer; aliases are also acceptable
        answer = row["answer"]["value"]
        items.append({
            "id": f"triviaqa_{split}_{i:04d}",
            "question": row["question"],
            "reference": answer,
        })
    return items


# ---------------------------------------------------------------------------
# CNN / DailyMail
# ---------------------------------------------------------------------------

def _load_cnn_dailymail(split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # Truncate very long articles to ~1 500 words to stay within token limits
        article_words = row["article"].split()
        article = " ".join(article_words[:1500])
        items.append({
            "id": f"cnndm_{split}_{i:04d}",
            "article": article,
            "reference": row["highlights"],   # multi-sentence reference summary
        })
    return items


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def _load_gsm8k(split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    # GSM8K only has "train" and "test"
    gsm_split = "test" if split == "validation" else split
    ds = load_dataset("openai/gsm8k", "main", split=gsm_split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    items = []
    for i, row in enumerate(ds):
        # GSM8K answers look like "...\n#### 42" — extract the number after ####
        reference = _extract_gsm8k_answer(row["answer"])
        items.append({
            "id": f"gsm8k_{gsm_split}_{i:04d}",
            "problem": row["question"],
            "reference": reference,
        })
    return items


def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K's '#### 42' format."""
    match = re.search(r"####\s*(-?\d[\d,]*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    # Fallback: last number in the string
    nums = re.findall(r"-?\d[\d,]*", answer_text)
    return nums[-1].replace(",", "") if nums else answer_text.strip()
