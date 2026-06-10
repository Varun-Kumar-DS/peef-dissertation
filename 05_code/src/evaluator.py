"""
Evaluator — PEEF Module 3
===========================
Scores model responses against ground-truth answers.
Automatically selects the correct metric per task type.

Metrics:
    qa          → exact_match (normalised string comparison)
    summarisation → ROUGE-L + BERTScore
    reasoning   → exact_match on final numerical answer

Usage:
    ev = Evaluator()
    score = ev.score(task="qa", prediction="Paris", reference="Paris")
    scores = ev.score_batch(task="summarisation", predictions=[...], references=[...])
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Literal

Task = Literal["qa", "summarisation", "reasoning"]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation and extra whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_number(text: str) -> str | None:
    """Extract the last number from a model response (for GSM8K)."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class Evaluator:
    """
    Scores predictions against references for a given task.

    For summarisation, installs rouge-score and bert-score on first use.
    """

    bertscore_model: str = "distilbert-base-uncased"
    bertscore_lang: str = "en"

    # ── public API ──────────────────────────────────────────────────────────

    def score(self, task: Task, prediction: str, reference: str) -> dict[str, float]:
        """Score a single prediction. Returns a dict of metric_name → score."""
        if task == "qa":
            return self._exact_match(prediction, reference)
        elif task == "reasoning":
            return self._reasoning_exact_match(prediction, reference)
        elif task == "summarisation":
            return self._summarisation_scores([prediction], [reference])
        else:
            raise ValueError(f"Unknown task: {task}")

    def score_batch(
        self,
        task: Task,
        predictions: list[str],
        references: list[str],
    ) -> list[dict[str, float]]:
        """Score a list of predictions. For summarisation, runs BERTScore in batch."""
        if task == "summarisation":
            batch = self._summarisation_scores(predictions, references)
            # batch returns one dict per prediction
            return batch  # type: ignore[return-value]
        return [self.score(task, p, r) for p, r in zip(predictions, references)]

    # ── per-task scorers ────────────────────────────────────────────────────

    def _exact_match(self, prediction: str, reference: str) -> dict[str, float]:
        match = float(_normalise(prediction) == _normalise(reference))
        return {"exact_match": match}

    def _reasoning_exact_match(self, prediction: str, reference: str) -> dict[str, float]:
        pred_num = _extract_number(prediction)
        ref_num = _extract_number(reference)
        if pred_num is None or ref_num is None:
            return {"exact_match": 0.0}
        match = float(pred_num == ref_num)
        return {"exact_match": match}

    def _summarisation_scores(
        self,
        predictions: list[str],
        references: list[str],
    ) -> list[dict[str, float]]:
        rouge_scores = self._rouge(predictions, references)
        bert_scores = self._bertscore(predictions, references)
        return [
            {**r, **b}
            for r, b in zip(rouge_scores, bert_scores)
        ]

    def _rouge(
        self,
        predictions: list[str],
        references: list[str],
    ) -> list[dict[str, float]]:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError("Run: pip install rouge-score")

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        results = []
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            results.append({
                "rouge1_f":  round(s["rouge1"].fmeasure,  4),
                "rouge2_f":  round(s["rouge2"].fmeasure,  4),
                "rougeL_f":  round(s["rougeL"].fmeasure,  4),
            })
        return results

    def _bertscore(
        self,
        predictions: list[str],
        references: list[str],
    ) -> list[dict[str, float]]:
        try:
            from bert_score import score as bs_score
        except ImportError:
            raise ImportError("Run: pip install bert-score")

        P, R, F1 = bs_score(
            predictions,
            references,
            lang=self.bertscore_lang,
            model_type=self.bertscore_model,
            verbose=False,
        )
        return [
            {"bertscore_p": round(float(p), 4),
             "bertscore_r": round(float(r), 4),
             "bertscore_f1": round(float(f), 4)}
            for p, r, f in zip(P, R, F1)
        ]
