"""Unit tests for the Reflexive Prompt Optimizer (Module 5).

All tests run offline: a scripted fake call_fn stands in for the Claude API.
"""

import json

import pytest

from src.evaluator import Evaluator
from src.prompt_optimizer import (
    OptimizerConfig,
    PromptGenome,
    ReflexivePromptOptimizer,
    paired_bootstrap,
)

SEED_TEMPLATE = """\
Solve the following maths problem. Give only the final numerical answer.

Problem: {{ problem }}
Answer:"""

DEV_SET = [
    {"problem": "2 + 2 = ?", "reference": "4"},
    {"problem": "10 - 3 = ?", "reference": "7"},
    {"problem": "5 * 6 = ?", "reference": "30"},
    {"problem": "9 / 3 = ?", "reference": "3"},
]


# ---------------------------------------------------------------------------
# Fake API
# ---------------------------------------------------------------------------

class ScriptedAPI:
    """
    Fake call_fn. Answers eval prompts from an answer book keyed on the
    problem text; returns canned critiques/mutations for meta-prompts.
    """

    def __init__(self, answer_book, mutations):
        self.answer_book = answer_book      # {problem_substring: answer}
        self.mutations = mutations          # list[str] returned as JSON array
        self.calls = []

    def __call__(self, prompt, model="m", max_tokens=512, temperature=0.0):
        self.calls.append(prompt)
        if "Diagnose the failure modes" in prompt:
            return "1. The template does not ask for working.", 100, 50
        if "Return ONLY a JSON array" in prompt:
            return json.dumps(self.mutations), 100, 80
        for key, answer in self.answer_book.items():
            if key in prompt:
                return answer, 20, 5
        return "unknown", 20, 5


def make_optimizer(api, tmp_path, **cfg_overrides):
    cfg = OptimizerConfig(
        generations=2, candidates_per_gen=1, bootstrap_iters=2000,
        **cfg_overrides,
    )
    return ReflexivePromptOptimizer(
        task="reasoning", evaluator=Evaluator(), config=cfg,
        call_fn=api, trace_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------

class TestPairedBootstrap:
    def test_clear_improvement_is_significant(self):
        champion = [0.0] * 20
        challenger = [1.0] * 20
        assert paired_bootstrap(champion, challenger) < 0.01

    def test_no_difference_is_not_significant(self):
        scores = [1.0, 0.0] * 10
        assert paired_bootstrap(scores, list(scores)) > 0.5

    def test_regression_is_not_significant(self):
        champion = [1.0] * 20
        challenger = [0.0] * 20
        assert paired_bootstrap(champion, challenger) > 0.9

    def test_rejects_unpaired_lists(self):
        with pytest.raises(ValueError):
            paired_bootstrap([1.0, 0.0], [1.0])


# ---------------------------------------------------------------------------
# Template safety
# ---------------------------------------------------------------------------

class TestPlaceholderGuard:
    def test_extracts_placeholders(self):
        ph = ReflexivePromptOptimizer._placeholders("Q: {{ question }} A: {{answer}}")
        assert ph == frozenset({"question", "answer"})

    def test_candidate_with_missing_placeholder_is_discarded(self, tmp_path):
        # mutation drops {{ problem }} → must be filtered out
        api = ScriptedAPI(
            answer_book={"2 + 2": "4", "10 - 3": "7", "5 * 6": "30", "9 / 3": "3"},
            mutations=["Solve it carefully. Answer:"],
        )
        opt = make_optimizer(api, tmp_path)
        champion = opt.optimize(SEED_TEMPLATE, DEV_SET)
        # only the seed genome remains; nothing valid to promote
        assert champion.generation == 0
        assert all(g.generation == 0 or g.parent_id for g in opt.genomes)
        assert len([g for g in opt.genomes if g.generation > 0]) == 0

    def test_parse_json_array_tolerates_prose_wrapper(self):
        raw = 'Here you go:\n["template a {{ x }}", "template b {{ x }}"]\nHope that helps!'
        assert len(ReflexivePromptOptimizer._parse_json_array(raw)) == 2

    def test_parse_json_array_handles_garbage(self):
        assert ReflexivePromptOptimizer._parse_json_array("no json here") == []


# ---------------------------------------------------------------------------
# Optimization loop
# ---------------------------------------------------------------------------

class TestOptimizationLoop:
    def test_promotes_genuinely_better_candidate(self, tmp_path):
        better = SEED_TEMPLATE.replace("Give only", "Think, then give")
        # Seed gets every answer wrong; the mutated template's prompt contains
        # "Think, then give", which the fake API recognises and answers right.
        api = ScriptedAPI(answer_book={"Think, then give": ""}, mutations=[better])

        def call_fn(prompt, model="m", max_tokens=512, temperature=0.0):
            if "Diagnose the failure modes" in prompt or "Return ONLY a JSON array" in prompt:
                return api(prompt, model, max_tokens, temperature)
            if "Think, then give" in prompt:           # mutated template → correct
                for item in DEV_SET:
                    if item["problem"] in prompt:
                        return item["reference"], 20, 5
            return "999999", 20, 5                     # seed template → wrong

        opt = make_optimizer(call_fn, tmp_path)
        champion = opt.optimize(SEED_TEMPLATE, DEV_SET)
        assert champion.generation > 0
        assert champion.mean_score == 1.0
        assert champion.promoted
        assert champion.p_value is not None and champion.p_value < 0.05

    def test_equal_candidate_is_rejected(self, tmp_path):
        # Both templates produce identical (perfect) answers → p high → no promotion
        api = ScriptedAPI(
            answer_book={"2 + 2": "4", "10 - 3": "7", "5 * 6": "30", "9 / 3": "3"},
            mutations=[SEED_TEMPLATE + "\n(Be precise.)"],
        )
        opt = make_optimizer(api, tmp_path)
        champion = opt.optimize(SEED_TEMPLATE, DEV_SET)
        assert champion.generation == 0
        rejected = [g for g in opt.genomes if g.generation > 0]
        assert all(not g.promoted for g in rejected)

    def test_writes_audit_trail(self, tmp_path):
        api = ScriptedAPI(
            answer_book={"2 + 2": "4", "10 - 3": "7", "5 * 6": "30", "9 / 3": "3"},
            mutations=[SEED_TEMPLATE + " extra"],
        )
        opt = make_optimizer(api, tmp_path)
        opt.optimize(SEED_TEMPLATE, DEV_SET)
        traces = list(tmp_path.glob("rpo_reasoning_*.json"))
        assert len(traces) == 1
        trace = json.loads(traces[0].read_text())
        assert trace["champion_score"] is not None
        assert trace["cost"]["api_calls"] > 0
        assert len(trace["genomes"]) >= 1

    def test_token_budget_stops_loop(self, tmp_path):
        api = ScriptedAPI(answer_book={"": "4"}, mutations=[SEED_TEMPLATE + " x"])
        opt = make_optimizer(api, tmp_path, token_budget=1)  # exhausted after gen 0
        champion = opt.optimize(SEED_TEMPLATE, DEV_SET)
        assert champion.generation == 0

    def test_eval_cache_avoids_duplicate_calls(self, tmp_path):
        api = ScriptedAPI(
            answer_book={"2 + 2": "4", "10 - 3": "7", "5 * 6": "30", "9 / 3": "3"},
            mutations=[SEED_TEMPLATE + " extra"],
        )
        opt = make_optimizer(api, tmp_path)
        opt.optimize(SEED_TEMPLATE, DEV_SET)
        # _reflect re-renders the champion's failure prompts; the cache must
        # serve them without new API calls
        eval_prompts = [
            c for c in api.calls
            if "Problem:" in c and "Diagnose" not in c and "JSON array" not in c
        ]
        assert len(eval_prompts) == len(set(eval_prompts))
