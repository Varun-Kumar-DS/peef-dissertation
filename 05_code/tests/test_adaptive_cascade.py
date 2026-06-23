"""Unit tests for the Adaptive Cascade (Module 6).

All tests run offline: a scripted fake call_fn stands in for the Claude API.
"""

import pytest

from src.adaptive_cascade import AdaptiveCascade, _normalise_answer
from src.evaluator import Evaluator
from src.prompt_builder import PromptBuilder

ITEMS = [
    {"id": "easy", "problem": "2 + 2 = ?", "reference": "4"},
    {"id": "hard", "problem": "What is 17 * 23?", "reference": "391"},
]


class ScriptedAPI:
    """
    Fake call_fn. The easy problem always gets a consistent cheap answer;
    the hard one gets disagreeing cheap answers (forcing escalation) and a
    correct CoT answer.
    """

    def __init__(self):
        self.hard_cheap_answers = iter(["390", "401", "391"])  # 1/3 agreement
        self.calls = {"cheap": 0, "strong": 0}

    def __call__(self, prompt, model="m", max_tokens=64, temperature=0.0):
        is_cot = "step by step" in prompt
        if "17 * 23" in prompt:
            if is_cot:
                self.calls["strong"] += 1
                return "17 * 23 = 17*20 + 17*3 = 340 + 51 = 391", 200, 300
            self.calls["cheap"] += 1
            return next(self.hard_cheap_answers), 20, 5
        # easy problem: unanimous cheap answers
        self.calls["strong" if is_cot else "cheap"] += 1
        return "4", 20, 5


def make_cascade(api, **kwargs):
    defaults = dict(
        task="reasoning",
        cheap_builder=PromptBuilder(task="reasoning", technique="zero_shot"),
        strong_builder=PromptBuilder(task="reasoning", technique="zero_shot_cot"),
        evaluator=Evaluator(),
        confidence_threshold=0.7,
        k_samples=3,
        call_fn=api,
    )
    defaults.update(kwargs)
    return AdaptiveCascade(**defaults)


class TestNormalisation:
    def test_reasoning_extracts_last_number(self):
        assert _normalise_answer("The answer is 391.", "reasoning") == "391"
        assert _normalise_answer("1,000 in total", "reasoning") == "1000"

    def test_qa_normalises_text(self):
        assert _normalise_answer("  Paris!  ", "qa") == "paris"


class TestRouting:
    def test_confident_item_stays_cheap(self):
        api = ScriptedAPI()
        decisions = make_cascade(api).run([ITEMS[0]])
        d = decisions[0]
        assert d.tier_used == "cheap"
        assert d.confidence == 1.0
        assert d.score == 1.0
        assert api.calls["strong"] == 0

    def test_uncertain_item_escalates_and_cot_answer_wins(self):
        api = ScriptedAPI()
        decisions = make_cascade(api).run([ITEMS[1]])
        d = decisions[0]
        assert d.tier_used == "strong"
        assert d.confidence == pytest.approx(1 / 3, abs=1e-4)
        assert "391" in d.answer
        assert d.score == 1.0
        assert d.metadata["strong_tokens"] > 0

    def test_threshold_zero_never_escalates(self):
        api = ScriptedAPI()
        decisions = make_cascade(api, confidence_threshold=0.0).run(ITEMS)
        assert all(d.tier_used == "cheap" for d in decisions)

    def test_tokens_accumulate_across_tiers(self):
        api = ScriptedAPI()
        d = make_cascade(api).run([ITEMS[1]])[0]
        # 3 cheap samples (25 each) + 1 strong call (500)
        assert d.tokens_spent == 3 * 25 + 500


class TestSummary:
    def test_summary_reports_savings(self):
        api = ScriptedAPI()
        cascade = make_cascade(api)
        decisions = cascade.run(ITEMS)
        s = cascade.summary(decisions)
        assert s["n_items"] == 2
        assert s["escalation_rate"] == 0.5
        assert s["accuracy"] == 1.0
        assert s["always_cot_tokens_est"] == 1000      # 500 * 2 items
        assert s["cascade_tokens"] < s["always_cot_tokens_est"]
        assert s["token_savings_pct"] > 0

    def test_threshold_sweep_is_monotonic(self):
        api = ScriptedAPI()
        cascade = make_cascade(api)
        decisions = cascade.run(ITEMS)
        rows = AdaptiveCascade.replay_threshold_sweep(decisions, [0.0, 0.5, 1.0])
        rates = [r["escalation_rate"] for r in rows]
        assert rates == sorted(rates)


class TestValidation:
    def test_rejects_bad_threshold(self):
        with pytest.raises(ValueError):
            make_cascade(ScriptedAPI(), confidence_threshold=1.5)

    def test_rejects_bad_k(self):
        with pytest.raises(ValueError):
            make_cascade(ScriptedAPI(), k_samples=0)
