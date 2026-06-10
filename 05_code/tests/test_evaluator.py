"""Unit tests for Evaluator."""

import pytest
from src.evaluator import Evaluator, _normalise, _extract_number


class TestNormalise:
    def test_lowercase(self):
        assert _normalise("PARIS") == "paris"

    def test_strip_punctuation(self):
        assert _normalise("Paris!") == "paris"

    def test_whitespace(self):
        assert _normalise("  Paris  ") == "paris"


class TestExtractNumber:
    def test_simple(self):
        assert _extract_number("The answer is 42") == "42"

    def test_last_number(self):
        assert _extract_number("3 dogs and 5 cats: total 8") == "8"

    def test_no_number(self):
        assert _extract_number("no digits here") is None

    def test_decimal(self):
        assert _extract_number("3.14") == "3.14"


class TestQAScoring:
    ev = Evaluator()

    def test_exact_match_correct(self):
        result = self.ev.score("qa", "paris", "Paris")
        assert result["exact_match"] == 1.0

    def test_exact_match_wrong(self):
        result = self.ev.score("qa", "London", "Paris")
        assert result["exact_match"] == 0.0

    def test_normalised_match(self):
        result = self.ev.score("qa", "the eiffel tower!", "The Eiffel Tower")
        assert result["exact_match"] == 1.0


class TestReasoningScoring:
    ev = Evaluator()

    def test_correct_number(self):
        result = self.ev.score("reasoning", "The answer is 42", "42")
        assert result["exact_match"] == 1.0

    def test_wrong_number(self):
        result = self.ev.score("reasoning", "I think 41", "42")
        assert result["exact_match"] == 0.0

    def test_no_number_in_prediction(self):
        result = self.ev.score("reasoning", "I don't know", "42")
        assert result["exact_match"] == 0.0
