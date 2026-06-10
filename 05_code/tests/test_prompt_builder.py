"""Unit tests for PromptBuilder."""

import pytest
from src.prompt_builder import PromptBuilder


EXAMPLES = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What colour is the sky?", "answer": "Blue"},
]

COT_EXAMPLES = [
    {"problem": "Alice has 3 apples. She buys 2 more. How many does she have?",
     "working": "She starts with 3, buys 2 more: 3 + 2 = 5",
     "answer": "5"},
    {"problem": "A shop has 10 pens. It sells 4. How many remain?",
     "working": "10 - 4 = 6",
     "answer": "6"},
]


class TestZeroShot:
    def test_qa_prompt_contains_question(self):
        b = PromptBuilder(task="qa", technique="zero_shot")
        prompt = b.build(question="What is the boiling point of water?")
        assert "What is the boiling point of water?" in prompt
        assert "Answer:" in prompt

    def test_reasoning_zero_shot(self):
        b = PromptBuilder(task="reasoning", technique="zero_shot")
        prompt = b.build(problem="5 + 7 = ?")
        assert "5 + 7" in prompt


class TestFewShot:
    def test_qa_few_shot_includes_examples(self):
        b = PromptBuilder(task="qa", technique="few_shot", n_shots=2, examples=EXAMPLES)
        prompt = b.build(question="Where is the Eiffel Tower?")
        assert "What is the capital of France?" in prompt
        assert "Who wrote Hamlet?" in prompt
        # Should NOT include 3rd example
        assert "What is 2 + 2?" not in prompt

    def test_n_shots_respected(self):
        for k in (2, 4):
            b = PromptBuilder(task="qa", technique="few_shot", n_shots=k, examples=EXAMPLES)
            prompt = b.build(question="Test?")
            # Each example contributes one "Question:" line
            assert prompt.count("Question:") == k + 1  # k examples + 1 query


class TestCoT:
    def test_cot_reasoning_includes_working(self):
        b = PromptBuilder(task="reasoning", technique="cot", n_shots=2, examples=COT_EXAMPLES)
        prompt = b.build(problem="Bob has 5 cats and gives away 2. How many remain?")
        assert "Working:" in prompt
        assert "3 + 2 = 5" in prompt  # from first example

    def test_cot_qa_includes_reasoning(self):
        cot_qa = [
            {"question": "Why is the sky blue?",
             "reasoning": "Due to Rayleigh scattering of light.",
             "answer": "Rayleigh scattering"},
        ]
        b = PromptBuilder(task="qa", technique="cot", n_shots=1, examples=cot_qa)
        prompt = b.build(question="Why is grass green?")
        assert "Reasoning:" in prompt


class TestErrors:
    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            PromptBuilder(task="translation", technique="zero_shot")  # type: ignore

    def test_template_key(self):
        b = PromptBuilder(task="summarisation", technique="zero_shot")
        assert b.template_key == "summarisation_zero_shot"
