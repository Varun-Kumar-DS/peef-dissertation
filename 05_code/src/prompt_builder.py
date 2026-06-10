"""
Prompt Builder — PEEF Module 1
================================
Constructs formatted prompts for zero-shot, few-shot, and chain-of-thought
experiments using Jinja2 templates. All three prompt types share the same
underlying logic; the template selects the appropriate format.

Usage:
    builder = PromptBuilder(task="qa", technique="few_shot", n_shots=4)
    prompt = builder.build(example={"question": "...", "answer": "..."})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any
from jinja2 import Environment, BaseLoader

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Technique = Literal["zero_shot", "few_shot", "cot", "zero_shot_cot"]
Task = Literal["qa", "summarisation", "reasoning"]


# ---------------------------------------------------------------------------
# Built-in Jinja2 templates
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, str] = {
    # ── Question Answering ──────────────────────────────────────────────────
    "qa_zero_shot": """\
Answer the following question. Give only the answer, no explanation.

Question: {{ question }}
Answer:""",

    "qa_zero_shot_cot": """\
Answer the following question. Think step by step before giving your final answer.

Question: {{ question }}
Answer:""",

    "qa_few_shot": """\
Answer each question. Give only the answer, no explanation.
{% for ex in examples %}
Question: {{ ex.question }}
Answer: {{ ex.answer }}
{% endfor %}
Question: {{ question }}
Answer:""",

    "qa_cot": """\
Answer each question by thinking step by step, then give the final answer.
{% for ex in examples %}
Question: {{ ex.question }}
Reasoning: {{ ex.reasoning }}
Answer: {{ ex.answer }}
{% endfor %}
Question: {{ question }}
Reasoning:""",

    # ── Summarisation ───────────────────────────────────────────────────────
    "summarisation_zero_shot": """\
Summarise the following article in 2-3 sentences.

Article: {{ article }}
Summary:""",

    "summarisation_zero_shot_cot": """\
Summarise the following article in 2-3 sentences. First identify the key points, then write the summary.

Article: {{ article }}
Summary:""",

    "summarisation_few_shot": """\
Summarise each article in 2-3 sentences.
{% for ex in examples %}
Article: {{ ex.article }}
Summary: {{ ex.summary }}
{% endfor %}
Article: {{ article }}
Summary:""",

    "summarisation_cot": """\
Summarise each article. First identify the main points, then write a 2-3 sentence summary.
{% for ex in examples %}
Article: {{ ex.article }}
Key points: {{ ex.key_points }}
Summary: {{ ex.summary }}
{% endfor %}
Article: {{ article }}
Key points:""",

    # ── Mathematical Reasoning (GSM8K) ──────────────────────────────────────
    "reasoning_zero_shot": """\
Solve the following maths problem. Give only the final numerical answer.

Problem: {{ problem }}
Answer:""",

    "reasoning_zero_shot_cot": """\
Solve the following maths problem. Think step by step.

Problem: {{ problem }}
Answer:""",

    "reasoning_few_shot": """\
Solve each maths problem. Give only the final numerical answer.
{% for ex in examples %}
Problem: {{ ex.problem }}
Answer: {{ ex.answer }}
{% endfor %}
Problem: {{ problem }}
Answer:""",

    "reasoning_cot": """\
Solve each maths problem by showing your working step by step, then state the final answer.
{% for ex in examples %}
Problem: {{ ex.problem }}
Working: {{ ex.working }}
Answer: {{ ex.answer }}
{% endfor %}
Problem: {{ problem }}
Working:""",
}


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

@dataclass
class PromptBuilder:
    """
    Builds prompts for a given task and technique.

    Parameters
    ----------
    task : Task
        One of "qa", "summarisation", "reasoning".
    technique : Technique
        One of "zero_shot", "few_shot", "cot", "zero_shot_cot".
    n_shots : int
        Number of few-shot examples (only used when technique="few_shot" or "cot").
    examples : list[dict]
        Pool of examples to draw from for few-shot/CoT prompts. The first
        n_shots are used (shuffle before passing if you want randomness).
    custom_template : str | None
        Optional Jinja2 template string to override the built-in one.
    """

    task: Task
    technique: Technique
    n_shots: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)
    custom_template: str | None = None

    def __post_init__(self) -> None:
        self._env = Environment(loader=BaseLoader())
        key = f"{self.task}_{self.technique}"
        raw = self.custom_template or _TEMPLATES.get(key)
        if raw is None:
            raise ValueError(
                f"No template found for task='{self.task}', technique='{self.technique}'. "
                f"Available keys: {list(_TEMPLATES)}"
            )
        self._template = self._env.from_string(raw)

    def build(self, **kwargs: Any) -> str:
        """
        Render the prompt.

        Pass the input fields as keyword arguments, e.g.:
            builder.build(question="What is the capital of France?")

        For few-shot / CoT prompts the 'examples' field is automatically
        populated from self.examples[:self.n_shots]; you do not need to
        pass it here.
        """
        context = dict(kwargs)
        if self.technique in ("few_shot", "cot") and "examples" not in context:
            context["examples"] = self.examples[: self.n_shots]
        return self._template.render(**context).strip()

    @property
    def template_key(self) -> str:
        return f"{self.task}_{self.technique}"
