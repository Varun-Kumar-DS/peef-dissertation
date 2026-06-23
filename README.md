# Prompt Engineering for AI Language Models

**MSc Dissertation — University of Liverpool**
**Student:** Varun (varunzayne@gmail.com)
**Supervisor:** Achilleas Koufonikos
**Timeline:** June – August 2026 (12 weeks)

---

## Overview

This project investigates how large language models (LLMs) can be used most effectively through systematic evaluation of prompt engineering techniques. The central research question is: **which prompting strategies produce the best outcomes across different task types?**

The study compares three core prompting paradigms — zero-shot, few-shot, and chain-of-thought (CoT) — across tasks spanning question answering, summarisation, and logical reasoning. All experiments are conducted via Anthropic's Claude API using Python.

---

## Research Questions

1. How do zero-shot, few-shot, and chain-of-thought prompting compare in accuracy and quality across task types?
2. Does the optimal prompting technique vary by task domain (QA vs. summarisation vs. reasoning)?
3. What design principles generalise across techniques to improve LLM performance?
4. **(Novel contribution)** Can prompts be improved *automatically* from their own failure cases, and can per-instance adaptive technique selection match chain-of-thought accuracy at a fraction of its token cost?

---

## Project Structure

```
.
├── 01_research_papers/
│   ├── foundational/          # Seminal LLM & NLP papers
│   ├── prompting_techniques/  # Papers on zero-shot, few-shot, CoT, etc.
│   ├── evaluation_methods/    # Papers on benchmarking & eval frameworks
│   └── llm_benchmarks/        # Benchmark datasets and leaderboards
│
├── 02_experiments/
│   ├── zero_shot/             # Zero-shot prompting experiments
│   ├── few_shot/              # Few-shot prompting experiments
│   ├── chain_of_thought/      # Chain-of-thought experiments
│   ├── combined_techniques/   # Hybrid / ablation experiments
│   └── notebooks/             # Exploratory Jupyter notebooks
│
├── 03_results/
│   ├── raw/                   # Raw API outputs and logs
│   ├── processed/             # Cleaned, scored result tables
│   └── figures/               # Charts and visualisations
│
├── 04_evaluation/
│   ├── metrics/               # Scoring scripts (ROUGE, BERTScore, accuracy, etc.)
│   ├── benchmarks/            # Benchmark task definitions and datasets
│   └── human_eval/            # Human evaluation rubrics and responses
│
├── 05_code/
│   ├── src/                   # Core modules (API client, prompt builders, runners)
│   ├── tests/                 # Unit and integration tests
│   ├── utils/                 # Helpers (logging, file I/O, config parsing)
│   └── config/                # API keys (gitignored), model configs
│
├── 06_writeup/
│   ├── chapters/              # Dissertation chapter drafts
│   ├── drafts/                # Full document drafts and supervisor feedback
│   ├── references/            # BibTeX / Zotero exports
│   └── figures/               # Publication-quality figures for the dissertation
│
├── README.md                  # This file
└── progress_tracker.md        # 12-week milestone tracker
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM API | Anthropic Claude API |
| Language | Python 3.10+ |
| Notebooks | Jupyter |
| Evaluation | ROUGE, BERTScore, custom accuracy metrics |
| Data handling | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Version control | Git |

---

## Setup

1. **Clone / open this project folder** in your preferred IDE.
2. **Install dependencies:**
   ```bash
   pip install anthropic pandas numpy matplotlib seaborn rouge-score bert-score jupyter
   ```
3. **Add your API key** — create `05_code/config/.env`:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```
   > ⚠️ Never commit `.env` to version control. Add it to `.gitignore`.
4. **Run a quick test:**
   ```bash
   python 05_code/src/test_connection.py
   ```

---

## Prompting Techniques Under Study

**Zero-shot** — The model receives only the task instruction, no examples. Tests the model's in-built reasoning with no guidance.

**Few-shot** — The prompt includes a small number of worked examples (typically 2–8) before the target query. Measures how demonstration context improves performance.

**Chain-of-Thought (CoT)** — The prompt encourages the model to reason step-by-step before giving a final answer. Particularly effective for multi-step reasoning tasks.

---

## Novel Contributions (PEEF Modules 5 & 6)

Beyond the static technique comparison, the framework contributes two adaptive methods:

**Module 5 — Reflexive Prompt Optimizer** (`05_code/src/prompt_optimizer.py`)
A closed-loop *evaluate → reflect → mutate → select* engine. The LLM diagnoses its own dev-set failures, proposes rewritten templates, and a candidate replaces the champion **only if it wins a paired bootstrap significance test (p < α)** — a statistical promotion gate that prevents dev-set noise from being mistaken for improvement (a guard typically absent from prior automatic-prompting work such as APE and OPRO). Every critique, candidate, score, and promotion decision is written to a reproducible audit-trail JSON, with a full token/cost ledger and hard budget cap.

**Module 6 — Adaptive Cascade** (`05_code/src/adaptive_cascade.py`)
Instead of asking "which technique is best on average?", the cascade decides **per question**: it samples a cheap zero-shot prompt *k* times, uses self-consistency agreement as a confidence estimate, and escalates to chain-of-thought only when the model disagrees with itself. The confidence threshold is a tunable accuracy/cost dial, and a replay utility derives the full escalation-rate curve without extra API calls.

Both modules accept an injectable `call_fn`, so the entire test suite (`05_code/tests/`) runs offline with no API key. End-to-end demo: `02_experiments/combined_techniques/run_adaptive_experiments.py`.

---

## Task Domains

- **Question Answering** — Factual and inferential QA (e.g., TriviaQA, SQuAD subsets)
- **Summarisation** — Abstractive and extractive summarisation (e.g., CNN/DailyMail subsets)
- **Reasoning** — Logical and arithmetic reasoning (e.g., GSM8K, BIG-Bench subsets)

---

## Key References

- Brown et al. (2020) — GPT-3: Language Models are Few-Shot Learners
- Wei et al. (2022) — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- Kojima et al. (2022) — Large Language Models are Zero-Shot Reasoners
- White et al. (2023) — A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT
- Anthropic Claude API Documentation — https://docs.anthropic.com

---

## Contact

Supervisor: Achilleas Koufonikos — University of Liverpool
Student: Varun — varunzayne@gmail.com
