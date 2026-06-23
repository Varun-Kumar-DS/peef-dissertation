# Key Paper Summaries — Prompting Techniques
> For use in the literature review chapter. Read the full papers for deeper detail.

---

## 1. Brown et al. (2020) — Language Models are Few-Shot Learners
**Link:** https://arxiv.org/abs/2005.14165

### What they did
Introduced GPT-3, a 175 billion parameter language model trained on web text. The key finding was that simply scaling up a model produces in-context learning — the ability to perform tasks from a handful of examples in the prompt, with no fine-tuning required.

### Key findings
- Few-shot prompting (providing 1-100 examples in the prompt) can match or approach fine-tuned models on many tasks
- Performance scales with model size — larger models make better use of few-shot examples
- Works across translation, QA, arithmetic, and text generation without any task-specific training
- Introduced the zero-shot / one-shot / few-shot terminology now standard in the field

### Why it matters for your project
This paper is the foundation for everything else. It established that prompting with examples works, and it defined the vocabulary (zero-shot, few-shot) that your project is built around.

### Key quote
> "GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation."

---

## 2. Wei et al. (2022) — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
**Link:** https://arxiv.org/abs/2201.11903

### What they did
Extended few-shot prompting by including intermediate reasoning steps in the examples — instead of showing input → answer, they showed input → reasoning steps → answer. Tested across arithmetic, commonsense, and symbolic reasoning tasks.

### Key findings
- CoT dramatically improves performance on complex reasoning tasks — GSM8K accuracy went from 17.7% to 78.7% on large models
- CoT is an emergent ability — it only helps models above ~100 billion parameters (smaller models perform the same or worse)
- Eight worked examples with reasoning chains was the optimal few-shot count tested
- The gains are largest on the hardest problems (where baseline performance is lowest)

### Why it matters for your project
Directly justifies your choice to use Claude (a frontier model) and to test CoT alongside few-shot. The scale-dependency finding means you need a large model — Claude qualifies. Your GSM8K experiments will replicate this finding.

### Key quote
> "Chain-of-thought prompting is an emergent ability of model scale — it does not positively impact performance for small models."

---

## 3. Kojima et al. (2022) — Large Language Models are Zero-Shot Reasoners
**Link:** https://arxiv.org/abs/2205.11916

### What they did
Showed that adding a single phrase — "Let's think step by step" — to a zero-shot prompt dramatically improves reasoning performance, with no examples needed at all. Called this Zero-Shot CoT (ZS-CoT).

### Key findings
- Six words ("Let's think step by step") unlocked multi-step reasoning without any examples
- Outperformed standard zero-shot on 4 out of 6 arithmetic tasks and all symbolic and logical reasoning tasks
- On some tasks went from ~10% accuracy to 40-70% with just this one addition
- Worked across GPT-3 and PaLM without any task-specific tuning

### Why it matters for your project
This justifies your zero_shot_cot technique in PEEF. It shows that the structure of a prompt (not just its content) matters — which is the central claim of your dissertation. It also shows CoT isn't exclusive to few-shot settings.

### Key quote
> "Large language models are capable of zero-shot multi-step reasoning without examples — simply through an effective prompt."

---

## Papers Still to Read (Week 3)

- [ ] Wang et al. (2022) — Self-consistency — https://arxiv.org/abs/2203.11171
- [ ] Liu et al. (2023) — Pre-train, prompt, predict survey — https://arxiv.org/abs/2107.13586
- [ ] Sahoo et al. (2024) — Systematic survey of prompt engineering — https://arxiv.org/abs/2402.07927
- [ ] White et al. (2023) — Prompt pattern catalog — https://arxiv.org/abs/2302.11382
- [ ] Zhang et al. (2020) — BERTScore — https://openreview.net/forum?id=SkeHuCVFDr
- [ ] Lin (2004) — ROUGE — https://aclanthology.org/W04-1013
- [ ] Cobbe et al. (2021) — GSM8K dataset — https://github.com/openai/grade-school-math
