# Annotated Bibliography
## COMP702 — Prompt Engineering for AI Language Models
**Last updated:** June 2026

---

## How to use this file
Each entry follows the same structure:
- **Citation** — full reference in Harvard format
- **What it says** — a plain-English summary of the paper
- **Key finding** — the single most important result
- **Relevance to PEEF** — how this directly connects to your experiments
- **Limitations / caveats** — what the paper doesn't tell you
- **Quote to cite** — a short passage useful in the dissertation

---

## 1. Few-Shot Prompting & In-Context Learning

---

### Brown et al. (2020) ⭐ FOUNDATIONAL

**Citation:**
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A. and Agarwal, S. (2020) 'Language models are few-shot learners', *Advances in Neural Information Processing Systems*, 33, pp. 1877–1901. Available at: https://arxiv.org/abs/2005.14165

**What it says:**
This is the paper that introduced GPT-3 — a 175-billion-parameter autoregressive language model — and demonstrated that massive scale unlocks *in-context learning*: the ability to perform new tasks from only a handful of examples given in the prompt, with no gradient updates or fine-tuning required. Brown et al. tested GPT-3 across dozens of NLP tasks, distinguishing between zero-shot (no examples), one-shot (one example), and few-shot (several examples) settings.

**Key finding:**
Scaling up model size dramatically improves few-shot performance, sometimes matching or exceeding fine-tuned models on specific benchmarks. The number of in-context examples consistently helps, but returns diminish beyond a certain point.

**Relevance to PEEF:**
This paper establishes the core experimental distinction PEEF is built around — zero-shot vs. few-shot. The definition of few-shot prompting used in your experiments (providing k worked examples in the prompt without gradient updates) comes directly from this work. The finding that more examples help (up to a point) motivates your 2-shot / 4-shot / 8-shot comparison.

**Limitations / caveats:**
GPT-3 is the subject, not Claude. Results may not transfer directly. Also, this paper focuses on task performance without measuring token cost — which is a gap your project explicitly fills.

**Quote to cite:**
> "For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model."

---

## 2. Chain-of-Thought Prompting

---

### Wei et al. (2022) ⭐ FOUNDATIONAL

**Citation:**
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. and Zhou, D. (2022) 'Chain-of-thought prompting elicits reasoning in large language models', *Advances in Neural Information Processing Systems*, 35, pp. 24824–24837. Available at: https://arxiv.org/abs/2201.11903

**What it says:**
Wei et al. introduced *chain-of-thought (CoT) prompting* — a variant of few-shot prompting where the in-context examples include not just input-output pairs but also a step-by-step reasoning trace connecting them. The key insight is that by showing the model *how to think*, not just *what to answer*, you can elicit reasoning capabilities that standard few-shot prompting fails to activate.

**Key finding:**
CoT prompting produces striking gains on arithmetic, commonsense, and symbolic reasoning tasks. On GSM8K (the exact dataset used in PEEF), prompting a 540B-parameter model with just eight CoT examples achieves state-of-the-art accuracy, outperforming fine-tuned GPT-3. Crucially, these gains only emerge at sufficient model scale (roughly ≥100B parameters).

**Relevance to PEEF:**
This is the primary justification for including CoT as one of PEEF's three techniques. The paper also directly uses GSM8K as a benchmark — your reasoning task — making your results directly comparable to theirs. Wei et al.'s scale-dependence finding also justifies using Claude (a frontier-scale model) rather than a smaller model.

**Limitations / caveats:**
CoT requires carefully crafted reasoning examples, which introduces the risk of *prompt sensitivity* — small changes in the exemplar quality may produce large changes in output. Wei et al. do not measure token cost. Also, their experiments use Google's PaLM and LaMDA, not Claude.

**Quote to cite:**
> "We explore how generating a chain of thought — a series of intermediate reasoning steps — significantly improves the ability of large language models to perform complex reasoning."

---

### Kojima et al. (2022) ⭐ KEY

**Citation:**
Kojima, T., Gu, S.S., Reid, M., Matsuo, Y. and Iwasawa, Y. (2022) 'Large language models are zero-shot reasoners', *Advances in Neural Information Processing Systems*, 35, pp. 22199–22213. Available at: https://arxiv.org/abs/2205.11916

**What it says:**
Kojima et al. showed that you don't need few-shot examples to activate chain-of-thought reasoning. Simply appending the phrase **"Let's think step by step"** to a zero-shot prompt — what they call *Zero-shot-CoT* — dramatically improves performance on reasoning tasks. This is a surprising and practically important result: six words, no examples, and you get much better reasoning.

**Key finding:**
Zero-shot-CoT raises accuracy on MultiArith from 17.7% to 78.7% and on GSM8K from 10.4% to 40.7% using InstructGPT (text-davinci-002). Similar gains are observed on PaLM-540B. The same single prompt template works across diverse reasoning task types.

**Relevance to PEEF:**
This paper establishes `zero_shot_cot` as a meaningful intermediate technique between pure zero-shot and full few-shot CoT — and explains why PEEF includes it as a separate condition. The massive jump on GSM8K also sets a baseline for what your reasoning experiments should be able to replicate or compare against.

**Limitations / caveats:**
The "Let's think step by step" trigger phrase may be less effective on non-reasoning tasks (QA, summarisation) — your experiments can test this. Also, Kojima et al. study a two-stage prompting process (trigger phrase → extract answer), which is slightly different from standard single-pass CoT.

**Quote to cite:**
> "LLMs are decent zero-shot reasoners by simply adding 'Let's think step by step' before each answer."

---

## 3. Self-Consistency (Desirable Extension — R10)

---

### Wang et al. (2022)

**Citation:**
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A. and Zhou, D. (2022) 'Self-consistency improves chain of thought reasoning in language models', *International Conference on Learning Representations (ICLR 2023)*. Available at: https://openreview.net/forum?id=1PL1NIMMrw

**What it says:**
Rather than accepting the first output from a CoT prompt, Wang et al. proposed *self-consistency*: sample multiple diverse reasoning chains (e.g., 40 samples) and pick the most frequently occurring final answer by majority vote. This replaces the greedy single-chain decode with a more robust ensemble.

**Key finding:**
Self-consistency consistently outperforms standard CoT by a significant margin across arithmetic and commonsense reasoning benchmarks. On GSM8K, it achieves 74.4% accuracy compared to 56.5% for standard CoT with PaLM-540B.

**Relevance to PEEF:**
Self-consistency is listed as Requirement R10 (desirable extension). This paper is why — the performance improvement is real, but the cost is proportional to the number of samples drawn (e.g., 40× the tokens of a single CoT run). Your cost-efficiency analysis is the ideal place to quantify this trade-off explicitly.

**Limitations / caveats:**
40× token cost is prohibitive at scale. The gains depend on having a model capable of producing diverse reasoning paths — if all paths are identical, voting adds nothing.

---

## 4. Emergent Abilities (Scale Justification)

---

### Wei et al. (2022b)

**Citation:**
Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D. and Chi, E.H. (2022) 'Emergent abilities of large language models', *Transactions on Machine Learning Research*. Available at: https://openreview.net/forum?id=yzkSU5zdwD

**What it says:**
This paper characterises *emergent abilities* in LLMs: capabilities that are absent at smaller scales and appear sharply at larger scales, not predicted by smooth extrapolation of smaller-model performance. CoT reasoning is one of the abilities shown to be emergent.

**Key finding:**
Many abilities — including multi-step arithmetic and symbolic reasoning — are essentially absent in models below ~10B parameters but appear suddenly above certain scale thresholds. This is non-linear; you cannot predict frontier-model CoT performance from small-model results.

**Relevance to PEEF:**
This is the theoretical justification for why CoT should be tested on a frontier-scale model (Claude) rather than a smaller model. It also explains why you cannot benchmark your CoT results against small open-source models — the comparison is not meaningful.

---

## 5. Prompting Surveys (Landscape Overview)

---

### Liu et al. (2023)

**Citation:**
Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H. and Neubig, G. (2023) 'Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing', *ACM Computing Surveys*, 55(9), pp. 1–35. Available at: https://arxiv.org/abs/2107.13586

**What it says:**
A comprehensive taxonomy of prompting methods, distinguishing between discrete prompting (handcrafted natural language instructions — what PEEF uses) and soft/continuous prompting (learnable token embeddings — not applicable to black-box APIs like Claude). Also covers prompt templates, answer engineering, and multi-prompt learning.

**Key finding / relevance:**
The discrete vs. soft distinction directly justifies PEEF's scope. Since Claude is a black-box API model, only discrete prompting is possible. Cite this paper in your methodology section when explaining why soft prompting is out of scope.

---

### Sahoo et al. (2024)

**Citation:**
Sahoo, P., Singh, A.K., Saha, S., Jain, V., Mondal, S. and Chadha, A. (2024) 'A systematic survey of prompt engineering in large language models: Techniques and applications', *arXiv preprint arXiv:2402.07927*. Available at: https://arxiv.org/abs/2402.07927

**What it says:**
A more recent survey covering the broader landscape of prompt engineering techniques including CoT, zero-shot CoT, self-consistency, role prompting, and tree-of-thought. Useful for situating PEEF within the current state of the field.

**Relevance:**
Cite in the introduction and literature review to show awareness of the full landscape. The survey confirms that no single technique dominates across all task types — which is exactly the hypothesis PEEF is designed to test empirically.

---

## 6. Evaluation Metrics

---

### Lin (2004) — ROUGE

**Citation:**
Lin, C.Y. (2004) 'ROUGE: A package for automatic evaluation of summaries', *Text Summarization Branches Out: Proceedings of the ACL-04 Workshop*, pp. 74–81. Available at: https://aclanthology.org/W04-1013

**What it says:**
Introduced ROUGE (Recall-Oriented Understudy for Gisting Evaluation) — a set of metrics based on n-gram overlap between a generated summary and reference summaries. ROUGE-L uses the longest common subsequence rather than fixed n-grams.

**Key limitation:**
ROUGE counts word overlap, not meaning. A perfectly valid paraphrase scores 0 if it shares no words with the reference. This is why BERTScore is used alongside it in PEEF.

---

### Zhang et al. (2020) — BERTScore

**Citation:**
Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q. and Artzi, Y. (2020) 'BERTScore: Evaluating text generation with BERT', *International Conference on Learning Representations (ICLR 2020)*. Available at: https://openreview.net/forum?id=SkeHuCVFDr

**What it says:**
BERTScore computes similarity between generated and reference texts using contextualised embeddings from BERT, matching tokens by cosine similarity rather than exact string overlap. This means semantically equivalent words score as matches even if spelled differently.

**Relevance:**
Directly addresses ROUGE's main limitation. PEEF uses both ROUGE-L and BERTScore for summarisation to get complementary perspectives — surface overlap and semantic similarity.

---

## 7. Datasets

---

### Joshi et al. (2017) — TriviaQA

**Citation:**
Joshi, M., Choi, E., Weld, D. and Zettlemoyer, L. (2017) 'TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension', *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics*, pp. 1601–1611. Available at: https://aclanthology.org/P17-1147

**What it says:**
TriviaQA is a large-scale reading comprehension dataset with 650K question-answer-evidence triples, sourced from trivia websites and Wikipedia. The dataset is notable for its diversity and the fact that answers are often short factual strings.

**PEEF usage:** QA task — 200 sampled examples, validation split.

---

### Hermann et al. (2015) / See et al. (2017) — CNN/DailyMail

**Citation:**
Hermann, K.M., Kocisky, T., Grefenstette, E., Espeholt, L., Kay, W., Suleyman, M. and Blunsom, P. (2015) 'Teaching machines to read and comprehend', *Advances in Neural Information Processing Systems*, 28. [Original dataset paper]

See, A., Liu, P.J. and Manning, C.D. (2017) 'Get to the point: Summarization with pointer-generator networks', *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics*, pp. 1073–1083. Available at: https://aclanthology.org/P17-1099 [Standard benchmark version]

**What it says:**
CNN/DailyMail pairs news articles with human-written bullet-point summaries. See et al.'s version is the standard abstractive summarisation benchmark.

**PEEF usage:** Summarisation task — 100 sampled examples (fewer due to long input length and token cost).

---

### Cobbe et al. (2021) — GSM8K

**Citation:**
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R. and Hesse, C. (2021) 'Training verifiers to solve math word problems', *arXiv preprint arXiv:2110.14168*. Available at: https://github.com/openai/grade-school-math

**What it says:**
GSM8K (Grade School Math 8K) is a dataset of 8,500 linguistically diverse grade-school maths word problems requiring 2–8 reasoning steps to solve.

**Key finding:**
GPT-3 with no prompting technique achieves only ~10% on GSM8K. CoT prompting raises this dramatically — making it the ideal stress-test for chain-of-thought techniques.

**PEEF usage:** Reasoning task — 300 sampled examples. This is the hardest task for zero-shot and the one where CoT is expected to show the largest gain.

---

## Reading Log

| Paper | Read? | Notes added? | Saved to 01_research_papers/? |
|-------|-------|-------------|-------------------------------|
| Brown et al. (2020) | ✅ | ✅ | [ ] |
| Wei et al. (2022) CoT | ✅ | ✅ | [ ] |
| Kojima et al. (2022) | ✅ | ✅ | [ ] |
| Wang et al. (2022) Self-Consistency | [ ] | [ ] | [ ] |
| Wei et al. (2022b) Emergent | [ ] | [ ] | [ ] |
| Liu et al. (2023) Survey | [ ] | [ ] | [ ] |
| Sahoo et al. (2024) Survey | [ ] | [ ] | [ ] |
| Lin (2004) ROUGE | [ ] | [ ] | [ ] |
| Zhang et al. (2020) BERTScore | [ ] | [ ] | [ ] |
| Joshi et al. (2017) TriviaQA | [ ] | [ ] | [ ] |
| See et al. (2017) CNN/DM | [ ] | [ ] | [ ] |
| Cobbe et al. (2021) GSM8K | [ ] | [ ] | [ ] |
