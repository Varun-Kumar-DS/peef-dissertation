# PROPOSAL CONTENT — Full Updated Draft
> All sections compiled with every change applied. Tell me what to adjust and I will update the Word document.

---

## TITLE PAGE

**Title:** Systematic Evaluation of Prompt Engineering Techniques for Large Language Models: A Multi-Task Empirical Framework

**Student:** Varun
**Student ID:** [Insert Student ID]
**Supervisor:** Achilleas Koufonikos
**Programme:** MSc Computer Science
**University:** University of Liverpool
**Date:** June 2026

---

## STATEMENT OF ETHICAL COMPLIANCE

**Data Category: A0**

This project does not collect personal data and does not involve human participants in any experimental activity. All datasets used are publicly available benchmarks released under open-source licences (Apache 2.0 and MIT). I confirm that this project will be carried out in full accordance with the University of Liverpool's ethical guidance for computing projects. Should any human evaluation activity be introduced later in the project, I will seek supervisor approval and update this statement before proceeding.

---

## SECTION 1 - PROJECT DESCRIPTION

Whether you have interacted with an AI assistant before or not, you have likely experienced that specific moment - when the way you phrase a question completely changes the quality of the answer you get back. Ask poorly and the response is vague. Ask well - with the right structure, the right context, the right examples - and the same model suddenly performs at a much higher level. This observation sits at the heart of a growing field called prompt engineering, and it raises a genuinely interesting research question: how can we tell which prompting approach will work best, and for which kind of task?

That is exactly what this project tries to answer. I will design and build the Prompt Engineering Evaluation Framework (PEEF) - a Python toolkit that runs controlled, repeatable experiments using Anthropic's Claude API, comparing three core prompting strategies across three different types of NLP tasks.

The three strategies are: zero-shot prompting, where the model receives no examples and relies entirely on its own training; few-shot prompting, where a handful of worked examples are included in the prompt to guide the model's response; and chain-of-thought (CoT) prompting, where the model is encouraged to reason step by step before giving a final answer. These are tested across question answering, text summarisation, and mathematical reasoning tasks, using well-established public datasets.

What makes this more than a simple comparison exercise is the dimension of cost-efficiency. Using an LLM through an API costs money - measured in tokens consumed - and a technique that performs 5% better but costs 40% more tokens is not always the right choice in practice. This project measures both performance and token usage to produce genuinely practical guidance for anyone deciding how to prompt a language model in a real application.

The end products will be: the PEEF software framework itself, a dataset of experimental results, and a set of clear guidelines giving evidence-backed advice on which technique to use for a given task. The aim is to transform prompt engineering from a gut feeling exercise to an evidence-based process.

---

## SECTION 2 - AIMS AND REQUIREMENTS

### 2.1 Aims

- Create PEEF: a clean, modular Python framework that leverages the Claude API to easily run, score, and compare prompt engineering experiments, and is extendable by others after this project concludes.
- Systematically investigate zero-shot, few-shot (2, 4, and 8 examples), and chain-of-thought prompting across question answering, summarisation, and reasoning tasks.
- Go beyond accuracy - evaluate output quality, consistency across repeated runs, and how performance changes with token cost, producing cost-efficiency curves for each technique.
- Translate findings into practical, task-specific guidelines that a developer or researcher can directly apply when deciding how to prompt a model.

### 2.2 Requirements

These requirements have been set in view of the project aims and consultation with the supervisor. They are listed in order of priority.

**Essential - these must be delivered:**

- **R1.** Zero-shot, few-shot (2-shot, 4-shot, 8-shot), and chain-of-thought prompting work out of the box with PEEF.
- **R2.** Experiments are conducted across all three task domains using the selected open-source benchmark datasets.
- **R3.** The correct metric is automatically applied for each task: exact-match accuracy for QA, ROUGE-L and BERTScore for summarisation, and step-level accuracy for reasoning.
- **R4.** All experiment results are saved with full context - prompt template, model version, temperature, and timestamp - so results can always be traced and reproduced.
- **R5.** The framework includes a statistics module that runs Wilcoxon signed-rank tests and reports effect sizes (Cohen's d), so conclusions are statistically supported rather than based on visual inspection of averages.
- **R6.** The codebase is fully documented - docstrings on every function, a README, and a short user guide.

**Desirable - delivered if time allows:**

- **R7.** An interactive Streamlit dashboard for visualising results, comparing techniques, filtering by task, and viewing significance test results.
- **R8.** Token cost tracking per run, with a viewable cost-efficiency scatter plot.
- **R9.** A prompt sensitivity module: assessing how sensitive model outputs are to small changes in prompt wording.
- **R10.** Extension to two additional techniques: self-consistency decoding and role-based prompting.

---

## SECTION 3 - KEY LITERATURE AND BACKGROUND READING

The modern tale of prompt engineering starts in 2020, when OpenAI released GPT-3 (Brown et al., 2020). What made this paper remarkable was not simply its scale, but that you could perform almost any task by writing a plain English instruction, with no fine-tuning required. This has been referred to as in-context learning, and it led to an explosion of research into just how sensitive these models are to the way they are prompted.

One of the most surprising findings from that wave was Kojima et al. (2022), whose results made zero-shot prompting dramatically more effective by simply appending the phrase "Let's think step by step" to a typical zero-shot prompt. Six words. No examples, no training. The result, which has been dubbed zero-shot chain-of-thought (ZS-CoT), indicated that the structure of a prompt can be just as important as its content.

Following this, Wei et al. (2022) introduced full chain-of-thought prompting by including intermediate reasoning steps in the few-shot examples. Rather than showing the model just an input-output pair, you show it an input, a worked reasoning process that leads to an answer, and then the output itself. The results were impressive on multi-step problems, especially in the subdomains of arithmetic and symbolic reasoning, and Wei et al. (2022b) later demonstrated that this capability only emerges in models above a certain scale. This scale-dependence directly justifies the choice of Claude, a frontier-scale model, as the experimental platform.

A further extension came from Wang et al. (2022), who introduced self-consistency. Instead of accepting the model's first answer, you sample multiple reasoning chains and select the most common one. This improves reliability but multiplies API costs - precisely the kind of trade-off this project is designed to quantify.

Beyond individual techniques, broader surveys of the field (Liu et al., 2023; Sahoo et al., 2024) have helped clarify the landscape. They distinguish between discrete prompting - which involves handcrafted natural language instructions and is the focus of this project - and soft prompting, which uses learnable token embeddings and requires access to model weights. Since Claude is a black-box API model, only discrete prompting is applicable here. White et al. (2023) take a more practical angle, proposing a catalogue of reusable prompt design patterns that closely resemble design patterns in software engineering. Several of these patterns will directly inform the prompt templates built into PEEF.

On the evaluation side, the challenge is that no single metric captures everything. ROUGE (Lin, 2004), which measures n-gram overlap between a generated summary and a reference, is the standard metric for summarisation, though it cannot capture perfectly valid paraphrases. To address this limitation, BERTScore (Zhang et al., 2020) uses contextualised word embeddings to measure semantic similarity - it understands that "automobile" and "car" mean the same thing, whereas ROUGE would count them as different. In Cobbe et al. (2021), for reasoning problems they use plain exact-match accuracy on final numerical answers in GSM8K, which is clean and unambiguous. This project deliberately combines all three approaches to get a fuller picture than any single metric could provide.

The datasets selected are the same ones used across most NLP research, as they are widely benchmarked, openly licensed, and represent genuinely different cognitive demands. TriviaQA (Joshi et al., 2017) tests factual recall across open-domain questions. CNN/DailyMail (See et al., 2017) tests the ability to generate a coherent summary of lengthy news articles. GSM8K (Cobbe et al., 2021) tests multi-step arithmetic reasoning through grade-school word problems. Together, they make it possible to say something meaningful about where each technique shines and where it falls short.

---

## SECTION 4 - DEVELOPMENT AND IMPLEMENTATION SUMMARY

### 4.1 Environment and Language

The project will be built using the Python programming language, the natural choice since Anthropic's SDK is centered on this language, and the majority of the libraries necessary (pandas, numpy, matplotlib, seaborn, rouge-score, bert-score) are part of the Python data science ecosystem. This project does not require a GPU as there is no model training - all heavy computation will take place in the cloud via the Claude API. Development will be done in VS Code, using Git version control, and pushing changes daily to GitHub as a backup. Production code will be in clean Python scripts with exploratory analysis carried out via Jupyter notebooks during the experiment phase.

### 4.2 How PEEF is Structured

PEEF is organised into four modules which each take on one clearly defined task:

- **Prompt Builder -** reads a task description and a collection of benchmark examples and combines this information to create a properly formatted prompt. Adopting Jinja2 templates means all three scenarios (zero-shot, k-shot, and CoT) share the same underlying logic without code duplication.
- **Experiment Runner -** handles the actual API calls. It manages rate limiting with exponential back-off if the API is busy, does not send the same prompt twice if it has already been run, and enables multiple experiments to run concurrently.
- **Evaluator -** is given the model's answer as well as the correct answer and provides a score. It knows which metric to use for each task type, so the caller never has to worry about that.
- **Analysis Engine -** takes a set of scored results and produces the statistics: means, confidence intervals, significance tests, effect sizes, and the charts that will appear in the dissertation.

### 4.3 Workflow Organisation

Each experiment is described in a YAML config file which specifies the dataset, technique, number of shots, model version, temperature, and output file location. This means any experiment can be reproduced exactly by anyone who has the same config file, without changing any code. Development will follow test-driven development - each module gets unit tests written before the implementation - which keeps the codebase reliable as it grows. The Streamlit dashboard (R7) will be built last, once the core pipeline is proven to work end-to-end.

---

## SECTION 5 - DATA SOURCES

Three publicly available datasets are used in this project, all free for academic research and carrying open-source licences:

| Dataset | Task | Licence | Sample Size | Source |
|---------|------|---------|-------------|--------|
| TriviaQA | Factual QA | Apache 2.0 | 500 questions | triviaqa.github.io |
| CNN/DailyMail | Summarisation | MIT | 300 articles | Hugging Face Datasets |
| GSM8K | Maths Reasoning | MIT | 1,319 problems | github.com/openai/grade-school-math |

None of these datasets contain personal or identifiable information. All data will be stored locally on a password-protected machine and will not be shared outside this project. Prompts sent to the Claude API are processed transiently and are not used by Anthropic for model training, in line with their standard API terms.

Samples are used for TriviaQA and CNN/DailyMail rather than full datasets to keep API costs and runtime manageable within the 12-week timeline. The sample sizes are large enough to draw reliable conclusions.

---

## SECTION 6 - TESTING AND EVALUATION

### 6.1 Technical Testing

Testing will be carried out in three stages. First, each individual module in PEEF will be tested on its own using pytest - checking that it handles normal inputs correctly and does not crash on unexpected ones like empty responses or API errors. Second, the full pipeline will be tested end-to-end to confirm that a single config file reliably produces a complete, scored results file. Third, a small set of experiments will be re-run to confirm that results are consistent across runs.

### 6.2 Research Evaluation

For each task, mean scores will be calculated for each prompting technique along with confidence intervals to show how reliable the results are. Statistical tests will be used to check whether the differences between techniques are genuine and not just down to chance. Effect sizes will also be reported to show whether a difference is practically meaningful, not just statistically significant. Cost-efficiency will be shown as a simple chart - accuracy on one axis, average tokens used on the other - so the trade-off between performance and cost is immediately visible.

---

## SECTION 7 - PROJECT ETHICS AND HUMAN PARTICIPANTS

This project involves no human participants at any stage. The data being processed consists entirely of benchmark questions and model-generated answers - no personal information is involved at any point. This project falls under Data Category A0 under the University of Liverpool ethical guidance.

If a small optional human evaluation is added later - for example asking a few fellow students to rate the quality of model summaries - this will be discussed with the supervisor first and the ethical statement updated before any activity begins. No deception, no sensitive data, and no vulnerable groups are involved at any stage.

---

## SECTION 8 - BCS PROJECT CRITERIA

| BCS Criterion | How This Project Meets It |
|---------------|--------------------------|
| Systematic understanding of knowledge at the forefront of the discipline | The literature review covers the field from its foundations through to recent advances, positioning the project within current academic debate on prompt engineering. |
| Comprehensive understanding of applicable techniques | Three prompting strategies are implemented and evaluated using multiple metrics, with statistical analysis applied to validate the findings. |
| Originality in the application of knowledge | PEEF is an original software tool. Adding cost-efficiency analysis alongside performance evaluation addresses a gap that most existing studies overlook. |
| Deal with complex issues systematically and creatively | The project handles real trade-offs between performance, cost, and feasibility, with deliberate design decisions made at every stage. Findings will be written for both technical and non-technical readers. |
| Self-direction and originality in tackling problems | The project is independently managed over 12 weeks, with the student making day-to-day research and implementation decisions autonomously. |
| Critical self-evaluation of the process | The dissertation will include a reflection chapter honestly assessing what worked, what did not, the limitations of the framework, and how far the findings can be generalised. |

---

## SECTION 9 - UI/UX MOCKUP

The PEEF dashboard will be a simple, clean Streamlit web application designed for a researcher who wants to explore results quickly. It has three areas: a left sidebar where you configure and launch an experiment; a main panel showing bar charts comparing technique performance across tasks with a results table below; and a right panel showing statistical test results and a cost vs. accuracy scatter plot.

The goal is not to build a polished consumer product - it is to make the experimental results easy to explore and present.

*[The Word document contains a generated wireframe diagram of this layout - Figure 1]*

---

## SECTION 10 - PROJECT PLAN

The project runs across 12 weeks divided into five phases. Writing starts at Week 6 alongside the experiments - not after them - so each section of the dissertation is drafted while the work is still fresh.

| Phase | Weeks | Dates | Key Activities |
|-------|-------|-------|---------------|
| Phase 1 - Foundation & Literature | 1-3 | 2-22 Jun | Literature review, project setup, API configuration |
| Phase 2 - Design & Pilot | 4-5 | 23 Jun - 6 Jul | PEEF build, pilot experiments, pipeline test |
| Phase 3 - Main Experiments | 6-8 | 7-27 Jul | Full experiment runs across all three tasks |
| Phase 4 - Analysis & Writing | 9-10 | 28 Jul - 10 Aug | Analysis, figures, dissertation chapter drafts |
| Phase 5 - Review & Submission | 11-12 | 11-25 Aug | Supervisor review, revisions, final submission |

*[The Word document contains the full Gantt chart - Figure 2]*

---

## SECTION 11 - RISKS AND CONTINGENCY PLANS

| Risk | Contingency | Likelihood | Impact |
|------|-------------|-----------|--------|
| API rate limits slow experiments | Use caching and back-off; run large experiments overnight | Medium | Medium |
| API costs go over budget | Use sampled datasets; reduce shot counts if needed | Medium | Medium |
| Dataset licensing issues | All three are verified open-source; fallback options identified | Low | High |
| Scope grows too large | Core requirements R1-R6 must be done before R7-R10 are touched | Medium | High |
| Hardware failure | Daily Git pushes to GitHub; data backed up to university OneDrive | Low | High |
| No significant difference found between techniques | A null result is still valid and will be reported honestly | Low | Medium |
| Supervisor unavailable at key point | Meetings booked one week ahead; decision log kept for async feedback | Low | Medium |
| Writing takes longer than expected | Writing starts at Week 6 alongside experiments, not after | Medium | High |

---

## SECTION 12 - REFERENCES

Anthropic (2024) *Claude API Documentation*. Available at: https://docs.anthropic.com [Accessed: June 2026].

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A. and Agarwal, S. (2020) 'Language models are few-shot learners', *Advances in Neural Information Processing Systems*, 33, pp. 1877-1901. Available at: https://arxiv.org/abs/2005.14165 [Accessed: June 2026].

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R. and Hesse, C. (2021) 'Training verifiers to solve math word problems', *arXiv preprint arXiv:2110.14168*. Available at: https://github.com/openai/grade-school-math [Accessed: June 2026].

Joshi, M., Choi, E., Weld, D. and Zettlemoyer, L. (2017) 'TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension', *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics*, pp. 1601-1611. Available at: https://aclanthology.org/P17-1147 [Accessed: June 2026].

Kojima, T., Gu, S.S., Reid, M., Matsuo, Y. and Iwasawa, Y. (2022) 'Large language models are zero-shot reasoners', *Advances in Neural Information Processing Systems*, 35, pp. 22199-22213. Available at: https://arxiv.org/abs/2205.11916 [Accessed: June 2026].

Lin, C.Y. (2004) 'ROUGE: A package for automatic evaluation of summaries', *Text Summarization Branches Out: Proceedings of the ACL-04 Workshop*, pp. 74-81. Available at: https://aclanthology.org/W04-1013 [Accessed: June 2026].

Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H. and Neubig, G. (2023) 'Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing', *ACM Computing Surveys*, 55(9), pp. 1-35. Available at: https://arxiv.org/abs/2107.13586 [Accessed: June 2026].

Sahoo, P., Singh, A.K., Saha, S., Jain, V., Mondal, S. and Chadha, A. (2024) 'A systematic survey of prompt engineering in large language models: Techniques and applications', *arXiv preprint arXiv:2402.07927*. Available at: https://arxiv.org/abs/2402.07927 [Accessed: June 2026].

See, A., Liu, P.J. and Manning, C.D. (2017) 'Get to the point: Summarization with pointer-generator networks', *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics*, pp. 1073-1083. Available at: https://aclanthology.org/P17-1099 [Accessed: June 2026].

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A. and Zhou, D. (2022) 'Self-consistency improves chain of thought reasoning in language models', *International Conference on Learning Representations (ICLR 2023)*. Available at: https://openreview.net/forum?id=1PL1NIMMrw [Accessed: June 2026].

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V. and Zhou, D. (2022) 'Chain-of-thought prompting elicits reasoning in large language models', *Advances in Neural Information Processing Systems*, 35, pp. 24824-24837. Available at: https://arxiv.org/abs/2201.11903 [Accessed: June 2026].

Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D. and Chi, E.H. (2022b) 'Emergent abilities of large language models', *Transactions on Machine Learning Research*. Available at: https://openreview.net/forum?id=yzkSU5zdwD [Accessed: June 2026].

White, J., Fu, Q., Hays, S., Sandborn, M., Olea, C., Gilbert, H., Elnashar, A., Spencer-Smith, J. and Schmidt, D.C. (2023) 'A prompt pattern catalog to enhance prompt engineering with ChatGPT', *arXiv preprint arXiv:2302.11382*. Available at: https://arxiv.org/abs/2302.11382 [Accessed: June 2026].

Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q. and Artzi, Y. (2020) 'BERTScore: Evaluating text generation with BERT', *International Conference on Learning Representations (ICLR 2020)*. Available at: https://openreview.net/forum?id=SkeHuCVFDr [Accessed: June 2026].

---
*Approximate word count (Sections 1-11, excluding tables and references): ~1,950 words*
*Tell me what to change and I will update the Word document.*
