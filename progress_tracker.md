# 12-Week Progress Tracker
## MSc Dissertation: Prompt Engineering for AI Language Models

**Start date:** 2 June 2026
**Submission target:** 25 August 2026
**Supervisor:** Achilleas Koufonikos

Progress key: `[ ]` Not started · `[~]` In progress · `[x]` Complete

---

## Phase 1 — Foundation & Literature (Weeks 1–3)

### Week 1 · 2–8 Jun 2026 — Project Setup & Orientation
- [ ] Set up project folder structure and Git repository
- [ ] Configure Claude API access and test connection
- [ ] Define initial research questions with supervisor
- [ ] Start reading foundational LLM papers (GPT-3, PaLM, Claude)
- [ ] Draft project proposal / introduction outline

**Supervisor check-in:** ___________
**Notes:**

---

### Week 2 · 9–15 Jun 2026 — Literature Review: Prompting Techniques
- [ ] Read and annotate zero-shot prompting papers
- [ ] Read and annotate few-shot prompting papers (Brown et al. 2020)
- [ ] Read and annotate chain-of-thought papers (Wei et al. 2022, Kojima et al. 2022)
- [ ] Begin literature review chapter draft (Section 2.1 — Prompting overview)
- [ ] Build annotated bibliography

**Supervisor check-in:** ___________
**Notes:**

---

### Week 3 · 16–22 Jun 2026 — Literature Review: Evaluation & Benchmarks
- [ ] Read papers on LLM evaluation frameworks and metrics
- [ ] Survey benchmark datasets (GSM8K, TriviaQA, CNN/DM, BIG-Bench)
- [ ] Complete literature review chapter draft
- [ ] Finalise research questions and hypotheses
- [ ] **Milestone: Literature review first draft complete**

**Supervisor check-in:** ___________
**Notes:**

---

## Phase 2 — Experimental Design & Pilot (Weeks 4–5)

### Week 4 · 23–29 Jun 2026 — Evaluation Framework Design
- [ ] Design evaluation metrics for each task type (accuracy, ROUGE, BERTScore)
- [ ] Select and prepare benchmark datasets (sample/subset for experiments)
- [ ] Write dataset loading and preprocessing utilities (`05_code/utils/`)
- [ ] Design prompt templates for all three techniques
- [ ] Write methodology chapter draft (experimental design section)

**Supervisor check-in:** ___________
**Notes:**

---

### Week 5 · 30 Jun – 6 Jul 2026 — Pilot Experiments
- [ ] Build core API client and experiment runner (`05_code/src/`)
- [ ] Run pilot zero-shot experiments on QA task
- [ ] Run pilot few-shot experiments on QA task (2-shot, 4-shot, 8-shot)
- [ ] Run pilot CoT experiments on QA task
- [ ] Sanity-check results, debug API calls, tune rate limiting
- [ ] **Milestone: End-to-end experiment pipeline working**

**Supervisor check-in:** ___________
**Notes:**

---

## Phase 3 — Main Experiments (Weeks 6–8)

### Week 6 · 7–13 Jul 2026 — QA Experiments (Full Run)
- [ ] Run full zero-shot QA experiments
- [ ] Run full few-shot QA experiments (varying shot counts)
- [ ] Run full CoT QA experiments
- [ ] Log all results to `03_results/raw/`
- [ ] Process and score results; save to `03_results/processed/`

**Supervisor check-in:** ___________
**Notes:**

---

### Week 7 · 14–20 Jul 2026 — Summarisation Experiments
- [ ] Run zero-shot, few-shot, CoT experiments on summarisation task
- [ ] Evaluate with ROUGE-1, ROUGE-2, ROUGE-L, BERTScore
- [ ] Compare technique performance on summarisation
- [ ] Produce preliminary figures for summarisation results

**Supervisor check-in:** ___________
**Notes:**

---

### Week 8 · 21–27 Jul 2026 — Reasoning Experiments
- [ ] Run zero-shot, few-shot, CoT experiments on reasoning task (GSM8K)
- [ ] Evaluate with accuracy and step-level correctness
- [ ] Compare technique performance on reasoning
- [ ] Run any ablation or combined-technique experiments
- [ ] **Milestone: All main experiments complete — raw results collected**

**Supervisor check-in:** ___________
**Notes:**

---

## Phase 4 — Analysis & Writing (Weeks 9–10)

### Week 9 · 28 Jul – 3 Aug 2026 — Results Analysis
- [ ] Aggregate and compare results across all tasks and techniques
- [ ] Produce final figures and tables (`03_results/figures/`)
- [ ] Statistical analysis (significance tests where applicable)
- [ ] Write results chapter draft
- [ ] Identify key findings and anomalies to discuss

**Supervisor check-in:** ___________
**Notes:**

---

### Week 10 · 4–10 Aug 2026 — Discussion & Contributions
- [ ] Write discussion chapter (interpret results, link to literature)
- [ ] Articulate contributions and limitations
- [ ] Draft conclusion chapter
- [ ] Integrate all chapters into a single dissertation document
- [ ] **Milestone: Full dissertation first draft complete**

**Supervisor check-in:** ___________
**Notes:**

---

## Phase 5 — Review & Submission (Weeks 11–12)

### Week 11 · 11–17 Aug 2026 — Supervisor Review & Revisions
- [ ] Submit full draft to supervisor for feedback
- [ ] Revise based on supervisor comments
- [ ] Proofread for clarity, grammar, and consistency
- [ ] Verify all citations and references (BibTeX / Zotero)
- [ ] Check formatting meets University of Liverpool dissertation guidelines

**Supervisor check-in:** ___________
**Notes:**

---

### Week 12 · 18–25 Aug 2026 — Final Polish & Submission
- [ ] Final proofread pass
- [ ] Verify all figures, tables, and appendices are correct
- [ ] Confirm code and data are archived / submitted if required
- [ ] Submit dissertation by deadline (25 Aug 2026)
- [ ] **🎓 Milestone: SUBMISSION COMPLETE**

**Supervisor check-in:** ___________
**Notes:**

---

## Summary Dashboard

| Phase | Weeks | Status |
|-------|-------|--------|
| Foundation & Literature | 1–3 | `[ ]` |
| Experimental Design & Pilot | 4–5 | `[ ]` |
| Main Experiments | 6–8 | `[ ]` |
| Analysis & Writing | 9–10 | `[ ]` |
| Review & Submission | 11–12 | `[ ]` |

---

## Supervisor Meeting Log

| Date | Topics Discussed | Actions |
|------|-----------------|---------|
| | | |
| | | |
| | | |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| API rate limits slow experiments | Medium | Medium | Batch requests; cache results |
| Dataset licensing issues | Low | High | Use open-licensed benchmarks only |
| Scope creep | Medium | High | Stick to 3 techniques × 3 tasks |
| Writer's block on chapters | Medium | Medium | Write alongside experiments, not after |
| Supervisor unavailability | Low | Medium | Schedule meetings 1 week in advance |
