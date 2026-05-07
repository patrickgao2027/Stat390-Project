# Week 4 — Error Taxonomy

The four error categories from the Week 4 framework, applied to my actual loop with quantified evidence from the controlled experiment (iters 10–13). Naming the error type is the first step toward fixing it.

---

## 1. Signal Failure — *the loop runs, but no meaningful improvement appears across iterations*

**Severity in this loop: dominant.** The week-4 controlled experiment turned what was a qualitative observation into a measured number.

| Evidence | Number |
|----------|--------|
| Within-condition recall std at pw=10 (n=2 reps) | **0.012** |
| Within-condition recall std at pw=20 (n=2 reps) | **0.047** |
| Two reps of pw=20 produced recalls of | 0.888 and **0.955** (spread of 0.067) |
| Between-condition recall difference (pw=20 mean − pw=10 mean) | **+0.014** |
| Ratio (effect / pw=20 noise) | **0.30** — the candidate effect is *smaller than* the noise it sits in |
| Within-condition AUC std at pw=20 | 0.005 |
| Within-condition cal-threshold std at pw=20 | **0.171** (range: 0.317 to 0.558) |

**Why this is Signal Failure, not just bad luck:** when between-condition effect size is smaller than within-condition std, no single-run hyperparameter conclusion can be trusted. Two of my prior "discoveries" — iter 8's `SAFETY_MARGIN` regression and iter 9's larger-cal-sample regression — fall fully inside the noise band measured here and therefore cannot be attributed to the changes I made. They might be real effects; they might be the same noise that produced B1=0.888 and B2=0.955 at the same code.

**B2's recall = 0.9555 crossing the project's 0.95 target** is the most dramatic example: the run that *would* end the project is statistically indistinguishable from B1's failure at 0.888. Reporting this as a project win would be exactly the failure mode the PDF warns against — *"the score changed, but I'm not sure why."*

---

## 2. Code Instability — *crashes, inconsistent runs, broken pipeline*

**Severity: medium during week-3, fully resolved by week-4.** No new instabilities during the controlled experiment runs (iters 10–13 all completed cleanly).

| Incident | Resolution |
|----------|------------|
| Iter 7 first launch crashed immediately with `FileNotFoundError: 'challenge-2019-training_metadata_2026-04-22.csv'` because training was launched from inside a git worktree where the data CSVs were gitignored | Switched to running from the main project root; recorded a memory rule never to edit the worktree copy |
| For ~47 minutes during iter 7, the training log was 6 lines long with no per-epoch loss output — looked identical to a hung process | Diagnosed as Python stdout block-buffering when output is redirected to a file. Iter 8+ used `python -u` for unbuffered output |
| `model.py` was being edited in two places at once (worktree + main project) by two different Claude sessions; came close to overwriting in-progress ResNet-18 work | Caught before destructive `git checkout`; preference saved to memory |

---

## 3. Evaluation Leakage — *metric improves, but comparability is compromised*

**Severity: low for headline metrics, moderate for the threshold-tuning subsystem (and feeding into Signal Failure).**

| Where | What |
|-------|------|
| Headline ROC-AUC and recall | **Clean.** `prepare.py` does a stratified 80/20 split into train/test; `run.py` evaluates on the held-out test set with no augmentation. AUC is threshold-free so untouched by anything `predict()` does. All 13 iterations are directly comparable on the headline metrics. |
| Threshold-tuning Phase 3 (added in iter 6) | **Leaky by design.** Phase 3 calibrates the decision threshold by running inference on batches of *training data* — data the model has already memorized. Cal-set recall is therefore artificially 1.000 in every single run, and the resulting threshold is systematically too high for unseen test data. The 0.06–0.09 train→test recall gap is the direct symptom. |
| Feeding into Signal Failure | The leaky cal threshold magnifies between-run variance because the threshold is tightly coupled to the trained model's idiosyncratic probability distribution rather than to a stable validation signal. Iter 11 (pw=20 B1) put cal threshold at 0.558, iter 13 (pw=20 B2) put it at 0.317 — same code, std of 0.171 in probability space. |

**This is real Evaluation Leakage**, and it is mechanistically responsible for a significant share of the Signal Failure measured above. **Fix: hold out a 10 % slice of training data from `fit()` and use it for cal only — true out-of-sample threshold selection.**

---

## 4. Agent Misbehavior — *agent ignores rules or makes uncontrolled changes outside the intended scope*

**Severity: low–medium, mostly procedural and addressed in this taxonomy.**

| Incident | Why it's misbehavior |
|----------|----------------------|
| Iters 4, 5, 7, 9 each changed *two* variables in one run | Violates "exactly one variable changed" — confounded experiments. Caught and flagged in the matrix. |
| Iters 8 and 9 logged with `status=keep` in `results.tsv` despite both regressing | `run.py`'s default behavior when no `--discard` flag is passed. The matrix and this taxonomy correct the record. |
| Initial `model.py` edits made to a worktree copy that the trainer never read | Changes were "made" but had no effect; diagnosed mid-loop after the iter 7 crash. |
| In the autonomous loop, decision rule was applied locally per-iteration but the loop continued without acknowledging the underlying signal-failure issue surfaced in iter 8 | Should have stopped after iter 8 to question reproducibility before launching iter 9. The week-4 controlled experiment is the formal correction for this behavior. |

---

## Severity ranking and weekly takeaway

| Rank | Category | Severity | Status |
|------|----------|----------|--------|
| 1 | **Signal Failure** (between-run threshold variance, now quantified) | High | **Open — dominant failure mode** |
| 2 | **Evaluation Leakage** (cal uses training data) | Medium | Open — mechanistically feeds into #1 |
| 3 | **Agent Misbehavior** (multi-variable runs, keep-by-default) | Low–medium | Procedural; addressed by the matrix and this taxonomy |
| 4 | **Code Instability** (worktree, buffering) | Low | Resolved — none recurred during iters 10–13 |

The dominant failure is Signal Failure, and the week-4 controlled experiment **measured** it for the first time. Evaluation Leakage in the cal subsystem is the most likely upstream cause and is the highest-leverage week-5 fix. The Failure Analysis Memo covers this in depth.
