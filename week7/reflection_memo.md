# Reflection Memo — Agent-Assisted Research

**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Author:** Patrick Gao (Northwestern STAT 390)
**Final result:** ROC-AUC 0.903 ± 0.001, recall 0.952 ± 0.023 across 3 reps (iters 32-34)
**Loop window:** 36 model iterations + 1 deployment-verification run, ~50 GPU-hours total

The Week 7 brief asks five things of this memo: what the agent did well, what it
did poorly, what required human judgment, how I would redesign the loop, and
overall what I learned about doing research with AI agents. I answer each in
order, with specific iteration numbers as evidence so the claims are auditable
against [experiment_archive.md](experiment_archive.md).

---

## 1. What the agent did well

**Disciplined single-variable iteration once given the protocol.** After Week 4
established the "change exactly one thing per iter, replicate before deciding"
rule, the agent honored it for the rest of the project — every iteration from 14
onward names a single controlled variable in its `results.tsv` description
and runs in pre-declared 3-rep blocks (14-16, 17-19, 23-25, 27-28, 29-31, 32-34).
This is the *one* discipline failure of the early block (iters 4, 5, 7, 9 are all
marked "confounded" in the archive) and the *one* discipline that the agent
internalized and never broke again. The cleanest demonstration: when iter 26
moved a single variable and regressed, the rep was abandoned after 1 run (not
3), the reversion was committed with `--discard`, and the next experiment moved
on without sunk-cost defending the choice.

**Rapid execution of mechanical follow-throughs.** Once a decision was made,
the agent was very fast at the mechanical parts: edit one line in `model.py`,
launch `python -u run.py "iter N description"`, append the row to `results.tsv`,
draft the commit message, push to GitHub. The end-to-end "iter idea → committed
result" overhead was a few minutes of human time per iteration (most of the
wall clock was GPU training). At 36 iterations, that compression mattered.

**Honest writeup of negative results.** The Week-5 "what actually worked" memo
labels 6 of 10 modification classes as failures and explains each one's mechanism
(multiplicative SAFETY_MARGIN regressed because the baseline it multiplied was
noisy; full RNG seeding failed because `prepare.py` shuffles before `model.py`
imports; etc.). Week 6's ablation table marks the abandoned levers explicitly.
The agent did not paper over the abandoned experiments to make the project look
cleaner — it preserved them as part of the trace.

**Catching its own confounds.** Iters 4, 5, 7, and 9 changed two variables at
once. In every case, the agent flagged the run as "confounded" in its own
writeup (see `experiment_archive.md`) and explicitly noted that the gain could
not be attributed. It did not silently claim credit for a confounded win.

## 2. What the agent did poorly

**Early-block confounding.** Iters 4, 5, and 7 each bundled two changes, and
their combined gain was the most impressive AUC jump of the early project
(0.79 → 0.90). Because of the confounding, I genuinely don't know whether
the `pos_weight=10` or the 224×224 upscale was the actual lever in iter 4.
The Week-4 controlled experiment (iters 10-13) eventually went back and
tested `pos_weight` in isolation and found it within noise — but that's
4 retroactive iters of compute paid as the tax for the original sin.

**Over-trusting noise tails as signal.** Iter 13 produced recall 0.9555 (the
first single run to clear 0.95). Iter 15 produced recall 0.9650. Both were
noise tails — sibling runs in the same condition came in at 0.888 and 0.892
respectively. The agent initially read these as the recall target being
"basically met" before the Week-4 framework forced the rule that single-run
recall in [0.93, 0.97] is within-condition noise and not a real win. Without
the explicit replication discipline, the project could have stopped at iter 13
with a false claim.

**Pursuing the wrong root cause for recall variance.** The Week-5 priority-1
fix was "remove cal-set leakage" (iters 14-16). It was wrong — the dominant
variance source was the noisy `min` order statistic, not leakage. The agent
chased the leakage hypothesis through 3 reps before reading its own data and
pivoting to the estimator-swap experiment (iters 17-19), which produced the
real variance fix. ~5 hours of GPU time spent on the wrong hypothesis.

**Treating local optimization as the goal.** The agent kept finding "one more
lever" to push recall higher: TARGET_RECALL, CALIBRATION_BATCHES,
SAFETY_MARGIN, TTA. Each one had a plausible theoretical story. Most regressed
or sat inside within-condition noise (see the table in §4 of
[final_results_table.md](final_results_table.md)). The Week 6 scope-lock brief
had to externally impose "stop searching" — left to its own devices the agent
would have kept proposing levers indefinitely, because the marginal cost of
proposing was lower than the marginal cost of stopping.

**Documentation drift.** `CLAUDE.md` was anchored to iter 25 even though the
project was at iter 34 + deployment by Week 7. The agent updated `model.py`
docstring and per-week deliverable folders religiously but didn't update the
top-level CLAUDE.md until prompted. Information stayed current in the most
recent artifact and got stale in the older umbrella documents — a real
maintenance pattern in any agent loop.

## 3. What required human judgment (was irreplaceable)

**Setting the success criteria.** AUC ≥ 0.85 and recall ≥ 0.95 are not
defaults the agent invented — I set them because the medical screening
context demands recall over precision (a missed cancer is worse than a
false alarm). An agent left to optimize AUC alone would have stopped at
iter 7 (AUC 0.90, recall 0.94) and called it done. The recall target is
what kept the project iterating into B4 + TTA + SAFETY_MARGIN territory.

**Calling scope lock at iter 34.** Both targets were met; the next plausible
lever (an ensemble) would have required ~17 GPU-hours and a serialization
wrapper, and gained at most ~0.005 AUC. The decision to stop was a human
judgment about marginal returns vs. presentation/writeup time, framed by
the Week 6 brief. The agent's natural impulse was to keep exploring.

**Diagnosing the determinism dead-end.** When iters 20-21 produced different
results despite identical seeding, the question "why didn't this work?" had
two plausible answers: (a) the seeding code is incomplete, (b) something
upstream of `model.py` is non-deterministic. The agent's first instinct was
to add more seeding (NumPy, hash seed, etc.). It took human inspection to
diagnose that `prepare.py` runs `tf.data.shuffle()` before `model.py` is
imported, which is why no amount of seeding in `model.py` would ever fix
the issue. This is the *one* hard constraint in the project, and identifying
it required reading the frozen `prepare.py` source and reasoning about
import ordering — not something the agent surfaced unprompted.

**Choosing what to ship vs. what to discard.** The deployment-mode decision —
save best-of-N weights, ship the canonical checkpoint, accept that per-run
variance is structural — was a product-decision judgment, not a research
optimization. The agent built it well once the call was made, but the call
itself ("we are done iterating; we are shipping") was outside the loop.

**Editing the editable surface only.** The constraint that `prepare.py` and
`run.py` are frozen — and the related constraint that the worktree's
`model.py` edits don't reach the trainer because data is gitignored —
were rules I had to enforce repeatedly. The agent didn't violate them
in the end, but it required clear, repeated, written constraints (in
`CLAUDE.md` and memory) to stay inside the box.

## 4. How I would redesign the loop

**Force replication before commit.** Iters 8, 9, and 22 were logged as `keep`
(per `run.py`'s default) and only manually reverted after the fact. A better
loop would refuse to commit `model.py` until a 2-rep replication confirms the
direction of effect. The Week-4 retrospective added the human discipline; the
*tooling* still allows single-rep "wins."

**Inline confound detection.** A trivial linter on the `description` string
("how many distinct variable changes does this name?") would have caught
iters 4, 5, 7, and 9 before they ran. The Week-4 reframe ("change exactly
one thing per iter") came after the fact; better would be a `run.py` pre-flight
that diffs `model.py` against the parent commit and refuses if more than one
constant or one class changed.

**Stronger separation of "evaluator" from "tuner."** Test metrics were logged
every iteration. They never directly drove a model change (decisions used
holdout cal), but having test results in front of me on every row created
pressure to peek. A better loop would write test metrics to a file that is
mode-locked or sealed during exploration, and only unsealed at scope lock.
The brief's "test set opened only once" rule is hard to honor when the
default tooling makes the result immediately visible.

**Pre-declared stop conditions.** Most of the late iters (26, 27-28, 29-31,
32-34) were proposed in the spirit of "let's try one more thing." A
pre-declared stop condition ("once 3 reps cross both targets, lock; do not
propose new levers") would have stopped the loop at iter 25 if it had been
declared earlier, or at iter 31 with a wash on recall and a kept AUC win.
The Week 6 brief eventually imposed this externally.

**Deployment as a first-class loop output, not a stretch.** The deployment
pipeline (`SAVE_CHECKPOINT`/`LOAD_CHECKPOINT`, ONNX export, TFLite, Android)
landed in Week 7 after the lever search closed. With hindsight, building
the save/load determinism path *before* iter 14 would have made the
"per-run variance" problem mostly irrelevant — we'd have been training
candidates and shipping the best-of-N checkpoint from iter 6 onward.
Treating training-time variance as the central problem (which 11 iterations
attacked) was misframed; it became a deployment problem in the end.

## 5. What I learned about doing research with AI agents

The agent is a fast, tireless, honest **executor of a protocol I write**.
It is not, and should not be, the source of the protocol. Every actual
research decision in this project — what to count as a target, what to
treat as noise, when to stop, what to ship — came from a human reading
the brief, looking at the numbers, and writing a rule. The agent's job
was to follow the rule mechanically across many trials without losing
focus, and it did that well.

When I tried to outsource the *judgment* to the agent (e.g., "find a way to
make recall reproducible" early in Block B), the agent generated a plausible
hypothesis (seeding everything), spent 5 hours of GPU time on it, and
produced a clean negative result. It did not, of its own accord, step back
and ask "is the premise of this question correct?" That kind of frame-check
needs a human in the loop. When I gave the agent narrow, well-formed
questions ("change `pos_weight` from 10 to 20, run 2 reps, decide on the
basis of within-condition noise"), it was excellent.

The honest summary: **the agent compressed the wall-clock cost of running
each experiment by maybe 10×, and held the *discipline* of the protocol
better than a tired graduate student would. But the protocol itself —
the choice of what to ask, how to ask it, and when to stop — was
irreplaceably human.** A successful agent-assisted research loop is not
"the agent does the research"; it's "the human structures the search and
the agent executes it cleanly." Get the structure right and the agent is
a force multiplier. Get it wrong and the agent is a fast generator of
plausible-looking dead ends.
