# Reflection Memo — Agent-Assisted Research

**Project:** ISIC binary skin lesion classifier (benign vs. malignant)
**Author:** Patrick Gao (Northwestern STAT 390)
**Final result:** ROC-AUC 0.900 ± 0.005, recall 0.958 ± 0.019 across 5 reps (iters 32-36; 4/5 individually ≥ 0.95). Deployed checkpoint (iter 35 weights): AUC 0.9021, recall 0.9676 deterministically on every inference.
**Loop window:** 36 model iterations + 1 deployment-verification run, ~50 GPU-hours total

Five lessons, in roughly the order they bit me, with iteration numbers as
evidence so the claims are auditable against
[experiment_archive.md](experiment_archive.md).

---

## 1. The agent is only as good as your setup

The single biggest determinant of whether the agent produced useful work was
the structure I gave it *before* it started executing — the frozen contract
between `prepare.py`, `model.py`, and `run.py`; the requirement to log every
run to `results.tsv` with a status flag; the rule that one variable changes
per iteration; the `CLAUDE.md` block that named which files were editable.

Where that scaffolding existed, the agent ran cleanly for dozens of iterations
without supervision. Iters 14 onward, after the Week-4 protocol was written
down, are the cleanest stretch of the project: every row in `results.tsv` is
attributable to one variable, every multi-rep block was pre-declared, every
abandoned direction was committed with `--discard` rather than silently
deleted. The agent did not invent that discipline; it followed it because the
rules were explicit, in writing, and enforced by `run.py`'s logging.

Where the scaffolding was missing — early Block A, before the controlled-
experiment framework existed — the same agent confounded variables, claimed
single-run noise tails as wins, and would have stopped at iter 13 (recall
0.9555 on one run) if I hadn't externally imposed a replication rule. The
behavior change between Block A and Block B is not "the agent got smarter."
It's "the harness got better." When I designed `LOAD_CHECKPOINT` /
`SAVE_CHECKPOINT` in Week 7, the deployment work landed in two days because
the harness was already doing the right thing. When the harness was loose,
the same agent thrashed.

The implication for future agent-driven work is uncomfortable: most of the
real research effort goes into building the rules of the game *before*
turning the agent loose. The agent will execute whatever protocol you give
it, including a bad one, very quickly.

## 2. Early, undetected mistakes have serious future consequences

Iters 4, 5, and 7 each bundled two changes into one run. The combined gain
was the most impressive AUC jump of the early project (0.79 → 0.90).
Because of the confounding, I still don't know — and now cannot know — whether
`pos_weight=10` or the 224×224 upscale was the actual lever in iter 4.

That ambiguity propagated. The Week-4 controlled experiment (iters 10-13)
had to retroactively test `pos_weight` in isolation and found it within
noise; the upscale was never re-tested cleanly because by then it was
load-bearing for the architecture and we couldn't afford the GPU-hours to
unwind it. Four iterations of compute, plus a permanent unresolved hole
in the project's causal story, paid as tax for the early confounding.

Similar pattern with documentation: `CLAUDE.md` was written at iter 25,
the project ran to iter 36, and the umbrella doc went stale fast.
Every "what's the current locked config?" question in Week 7 required
re-deriving from `results.tsv` and `model.py`'s docstring because the
top-level reference had drifted. Small early gaps in record-keeping
became large late-stage research-archaeology costs.

The lesson is not "the agent should be more careful" — it's that the
*cost of catching a mistake* grows monotonically with how long it sits
undetected. A confounded iter caught the same day costs one rerun. A
confounded iter caught in Week 4 costs a 3-rep controlled experiment.
A confounded iter caught in Week 7 is unfixable. Forcing the check
*before commit* (a `run.py` pre-flight that diffs `model.py` against
the parent and rejects multi-variable changes) would have been worth
more than any of the actual model improvements.

## 3. An uninterpretable codebase is a research bottleneck

By iter 30 the `model.py` file had two-phase training, holdout threshold
calibration, 4-view TTA, an additive safety margin, a checkpoint
save/load path, a class-weight override, and an internal 128→224
upscale buried inside `_build_module`. Every one of these was added
for a defensible reason in the iteration that introduced it. Read
together at iter 36, the file is not interpretable without
`experiment_archive.md` open in another window.

This had concrete research costs. The determinism investigation
(iters 20-21) took two analysis passes — first blaming the unseeded
shuffle, then the augmentation ops — because the boundary between
"what `prepare.py` controls" and "what `model.py` can reach" was not
obvious from reading either file. The deployment work (Week 7)
required a careful audit of every place a tensor was reshaped or
normalized, because the ImageNet normalization happened inside the
`nn.Module`, the [0,1] scaling happened inside `prepare.py`, and the
NHWC→NCHW conversion happened inside `torch_adapter.py`. Three
files, three implicit contracts, no single place that documented the
end-to-end transform pipeline. The TFLite verification step
(`verify_predictions.py`) caught the mismatch only because we ran a
single image end-to-end and got bit-for-bit equality on the second
try, not the first.

The agent did not, of its own accord, refactor for clarity. It
added features. Each addition was locally sensible; the cumulative
effect was a file that worked correctly but read like sediment. A
human-imposed "refactor sweep at the end of each block" would have
caught this; nothing in the agent's natural incentives did.

## 4. Specific model choices have no reasoning attached unless you force it

`SAFETY_MARGIN=0.10`. `TARGET_RECALL=0.995`. `CALIBRATION_BATCHES=60`.
TTA with 4 views, not 2 or 8. Two-phase training: 3 epochs frozen,
7 epochs unfrozen, not 2/8 or 4/6.

Every one of these numbers is in `model.py` as a class constant.
Most of them were chosen by the agent during a single iteration,
ran, produced a number, and stayed. A few were ablated (TARGET_RECALL
at 0.95/0.99/0.995; CALIBRATION_BATCHES at 60/120). Most were not.
The defense in the locked-claim writeup is empirical: "this config
hits the targets." It is not "this number is principled because of
X." When a reviewer asks "why 0.10 and not 0.08 or 0.15?", the
honest answer is "we tried 0.10, it worked, we stopped." That is a
fine answer for a Week-7 deliverable. It is a bad answer if the
question is "would this generalize to a different dataset?"

The agent will happily propose specific numbers (0.10, 4 views,
60 batches) when asked. It does not, by default, distinguish
"this number is load-bearing and was ablated" from "this number is
arbitrary and survived because it wasn't worth ablating." Both end
up looking the same in `model.py`. Forcing the agent to annotate
each constant with `# ablated: range tested, decision basis` —
or refusing to land a new constant without that annotation —
would have surfaced the difference. We didn't, and the result is a
locked config whose specific values are defensible in aggregate
and arbitrary in detail.

## 5. Look at every edit before running the code

The agent will silently make changes that are not what you asked for.
This is not a malice problem; it's a "the agent's interpretation of
your prompt is not your interpretation of your prompt" problem.

Concrete examples from this project: iter 22 launched 15 epochs on
B4 because the prompt said "try a longer run" and the agent picked
15; the result was overfitting (recall 0.75) and a wasted 2.5
GPU-hours. Iter 9 bundled a threshold change with a class-weight
change because the agent's plan named both and I didn't read the
diff before saying "go." The 128→224 upscale that anchored the
entire B2/B4 line was introduced inside `_build_module` in a
multi-line edit that I approved without unfolding.

In every case, a 30-second diff review before launching `run.py`
would have caught the issue. Iter 22 would have become "10 epochs,
not 15." Iter 9 would have been split into two iters. The
upscale would have been a separate, attributable iteration with
its own controlled experiment. None of this required new tooling
— just the discipline of reading the patch before pressing run.

Late in the project, after Week 4, I started doing this consistently.
The Block-B confound rate dropped to zero. The agent did not change;
the review step did. Future agent-driven work should treat
"approve the diff, then approve the run" as two separate gates,
not one.

---

## Synthesis

The five lessons compound into one: **agent-assisted research is
mostly about the work you do around the agent, not the work the
agent does.** Set up the harness, catch mistakes the day they
happen, keep the codebase readable enough to reason about,
attach reasoning to every choice, and read every diff before
launching it. The agent will then run a long, disciplined search
faster than a human could, log it honestly, and not get bored.

When I gave the agent narrow, well-formed questions — "change
`pos_weight` from 10 to 20, run 2 reps, decide on the basis of
within-condition noise" — it was excellent. When I asked it
broader questions — "find a way to make recall reproducible" —
it generated plausible hypotheses, spent GPU time on them, and
produced clean negative results without ever stepping back to
ask whether the premise of the question was right. The frame-
check is the human's job. The execution is the agent's. Get the
division wrong and the speed advantage becomes a liability,
because plausible-looking dead ends generated quickly are still
dead ends.
