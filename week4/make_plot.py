"""
Week 4 deliverable #3 — Metric-over-time plot + controlled-experiment plot.

Renders two figures:
  * metric_over_time.png — ROC-AUC and recall across all 13 iterations,
    with success thresholds dashed and the controlled-experiment runs
    (iters 10-13) highlighted.
  * controlled_experiment.png — per-condition mean +/- std bars for the
    pos_weight controlled experiment (iters 10-13), the actual evidence
    base for week 4.

Run:
    python week4/make_plot.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results.tsv"
OUT_DIR = Path(__file__).resolve().parent

ITER_LABELS = [
    "1: LR baseline",
    "2: AlexNet",
    "3: EffNet-B0",
    "4: +224, +pw10",
    "5: B2, two-phase",
    "6: +cal (0.97/30)",
    "7: cal (0.995/60)",
    "8: +SAFETY 0.85",
    "9: cal (1.0/200)",
    "10: A1 pw=10",
    "11: B1 pw=20",
    "12: A2 pw=10",
    "13: B2 pw=20",
]

CONTROLLED_ITERS = {10, 11, 12, 13}  # highlight on metric-over-time plot


def make_metric_over_time(df):
    iters = list(range(1, len(df) + 1))
    auc = df["roc_auc"].to_numpy()
    recall = df["recall"].to_numpy()

    fig, (ax_auc, ax_rec) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Highlight controlled-experiment runs in a different color
    is_controlled = np.array([i in CONTROLLED_ITERS for i in iters])

    ax_auc.plot(iters, auc, linewidth=2, color="#1f77b4", zorder=1)
    ax_auc.scatter(np.array(iters)[~is_controlled], auc[~is_controlled],
                   color="#1f77b4", zorder=2, s=50, label="exploratory iter")
    ax_auc.scatter(np.array(iters)[is_controlled], auc[is_controlled],
                   color="#ff7f0e", zorder=3, s=80, marker="s",
                   label="controlled exp (iters 10-13)")
    ax_auc.axhline(0.85, color="red", linestyle="--", linewidth=1, label="target >= 0.85")
    ax_auc.set_ylabel("ROC-AUC")
    ax_auc.set_title("ROC-AUC over iterations (target met since iter 2)")
    ax_auc.set_ylim(0.75, 0.95)
    ax_auc.grid(alpha=0.3)
    ax_auc.legend(loc="lower right", fontsize=9)
    for x, y in zip(iters, auc):
        ax_auc.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)

    ax_rec.plot(iters, recall, linewidth=2, color="#d62728", zorder=1)
    ax_rec.scatter(np.array(iters)[~is_controlled], recall[~is_controlled],
                   color="#d62728", zorder=2, s=50, label="exploratory iter")
    ax_rec.scatter(np.array(iters)[is_controlled], recall[is_controlled],
                   color="#ff7f0e", zorder=3, s=80, marker="s",
                   label="controlled exp (iters 10-13)")
    ax_rec.axhline(0.95, color="red", linestyle="--", linewidth=1, label="target >= 0.95")
    ax_rec.set_ylabel("Recall")
    ax_rec.set_xlabel("Iteration")
    ax_rec.set_title("Recall over iterations (target crossed once at iter 13 — not reproducible)")
    ax_rec.set_ylim(0.70, 1.00)
    ax_rec.grid(alpha=0.3)
    ax_rec.legend(loc="lower right", fontsize=9)
    for x, y in zip(iters, recall):
        ax_rec.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)

    ax_rec.set_xticks(iters)
    ax_rec.set_xticklabels(ITER_LABELS, rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "metric_over_time.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_controlled_experiment_plot(df):
    # Iters 10-13 are 0-indexed rows 9-12 in df (after iter 1 baseline at row 0)
    controlled = df.iloc[9:13].reset_index(drop=True)
    pw10 = controlled[controlled["description"].str.contains("pos_weight=10")]
    pw20 = controlled[controlled["description"].str.contains("pos_weight=20")]

    metrics = ["roc_auc", "recall", "precision"]
    titles = ["ROC-AUC", "Recall", "Precision"]
    targets = {"roc_auc": 0.85, "recall": 0.95}

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        means = [pw10[metric].mean(), pw20[metric].mean()]
        stds = [pw10[metric].std(ddof=1), pw20[metric].std(ddof=1)]

        x = np.arange(2)
        bars = ax.bar(x, means, yerr=stds, capsize=10,
                      color=["#1f77b4", "#ff7f0e"],
                      error_kw=dict(linewidth=2, ecolor="black"))
        ax.set_xticks(x)
        ax.set_xticklabels(["pw=10\n(n=2)", "pw=20\n(n=2)"])
        ax.set_title(f"{title}\nmean +/- std, n=2 reps per condition")
        ax.grid(axis="y", alpha=0.3)

        # Plot raw points to show the spread visibly
        ax.scatter([0]*len(pw10), pw10[metric], color="black", zorder=5, s=50, alpha=0.7)
        ax.scatter([1]*len(pw20), pw20[metric], color="black", zorder=5, s=50, alpha=0.7)

        # Target line for AUC and recall
        if metric in targets:
            ax.axhline(targets[metric], color="red", linestyle="--",
                       linewidth=1, label=f"target >= {targets[metric]}")
            ax.legend(loc="lower right", fontsize=8)

        # Label the bars with the actual numbers
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + s + 0.005, f"{m:.3f}\n+/- {s:.3f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Week 4 controlled experiment: pos_weight effect on test metrics\n"
        "between-condition recall diff (+0.014) is smaller than within-condition std at pw=20 (0.047) "
        "-> Signal Failure dominant",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    out = OUT_DIR / "controlled_experiment.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    df = pd.read_csv(RESULTS, sep="\t")
    if len(df) != len(ITER_LABELS):
        raise SystemExit(
            f"Row count mismatch: results.tsv has {len(df)} rows, "
            f"ITER_LABELS has {len(ITER_LABELS)}. Update the labels list."
        )
    make_metric_over_time(df)
    make_controlled_experiment_plot(df)


if __name__ == "__main__":
    main()
