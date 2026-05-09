"""
Week 5 deliverable #2 — Metric trajectory plot.

Reads results.tsv and renders three figures:
  * metric_trajectory.png — ROC-AUC and recall across all iterations of the
    autonomous block, with the controlled-experiment sub-blocks highlighted.
  * controlled_experiments.png — per-condition mean +/- std for the three
    completed controlled experiments (iters 10-13 pos_weight, iters 14-16
    holdout cal, iters 17-19 percentile estimator).
  * keep_discard_crash.png — bar chart of run-status counts for the block.

Run:
    python week5/make_plot.py
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
    "4: +224 +pw10",
    "5: B2 two-phase",
    "6: +cal (0.97/30)",
    "7: cal (0.995/60)",
    "8: +SAFETY 0.85",
    "9: cal (1.0/200)",
    "10: pw=10 A1",
    "11: pw=20 B1",
    "12: pw=10 A2",
    "13: pw=20 B2",
    "14: holdout 1",
    "15: holdout 2",
    "16: holdout 3",
    "17: pct 1",
    "18: pct 2",
    "19: pct 3",
]

# Mark the run-status of each iter (matches keep_discard_crash_summary.md)
STATUS = {
    1: "baseline",
    2: "keep", 3: "keep", 4: "keep", 5: "keep", 6: "keep", 7: "keep",
    8: "discard", 9: "discard",
    10: "keep", 11: "keep", 12: "keep", 13: "keep",
    14: "keep", 15: "keep", 16: "keep",
    17: "keep", 18: "keep", 19: "keep",
}

CONTROLLED_BLOCKS = {
    "pos_weight (Wk4)": (10, 13),
    "holdout cal (Wk5 p1)": (14, 16),
    "percentile estimator (Wk5 p1.5)": (17, 19),
}


def make_metric_trajectory(df):
    iters = list(range(1, len(df) + 1))
    auc = df["roc_auc"].to_numpy()
    recall = df["recall"].to_numpy()

    fig, (ax_auc, ax_rec) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    status_arr = np.array([STATUS.get(i, "keep") for i in iters])
    color_for = {"baseline": "#7f7f7f", "keep": "#1f77b4", "discard": "#d62728"}

    # AUC subplot
    ax_auc.plot(iters, auc, linewidth=1.5, color="#1f77b4", alpha=0.5, zorder=1)
    for status, color in color_for.items():
        mask = status_arr == status
        if mask.any():
            ax_auc.scatter(np.array(iters)[mask], auc[mask],
                           color=color, s=70, zorder=3, edgecolor="black",
                           label=status)
    ax_auc.axhline(0.85, color="red", linestyle="--", linewidth=1, label="target ≥ 0.85")
    ax_auc.set_ylabel("ROC-AUC")
    ax_auc.set_title("ROC-AUC trajectory across the autonomous block (iters 1–19)")
    ax_auc.set_ylim(0.75, 0.92)
    ax_auc.grid(alpha=0.3)
    ax_auc.legend(loc="lower right", fontsize=9)
    for x, y in zip(iters, auc):
        ax_auc.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)

    # Recall subplot
    ax_rec.plot(iters, recall, linewidth=1.5, color="#d62728", alpha=0.5, zorder=1)
    for status, color in color_for.items():
        mask = status_arr == status
        if mask.any():
            ax_rec.scatter(np.array(iters)[mask], recall[mask],
                           color=color, s=70, zorder=3, edgecolor="black",
                           label=status)
    ax_rec.axhline(0.95, color="red", linestyle="--", linewidth=1, label="target ≥ 0.95")
    ax_rec.set_ylabel("Recall")
    ax_rec.set_xlabel("Iteration")
    ax_rec.set_title("Recall trajectory across the autonomous block (iters 1–19)")
    ax_rec.set_ylim(0.70, 1.00)
    ax_rec.grid(alpha=0.3)
    ax_rec.legend(loc="lower right", fontsize=9)
    for x, y in zip(iters, recall):
        ax_rec.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7)

    # Shade the controlled-experiment sub-blocks on both axes
    block_colors = ["#fff2cc", "#dcedc8", "#cfe2f3"]
    for (label, (start, end)), bcolor in zip(CONTROLLED_BLOCKS.items(), block_colors):
        for ax in (ax_auc, ax_rec):
            ax.axvspan(start - 0.4, end + 0.4, color=bcolor, alpha=0.6, zorder=0)
        ax_auc.text((start + end) / 2, 0.755, f"controlled: {label}",
                    ha="center", fontsize=8, style="italic", color="#444")

    ax_rec.set_xticks(iters)
    ax_rec.set_xticklabels(ITER_LABELS, rotation=40, ha="right", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "metric_trajectory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_controlled_experiments(df):
    """Per-condition mean +/- std bars for the three completed controlled experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Experiment 1: pos_weight (Wk4) — iters 10-13
    pw10 = df.iloc[[9, 11]]   # iters 10, 12 (0-indexed 9, 11)
    pw20 = df.iloc[[10, 12]]  # iters 11, 13
    plot_controlled(axes[0],
                    "Wk4: pos_weight\n(iters 10–13, n=2 each)",
                    [("pw=10\nn=2", pw10), ("pw=20\nn=2", pw20)],
                    color=["#1f77b4", "#ff7f0e"])

    # Experiment 2: holdout cal (Wk5 priority 1) — iters 14-16 (holdout) vs iters 7,10,12 (leaked, same hyperparams)
    leaked = df.iloc[[6, 9, 11]]   # iters 7, 10, 12
    holdout = df.iloc[[13, 14, 15]] # iters 14, 15, 16
    plot_controlled(axes[1],
                    "Wk5 p1: holdout cal\n(leaked iters 7/10/12 vs holdout 14–16)",
                    [("leaked\nn=3", leaked), ("holdout\nn=3", holdout)],
                    color=["#1f77b4", "#ff7f0e"])

    # Experiment 3: percentile estimator (Wk5 corrected p1) — iters 14-16 vs 17-19, both holdout
    min_est = df.iloc[[13, 14, 15]]   # iters 14-16 (target 0.995 = min)
    pct_est = df.iloc[[16, 17, 18]]   # iters 17-19 (target 0.95 = 5th percentile)
    plot_controlled(axes[2],
                    "Wk5 p1.5: estimator\n(min iters 14–16 vs 5th-pct iters 17–19)",
                    [("min\nn=3", min_est), ("5th-pct\nn=3", pct_est)],
                    color=["#1f77b4", "#ff7f0e"])

    fig.suptitle(
        "Three controlled experiments in the autonomous block (one variable each, replicated)",
        fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out = OUT_DIR / "controlled_experiments.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_controlled(ax, title, conditions, color):
    """Plot recall mean +/- std for a 2-condition controlled experiment with raw points."""
    means = [c[1]["recall"].mean() for c in conditions]
    stds = [c[1]["recall"].std(ddof=1) for c in conditions]
    x = np.arange(len(conditions))

    ax.bar(x, means, yerr=stds, capsize=10, color=color,
           error_kw=dict(linewidth=2, ecolor="black"), alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Recall")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.70, 1.00)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1, alpha=0.7)

    # Plot raw points
    for xi, (label, df) in zip(x, conditions):
        ax.scatter([xi] * len(df), df["recall"], color="black", zorder=5, s=40, alpha=0.7)

    # Label each bar with mean +/- std
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.01, f"{m:.3f}\n±{s:.3f}",
                ha="center", va="bottom", fontsize=8)


def make_keep_discard_crash(df):
    counts = {
        "keep (committed)": sum(1 for s in STATUS.values() if s == "keep"),
        "baseline": sum(1 for s in STATUS.values() if s == "baseline"),
        "discard (reverted)": sum(1 for s in STATUS.values() if s == "discard"),
        "crash (worktree path issue, recovered)": 1,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#7f7f7f", "#d62728", "#ff7f0e"]
    labels = list(counts.keys())
    values = list(counts.values())

    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Run outcomes across the autonomous block (n=21 attempts)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) + 3)

    fig.tight_layout()
    out = OUT_DIR / "keep_discard_crash.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    df = pd.read_csv(RESULTS, sep="\t")
    if len(df) != len(ITER_LABELS):
        # Allow extra rows (iter 20+) that we don't have labels for yet
        if len(df) > len(ITER_LABELS):
            print(f"Note: results.tsv has {len(df)} rows, labels list has {len(ITER_LABELS)}. "
                  f"Plotting only the first {len(ITER_LABELS)} (iters 1-19).")
            df = df.iloc[:len(ITER_LABELS)].reset_index(drop=True)
        else:
            raise SystemExit(
                f"Row count mismatch: results.tsv has {len(df)} rows, "
                f"ITER_LABELS has {len(ITER_LABELS)}. Update the labels list."
            )

    make_metric_trajectory(df)
    make_controlled_experiments(df)
    make_keep_discard_crash(df)


if __name__ == "__main__":
    main()
