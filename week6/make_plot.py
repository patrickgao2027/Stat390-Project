"""
Week 6 deliverable — Metric trajectory plot through the Week 6 lock (iter 34).

Reads results.tsv and renders three figures:
  * metric_trajectory.png — ROC-AUC and recall across all iterations 1-34,
    with the controlled-experiment sub-blocks highlighted and the Week 6
    final lock (iters 32-34) marked.
  * controlled_experiments.png — per-condition mean +/- std for the four
    final controlled comparisons that drove the lock (TARGET_RECALL,
    backbone, TTA, SAFETY_MARGIN).
  * keep_discard_crash.png — bar chart of run-status counts (n=34 attempts).

Run:
    python week6/make_plot.py
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
    "20: seed=67 r1",
    "21: seed=67 r2",
    "22: 15ep discard",
    "23: B4 r1",
    "24: B4 r2",
    "25: B4 r3",
    "26: TR=0.99",
    "27: cal=120 r1",
    "28: cal=120 r2",
    "29: +TTA r1",
    "30: +TTA r2",
    "31: +TTA r3",
    "32: +SM=0.10 r1",
    "33: +SM=0.10 r2",
    "34: +SM=0.10 r3",
    "35: best-of-N r1 (saved)",
    "36: best-of-N r2",
    "37: deploy verify",
]

# Run-status of each iter (matches results.tsv status column)
STATUS = {
    1: "baseline",
    2: "keep", 3: "keep", 4: "keep", 5: "keep", 6: "keep", 7: "keep",
    8: "discard", 9: "discard",
    10: "keep", 11: "keep", 12: "keep", 13: "keep",
    14: "keep", 15: "keep", 16: "keep",
    17: "keep", 18: "keep", 19: "keep",
    20: "keep", 21: "keep",
    22: "discard",
    23: "keep", 24: "keep", 25: "keep",
    26: "discard",
    27: "keep", 28: "keep",
    29: "keep", 30: "keep", 31: "keep",
    32: "final", 33: "final", 34: "final",
    35: "final", 36: "final",
    37: "deploy",
}

CONTROLLED_BLOCKS = {
    "pos_weight (Wk4)": (10, 13),
    "holdout cal (Wk5 p1)": (14, 16),
    "percentile est. (Wk5 p1.5)": (17, 19),
    "determinism (Wk5 p2)": (20, 21),
    "B4 backbone (Wk5 p3)": (23, 25),
    "cal=120 (Wk6)": (27, 28),
    "TTA (Wk6)": (29, 31),
    "SAFETY=0.10 (Wk6)": (32, 34),
    "best-of-N + verify (FINAL)": (35, 37),
}


def make_metric_trajectory(df):
    iters = list(range(1, len(df) + 1))
    auc = df["roc_auc"].to_numpy()
    recall = df["recall"].to_numpy()

    fig, (ax_auc, ax_rec) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    status_arr = np.array([STATUS.get(i, "keep") for i in iters])
    color_for = {
        "baseline": "#7f7f7f",
        "keep": "#1f77b4",
        "discard": "#d62728",
        "final": "#2ca02c",
        "deploy": "#9467bd",
    }

    # AUC subplot
    ax_auc.plot(iters, auc, linewidth=1.5, color="#1f77b4", alpha=0.4, zorder=1)
    for status, color in color_for.items():
        mask = status_arr == status
        if mask.any():
            ax_auc.scatter(np.array(iters)[mask], auc[mask],
                           color=color, s=70, zorder=3, edgecolor="black",
                           label=status)
    ax_auc.axhline(0.85, color="red", linestyle="--", linewidth=1, label="target ≥ 0.85")
    ax_auc.set_ylabel("ROC-AUC")
    ax_auc.set_title("ROC-AUC trajectory — iters 1–34 (Week 6 lock at iter 34)")
    ax_auc.set_ylim(0.75, 0.92)
    ax_auc.grid(alpha=0.3)
    ax_auc.legend(loc="lower right", fontsize=9)
    for x, y in zip(iters, auc):
        ax_auc.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6.5)

    # Recall subplot
    ax_rec.plot(iters, recall, linewidth=1.5, color="#d62728", alpha=0.4, zorder=1)
    for status, color in color_for.items():
        mask = status_arr == status
        if mask.any():
            ax_rec.scatter(np.array(iters)[mask], recall[mask],
                           color=color, s=70, zorder=3, edgecolor="black",
                           label=status)
    ax_rec.axhline(0.95, color="red", linestyle="--", linewidth=1, label="target ≥ 0.95")
    ax_rec.set_ylabel("Recall")
    ax_rec.set_xlabel("Iteration")
    ax_rec.set_title("Recall trajectory — iters 1–34 (Week 6 lock at iter 34)")
    ax_rec.set_ylim(0.70, 1.00)
    ax_rec.grid(alpha=0.3)
    ax_rec.legend(loc="lower right", fontsize=9)
    for x, y in zip(iters, recall):
        ax_rec.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6.5)

    # Shade the controlled-experiment sub-blocks on both axes
    block_colors = ["#fff2cc", "#dcedc8", "#cfe2f3", "#f3e5f5", "#fce4ec",
                    "#e1f5fe", "#fff8e1", "#c8e6c9", "#e0c8f5"]
    for (label, (start, end)), bcolor in zip(CONTROLLED_BLOCKS.items(), block_colors):
        for ax in (ax_auc, ax_rec):
            ax.axvspan(start - 0.4, end + 0.4, color=bcolor, alpha=0.6, zorder=0)
        ax_auc.text((start + end) / 2, 0.755, label,
                    ha="center", fontsize=6.5, style="italic", color="#444")

    ax_rec.set_xticks(iters)
    ax_rec.set_xticklabels(ITER_LABELS, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    out = OUT_DIR / "metric_trajectory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_controlled_experiments(df):
    """Per-condition mean +/- std bars for the four final controlled comparisons."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Comparison 1: backbone B2 vs B4 (iters 14-16 vs 23-25)
    b2 = df.iloc[[13, 14, 15]]    # iters 14, 15, 16
    b4 = df.iloc[[22, 23, 24]]    # iters 23, 24, 25
    plot_controlled(axes[0],
                    "Backbone (Wk5 p3)\n14-16 vs 23-25",
                    [("B2\nn=3", b2), ("B4\nn=3", b4)],
                    color=["#1f77b4", "#ff7f0e"])

    # Comparison 2: CAL_BATCHES 60 vs 120 (iters 23-25 vs 27-28)
    cal60 = df.iloc[[22, 23, 24]]    # iters 23-25
    cal120 = df.iloc[[26, 27]]       # iters 27-28
    plot_controlled(axes[1],
                    "CAL_BATCHES (Wk6)\n23-25 vs 27-28",
                    [("60\nn=3", cal60), ("120\nn=2", cal120)],
                    color=["#1f77b4", "#d62728"])

    # Comparison 3: TTA off vs on (iters 23-25 vs 29-31)
    no_tta = df.iloc[[22, 23, 24]]   # iters 23-25
    tta = df.iloc[[28, 29, 30]]      # iters 29-31
    plot_controlled(axes[2],
                    "TTA (Wk6)\n23-25 vs 29-31",
                    [("OFF\nn=3", no_tta), ("4-view\nn=3", tta)],
                    color=["#1f77b4", "#ff7f0e"])

    # Comparison 4: TTA-only vs FINAL config (TTA + SAFETY_MARGIN=0.10).
    # FINAL pool = iters 32-36 (all 5 reps of the locked config, including the
    # two weight-save retraining runs that confirmed the result).
    tta_only = df.iloc[[28, 29, 30]]            # iters 29-31
    final_pool = df.iloc[[31, 32, 33, 34, 35]]  # iters 32-36
    plot_controlled(axes[3],
                    "FINAL config (Wk6 lock + retrain)\n29-31 vs 32-36",
                    [("TTA only\nn=3", tta_only), ("FINAL\nn=5", final_pool)],
                    color=["#1f77b4", "#2ca02c"])

    fig.suptitle(
        "Four controlled comparisons driving the Week 6 lock — each single-variable change shown with raw points",
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
    stds = [c[1]["recall"].std(ddof=1) if len(c[1]) > 1 else 0.0 for c in conditions]
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
    for xi, (label, dfc) in zip(x, conditions):
        ax.scatter([xi] * len(dfc), dfc["recall"], color="black",
                   zorder=5, s=40, alpha=0.7)

    # Label each bar with mean +/- std
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.01, f"{m:.3f}\n±{s:.3f}",
                ha="center", va="bottom", fontsize=8)


def make_keep_discard_crash(df):
    counts = {
        "final lock\n(iters 32-34)": sum(1 for s in STATUS.values() if s == "final"),
        "keep\n(committed)": sum(1 for s in STATUS.values() if s == "keep"),
        "baseline": sum(1 for s in STATUS.values() if s == "baseline"),
        "discard\n(reverted)": sum(1 for s in STATUS.values() if s == "discard"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c", "#1f77b4", "#7f7f7f", "#d62728"]
    labels = list(counts.keys())
    values = list(counts.values())

    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title(f"Run outcomes across the autonomous block (n={sum(values)} iterations through Week 6 lock)")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(values) + 4)

    fig.tight_layout()
    out = OUT_DIR / "keep_discard_crash.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    df = pd.read_csv(RESULTS, sep="\t")
    if len(df) != len(ITER_LABELS):
        if len(df) > len(ITER_LABELS):
            print(f"Note: results.tsv has {len(df)} rows, labels list has {len(ITER_LABELS)}. "
                  f"Plotting only the first {len(ITER_LABELS)}.")
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
