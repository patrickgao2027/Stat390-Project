"""

Run one experiment: build model, train, evaluate, log result.

Usage:
    python run.py "description"              # logs as status=keep
    python run.py "description" --baseline   # logs as status=baseline
    python run.py "description" --discard    # logs as status=discard
"""

import sys
import time
import subprocess
from prepare import load_data, evaluate, plot_model_performance
from model import build_model


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"

def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    ### Load data finish editing

    train_ds, test_ds, steps_per_epoch, val_steps, class_weight, y_test = load_data()

    model = build_model()
    print(f"Model: {model}")

    # 3. Train
    t0 = time.time()
    model.fit(train_ds, epochs=10, steps_per_epoch=steps_per_epoch, class_weight=class_weight)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    # 4. Evaluate
    roc_auc, accuracy, recall, precision = evaluate(model, test_ds, y_test)
    print(f"ROC-AUC: {roc_auc:.4f} | Recall: {recall:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f}")

    # 5. Log
    commit = get_git_hash()
    log_result(commit, roc_auc, recall, status, description)
    print(f"Result logged to results.tsv (status={status})")

if __name__ == "__main__":
    main()