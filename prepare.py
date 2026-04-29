"""
FROZEN -- Do not modify this file.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv 
import PIL
import PIL.Image
import tensorflow as tf

### Vars
RANDOM_STATE = 67
RESULTS_FILE = "results.tsv"
###


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, test_ds, y_test):
    x_only  = test_ds.map(lambda img, lbl: img)
    y_pred  = model.predict(x_only)
    y_proba = model.predict_proba(x_only)[:, 1]
    mainmetric = roc_auc_score(y_test, y_proba)
    accuracy   = accuracy_score(y_test, y_pred)
    recall     = recall_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred)
    return mainmetric, accuracy, recall, precision


# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, roc_auc, recall, status, description):
    """Append one row to results.tsv."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "roc_auc", "recall", "status", "description"])
        writer.writerow([experiment_id, f"{roc_auc:.6f}", f"{recall:.6f}", status, description])


# ── Plotting ───────────────────────────────────────────────
def plot_model_performance(model, x_test, y_test):
    # 1. Calculate Data
    y_probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc, accuracy, recall, precision = evaluate(model, x_test, y_test)
    
    # 2. Setup Plot
    plt.figure(figsize=(8, 6))
    
    # Plot ROC Curve
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f"ROC curve (AUC = {roc_auc:.2f})")
    
    # Plot Baseline (Random Guess)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 3. Aesthetics
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Analysis')
    
    # Add metrics summary as a text box
    stats_text = (f"Accuracy: {accuracy:.2f}\n"
                  f"Precision: {precision:.2f}\n"
                  f"Recall: {recall:.2f}")
    
    plt.gca().text(0.6, 0.2, stats_text, style='italic',
                   bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

# ── Paths ───────────────────────────────────────────────────
IMG_DIR_2019_TRAIN = r"C:\Users\Owner\Documents\Stat390-Project\ISIC-images 2019 train"
IMG_DIR_2020_TRAIN = r"C:\Users\Owner\Documents\Stat390-Project\ISIC-images train 2020"
IMG_DIR_2019_TEST  = r"C:\Users\Owner\Documents\Stat390-Project\ISIC-images 2019 test"
IMG_DIR_2020_TEST  = r"C:\Users\Owner\Documents\Stat390-Project\ISIC-images 2020 test"
IMG_SIZE           = (128, 128)
BATCH_SIZE         = 16
AUTOTUNE           = tf.data.AUTOTUNE

def load_data():
    """
    Uses the pre-defined ISIC train/test splits across 2019 and 2020 datasets.
    Subsamples training to 30% for iteration speed, grouping 2020 by patient_id
    so all lesions from a patient stay together.

    Returns:
        train_ds, test_ds      : batched, prefetched tf.data.Dataset
        steps_per_epoch        : int — pass to model.fit()
        val_steps              : int — pass to model.fit()
        class_weight           : dict — pass to model.fit() to handle imbalance
        y_test                 : numpy array — ground-truth labels for evaluate()
    """

    # ── 1. Load training metadata (2019 train + 2020 train) ─
    df19 = pd.read_csv("challenge-2019-training_metadata_2026-04-22.csv", low_memory=False)
    df20 = pd.read_csv("challenge-2020-training_metadata_2026-04-09.csv", low_memory=False)

    df19 = df19[df19["diagnosis_1"] != "Indeterminate"].copy()
    df20 = df20[df20["diagnosis_1"] != "Indeterminate"].copy()

    df19 = df19[["isic_id", "diagnosis_1"]].copy()
    df20 = df20[["isic_id", "diagnosis_1", "patient_id"]].copy()

    df19["img_dir"] = IMG_DIR_2019_TRAIN
    df20["img_dir"] = IMG_DIR_2020_TRAIN

    df19["target"] = (df19["diagnosis_1"].str.lower() == "malignant").astype(int)
    df20["target"] = (df20["diagnosis_1"].str.lower() == "malignant").astype(int)

    # ── 2. Load test metadata (2019 test + 2020 test) ───────
    df19t = pd.read_csv("challenge-2019-test_metadata_2026-04-10.csv", low_memory=False)
    df20t = pd.read_csv("challenge-2020-test_metadata_2026-04-09.csv", low_memory=False)

    df19t = df19t[df19t["diagnosis_1"] != "Indeterminate"].copy()
    df20t = df20t[df20t["diagnosis_1"] != "Indeterminate"].copy()

    df19t = df19t[["isic_id", "diagnosis_1"]].copy()
    df20t = df20t[["isic_id", "diagnosis_1"]].copy()

    df19t["img_dir"] = IMG_DIR_2019_TEST
    df20t["img_dir"] = IMG_DIR_2020_TEST

    df_test = pd.concat([df19t, df20t], ignore_index=True)
    df_test["target"] = (df_test["diagnosis_1"].str.lower() == "malignant").astype(int)

    # ── 3. Drop rows where image file doesn't exist ─────────
    def file_exists(row):
        return os.path.exists(os.path.join(row["img_dir"], row["isic_id"] + ".jpg"))

    before19 = len(df19)
    df19 = df19[df19.apply(file_exists, axis=1)].reset_index(drop=True)
    before20 = len(df20)
    df20 = df20[df20.apply(file_exists, axis=1)].reset_index(drop=True)
    before_test = len(df_test)
    df_test = df_test[df_test.apply(file_exists, axis=1)].reset_index(drop=True)

    print(f"2019 train: {len(df19)} images ({before19 - len(df19)} missing)")
    print(f"2020 train: {len(df20)} images ({before20 - len(df20)} missing)")
    print(f"Test:       {len(df_test)} images ({before_test - len(df_test)} missing)")

    # ── 4. Subsample training to 30% ────────────────────────
    # 2019: no patient IDs — stratified random sample
    _, df19_sub = train_test_split(
        df19, test_size=0.3, stratify=df19["target"], random_state=RANDOM_STATE
    )

    # 2020: group by patient_id so all lesions from a patient stay together
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    _, sub_idx = next(gss.split(df20, df20["target"], groups=df20["patient_id"]))
    df20_sub = df20.iloc[sub_idx]

    df_train = pd.concat([df19_sub, df20_sub], ignore_index=True)

    print(f"\nTraining subsample: {len(df_train)} images")
    print(f"Train class distribution:\n{df_train['target'].value_counts()}\n")
    print(f"Test class distribution:\n{df_test['target'].value_counts()}\n")

    # ── 5. Build path/label arrays ───────────────────────────
    paths_train = (df_train["img_dir"] + "\\" + df_train["isic_id"] + ".jpg").values
    y_train     = df_train["target"].values
    paths_test  = (df_test["img_dir"]  + "\\" + df_test["isic_id"]  + ".jpg").values
    y_test      = df_test["target"].values

    # ── 6. Class weight for imbalance ───────────────────────
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: n_neg / n_pos}
    print(f"Class weight — benign: 1.0, malignant: {class_weight[1]:.2f}")

    # ── 7. tf.data pipeline ──────────────────────────────────
    def parse_image(path, label):
        raw   = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    def build_dataset(paths, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_STATE)
        ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = build_dataset(paths_train, y_train, training=True)
    test_ds  = build_dataset(paths_test,  y_test,  training=False)

    steps_per_epoch = len(paths_train) // BATCH_SIZE
    val_steps       = len(paths_test)  // BATCH_SIZE

    return train_ds, test_ds, steps_per_epoch, val_steps, class_weight, y_test