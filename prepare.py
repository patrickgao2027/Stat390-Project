"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metric, and plotting.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import os
import PIL
import PIL.Image
import tensorflow as tf
import csv


### VARS
RANDOM_STATE = 67
RESULTS_FILE = "results.tsv"
###


df2019train = pd.read_csv("ISIC_2019_Training_Metadata.csv")
df2019test = pd.read_csv("challenge-2019-test_metadata_2026-04-10.csv")

df2020train = pd.read_csv("challenge-2020-training_metadata_2026-04-09.csv")
df2020test = pd.read_csv("challenge-2020-test_metadata_2026-04-09.csv")

# ground truth for test and train
df2019trainResponse = df2019train['diagnosis_1']
df2019testResponse = df2019test['diagnosis_1']
df2020trainResponse = df2020train['diagnosis_1']
df2020testResponse = df2020test['diagnosis_1']

# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:,1]
    mainmetric = roc_auc_score(y_test, y_proba)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

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
             label=f"ROC curve (AUC = {metrics['ROC AUC']:.2f})")
    
    # Plot Baseline (Random Guess)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 3. Aesthetics
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Analysis')
    
    # Add metrics summary as a text box
    stats_text = (f"Accuracy: {metrics['Accuracy']:.2f}\n"
                  f"Precision: {metrics['Precision']:.2f}\n"
                  f"Recall: {metrics['Recall']:.2f}")
    
    plt.gca().text(0.6, 0.2, stats_text, style='italic',
                   bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

from sklearn.model_selection import train_test_split

# ── Paths ───────────────────────────────────────────────────
IMG_DIR_2019 = r"C:\Users\Owner\Documents\Stat390-Project\ISIC-images 2019 train"
IMG_DIR_2020 = r"C:\Users\Owner\Documents\Stat390-Project\ISIC-images train 2020"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
AUTOTUNE     = tf.data.AUTOTUNE

def load_data():
    """
    Merge ISIC 2019 + 2020 metadata, build stratified train/test splits,
    and return tf.data.Dataset pipelines — images are loaded from disk
    on the fly in batches, never all at once.

    Returns:
        train_ds, test_ds: Batched, prefetched tf.data.Dataset objects
        steps_per_epoch, val_steps: int — pass these to model.fit()
        class_weight: dict — pass to model.fit() to handle class imbalance
    """

    # ── 1. Build merged metadata DataFrame ─────────────────
    df19 = pd.read_csv("challenge-2019-training_metadata_2026-04-22.csv", low_memory=False)
    df20 = pd.read_csv("challenge-2020-training_metadata_2026-04-09.csv", low_memory=False)

    df19 = df19[df19["diagnosis_1"] != "Indeterminate"].copy()
    df20 = df20[df20["diagnosis_1"] != "Indeterminate"].copy()

    df19 = df19[["isic_id", "diagnosis_1"]].copy()
    df20 = df20[["isic_id", "diagnosis_1"]].copy()

    df19["img_dir"] = IMG_DIR_2019
    df20["img_dir"] = IMG_DIR_2020

    df = pd.concat([df19, df20], ignore_index=True)
    df["target"] = (df["diagnosis_1"].str.lower() == "malignant").astype(int)

    # ── 2. Drop rows where the image file doesn't exist ────
    # Catches any missing downloads before the pipeline starts
    def file_exists(row):
        return os.path.exists(os.path.join(row["img_dir"], row["isic_id"] + ".jpg"))

    before = len(df)
    df = df[df.apply(file_exists, axis=1)].reset_index(drop=True)
    print(f"Images found: {len(df)} ({before - len(df)} missing, skipped)")
    print(f"Class distribution:\n{df['target'].value_counts()}\n")

    # ── 3. Stratified train/test split on file paths ───────
    # We split the DataFrame (just strings + ints), not image arrays
    paths = (df["img_dir"] + "\\" + df["isic_id"] + ".jpg").values
    labels = df["target"].values

    paths_train, paths_test, y_train, y_test = train_test_split(
        paths, labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE
    )

    # ── 4. Class weight for imbalance ──────────────────────
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    class_weight = {
        0: 1.0,
        1: n_neg / n_pos   # upweights malignant during training
    }
    print(f"Class weight applied — benign: 1.0, malignant: {class_weight[1]:.2f}")

    # ── 5. tf.data pipeline ─────────────────────────────────
    def parse_image(path, label):
        """Decode one JPEG, resize, normalize to [0, 1]."""
        raw   = tf.io.read_file(path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def augment(image, label):
        """Training-only augmentations applied on the fly."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        # Random rotation ±15° via tfa or a manual crop — skip for now,
        # tf.image doesn't have native rotation; add if you install
        # tensorflow-addons: tfa.image.rotate(image, angles)
        return image, label

    def build_dataset(paths, labels, training=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_STATE)
        ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(AUTOTUNE)   # loads next batch while GPU trains current
        return ds

    train_ds = build_dataset(paths_train, y_train, training=True)
    test_ds  = build_dataset(paths_test,  y_test,  training=False)

    steps_per_epoch = len(paths_train) // BATCH_SIZE
    val_steps       = len(paths_test)  // BATCH_SIZE

    return train_ds, test_ds, steps_per_epoch, val_steps, class_weight
