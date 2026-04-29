# AutoResearch Agent Instructions

## Objective
You are a medical imaging researcher. Using the combined ISIC 2019 and 2020 datasets, build the best possible binary classifier for skin lesion malignancy. First explore logistic regression and transition to more powerful models such as convolutional neural networks, then you are free to explore other options.  I already have a logistic regression baseline model that has ROC_AUC of 0.79 so start with more advanced models.  Consider Alex-Net before using Efficient-Net -B0.  Your stopping condition is: AUC above 0.85 with the testing data, a recall score above 0.95, and a plain-English summary a non-technical reader could understand. When building the model, keep in mind that I want to be able to use this machine learning model on a smartphone camera, potentially using TensorFlow Lite.  Although do not let this goal distract you from the stopping conditions.


### Rules
1. You may **ONLY** modify `model.py`
2. `prepare.py` and `run.py` are **FROZEN** — do not touch them
3. `build_model()` must return an object implementing all three of these methods:
   - `fit(train_ds, epochs, steps_per_epoch, class_weight)` — `train_ds` is a `tf.data.Dataset` yielding `(image_batch, label_batch)`
   - `predict(x_ds)` — `x_ds` is a `tf.data.Dataset` of image batches only; return a numpy array of class labels (0 or 1)
   - `predict_proba(x_ds)` — same input; return a numpy array of shape `(n, 2)` where column 1 is P(malignant)
4. Be aware of the class imbalance that heavily skews towards class 0 (benign)
5. Also be aware that there are sometimes multiple lesions that belong to the same patient ID
6. No additional data sources or external downloads

## Workflow

```
1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run:  python run.py "description of change"
5. Check roc_auc in output
6. If improved:  git add model.py && git commit -m "feat: <description>"
7. If worse:     git checkout model.py   (revert)
8. Repeat