"""
EDITABLE — modify this file each iteration.

build_model() must return an instance of a class that implements:
    fit(train_ds, epochs, steps_per_epoch, class_weight)
    predict(x_ds)         -> numpy array of class labels
    predict_proba(x_ds)   -> numpy array of shape (n, 2), column 1 = P(malignant)

train_ds and x_ds are tf.data.Dataset objects.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier


class SkinLesionModel:
    def __init__(self):
        # SGDClassifier with log_loss = logistic regression, supports batch updates
        self.model = SGDClassifier(
            loss="log_loss",
            random_state=67,
            max_iter=1,
            warm_start=True,
        )
        self._classes = np.array([0, 1])

    def fit(self, train_ds, epochs=10, steps_per_epoch=None, class_weight=None):
        for epoch in range(epochs):
            for images, labels in train_ds:
                X = images.numpy().reshape(len(images), -1)
                y = labels.numpy()
                sample_weight = np.array([class_weight[yi] for yi in y]) if class_weight else None
                self.model.partial_fit(X, y, classes=self._classes, sample_weight=sample_weight)
        return self

    def predict(self, x_ds):
        preds = []
        for batch in x_ds:
            X = batch.numpy().reshape(len(batch), -1)
            preds.append(self.model.predict(X))
        return np.concatenate(preds)

    def predict_proba(self, x_ds):
        probas = []
        for batch in x_ds:
            X = batch.numpy().reshape(len(batch), -1)
            probas.append(self.model.predict_proba(X))
        return np.concatenate(probas)


def build_model():
    return SkinLesionModel()
