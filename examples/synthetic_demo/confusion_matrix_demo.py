"""
SYNTHETIC DEMONSTRATION ONLY.
This script creates an illustrative confusion matrix from a fixed hypothetical
class distribution. It does NOT train a classifier, load observations, or
reproduce the manuscript's experimental classification results.
Do not cite its output as measured model performance.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

LABELS = ["Bearing", "Gear", "Electrical", "Thermal"]
COUNTS = [400, 300, 200, 100]
rng = np.random.default_rng(42)
y_true = np.concatenate([[label] * n for label, n in zip(LABELS, COUNTS)])
y_pred = y_true.copy()
# Deliberately illustrative error injection; parameters are not fitted to observations.
error_rates = {"Bearing": 0.12, "Gear": 0.06, "Electrical": 0.04, "Thermal": 0.02}
for label, rate in error_rates.items():
    idx = np.flatnonzero(y_true == label)
    chosen = rng.choice(idx, size=int(round(len(idx) * rate)), replace=False)
    for i in chosen:
        alternatives = [x for x in LABELS if x != label]
        y_pred[i] = rng.choice(alternatives)

cm = confusion_matrix(y_true, y_pred, labels=LABELS)
f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
print("SYNTHETIC DEMO — NOT EXPERIMENTAL RESULTS")
print(cm)
print(dict(zip(LABELS, f1.round(3))))
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Illustrative synthetic confusion matrix (not measured results)")
fig.tight_layout()
fig.savefig("synthetic_confusion_matrix_NOT_EXPERIMENTAL.png", dpi=200)
