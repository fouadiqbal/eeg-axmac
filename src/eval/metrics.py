"""Classification measures for held-out EEG trials."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           scores: np.ndarray | None = None, n_classes: int = 4) -> dict:
    """Return ACC, macro precision/recall/specificity/F1, and optional AUC.

    Sensitivity/TPR, specificity/TNR, and PPV are calculated one-versus-rest
    from the held-out confusion matrix. AUC requires probability scores.
    """
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.shape != pred.shape or true.ndim != 1 or len(true) == 0:
        raise ValueError("Labels and predictions must be aligned nonempty vectors")
    cm = confusion_matrix(true, pred, labels=np.arange(n_classes))
    per_class = []
    total = cm.sum()
    for class_id in range(n_classes):
        tp = int(cm[class_id, class_id])
        fn = int(cm[class_id].sum() - tp)
        fp = int(cm[:, class_id].sum() - tp)
        tn = int(total - tp - fn - fp)
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        ppv = tp / (tp + fp) if tp + fp else 0.0
        per_class.append({"class": class_id, "support": tp + fn,
                          "TPR": tpr, "TNR": tnr, "PPV": ppv})
    result = {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "macro_TPR": float(np.mean([item["TPR"] for item in per_class])),
        "macro_TNR": float(np.mean([item["TNR"] for item in per_class])),
        "macro_PPV": float(np.mean([item["PPV"] for item in per_class])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
    if scores is not None:
        probability = np.asarray(scores)
        if probability.shape != (len(true), n_classes):
            raise ValueError("AUC scores must have shape (trials, classes)")
        try:
            result["macro_auc_ovr"] = float(roc_auc_score(true, probability, multi_class="ovr",
                                                           average="macro", labels=np.arange(n_classes)))
        except ValueError:
            result["macro_auc_ovr"] = None
    return result
