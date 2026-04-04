import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

RANDOM_STATE = 42
CV_FOLDS = 5


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "confusion_matrix": cm.tolist(),
    }


def cross_validate(clf, X, y):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


def best_threshold(y_true, scores, thresholds):
    best_t = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(threshold)

    return best_t, best_f1


def print_metrics(name, metrics):
    print()
    print(f"{name} results:")
    for label, key in [
        ("Balanced Accuracy", "balanced_accuracy"),
        ("F1", "f1"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Accuracy", "accuracy"),
    ]:
        print(f"{label}: {metrics[key]}")

    cm = np.array(metrics["confusion_matrix"])
    print(f"Confusion matrix: TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")


def compute_multiclass_metrics(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(score, 4),
            "support": int(n),
        }
        for name, p, r, score, n in zip(
            class_names, precision, recall, f1, support, strict=True
        )
    }

    return {
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(
            f1_score(
                y_true, y_pred, labels=class_names, average="macro", zero_division=0
            ),
            4,
        ),
        "weighted_f1": round(
            f1_score(
                y_true, y_pred, labels=class_names, average="weighted", zero_division=0
            ),
            4,
        ),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=class_names
        ).tolist(),
        "class_names": class_names,
    }


def print_multiclass_metrics(name, metrics):
    print()
    print(f"{name} results:")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']}")
    print(f"Macro F1: {metrics['macro_f1']}")
    print(f"Weighted F1: {metrics['weighted_f1']}")
    for class_name, values in metrics["per_class"].items():
        print(
            f"{class_name}: "
            f"P={values['precision']} R={values['recall']} "
            f"F1={values['f1']} n={values['support']}"
        )
