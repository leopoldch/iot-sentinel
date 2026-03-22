import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def run(train, val, test, meta) -> dict:
    feature_cols = meta["feature_columns"]
    target_col = meta["target_binary"]

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_train_normal = X_train[y_train == 0]

    clf = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_normal)

    results = {"strategy": "isolation_forest", "dataset": meta["dataset"]}

    for split_name, split_df in [("val", val), ("test", test)]:
        X = split_df[feature_cols]
        y_true = split_df[target_col].values

        raw_pred = clf.predict(X)
        y_pred = (raw_pred == -1).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "confusion_matrix": cm.tolist(),
        }
        results[split_name] = metrics

        print(f"\n  {split_name.upper()} results:")
        print(f"    Accuracy:  {metrics['accuracy']}")
        print(f"    Precision: {metrics['precision']}")
        print(f"    Recall:    {metrics['recall']}")
        print(f"    F1:        {metrics['f1']}")
        cm = np.array(metrics["confusion_matrix"])
        print(f"    Confusion matrix:")
        print(f"      TN={cm[0,0]:>8}  FP={cm[0,1]:>8}")
        print(f"      FN={cm[1,0]:>8}  TP={cm[1,1]:>8}")

    return results
