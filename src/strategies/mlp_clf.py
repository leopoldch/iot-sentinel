import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from ._eval import (
    RANDOM_STATE,
    best_threshold,
    compute_metrics,
    compute_multiclass_metrics,
    cross_validate,
    print_metrics,
    print_multiclass_metrics,
)

DEFAULT_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "max_iter": 300,
    "early_stopping": True,
}
THRESHOLD_GRID = np.arange(0.1, 0.91, 0.05)


def build_model(**params):
    return MLPClassifier(random_state=RANDOM_STATE, **params)


def run(train, val, test, meta):
    feature_cols = meta["feature_columns"]
    target_col = meta["target_binary"]

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col].values

    clf = build_model(**DEFAULT_PARAMS)
    y_pred_cv, y_prob_cv = cross_validate(clf, X_train, y_train)
    cv_metrics = compute_metrics(y_train.values, y_pred_cv, y_prob_cv)
    print_metrics("CV (5-fold)", cv_metrics)

    print()
    print(f"Fixed params: {DEFAULT_PARAMS}")

    best_clf = build_model(**DEFAULT_PARAMS)
    best_clf.fit(X_train, y_train)

    val_probs = best_clf.predict_proba(X_val)[:, 1]
    thresh, thresh_f1 = best_threshold(y_val, val_probs, THRESHOLD_GRID)
    thresh = round(thresh, 2)
    print(f"Best threshold (val F1={thresh_f1:.4f}): {thresh}")

    val_metrics = compute_metrics(y_val, (val_probs >= thresh).astype(int), val_probs)
    print_metrics("VAL", val_metrics)

    X_test = test[feature_cols]
    y_true = test[target_col].values
    y_prob = best_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= thresh).astype(int)

    test_metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics("TEST", test_metrics)

    target_multi = meta["target_multiclass"]
    class_names = sorted(train[target_multi].astype(str).unique())

    le = LabelEncoder()
    le.fit(class_names)

    multi_clf = build_model(**DEFAULT_PARAMS)
    multi_clf.fit(X_train, le.transform(train[target_multi].astype(str)))

    multi_pred = le.inverse_transform(multi_clf.predict(test[feature_cols]))
    multi_metrics = compute_multiclass_metrics(
        test[target_multi].astype(str).values,
        multi_pred,
        class_names,
    )
    print_multiclass_metrics("TEST multiclass", multi_metrics)

    return {
        "strategy": "mlp",
        "dataset": meta["dataset"],
        "params": {k: str(v) if isinstance(v, tuple) else v for k, v in DEFAULT_PARAMS.items()},
        "best_threshold": thresh,
        "cv": cv_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "test_multiclass": multi_metrics,
    }
