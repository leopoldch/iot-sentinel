import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ._eval import (
    RANDOM_STATE,
    best_threshold,
    compute_metrics,
    compute_multiclass_metrics,
    cross_validate,
    print_metrics,
    print_multiclass_metrics,
)

DEFAULT_PARAMS = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
THRESHOLD_GRID = np.arange(0.1, 0.91, 0.05)


def build_model(n_jobs, eval_metric, **params):
    return XGBClassifier(
        random_state=RANDOM_STATE, n_jobs=n_jobs, eval_metric=eval_metric, **params
    )


def run(train, val, test, meta):
    feature_cols = meta["feature_columns"]
    target_col = meta["target_binary"]
    n_jobs = int(meta.get("n_jobs", 2))

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col].values

    base_clf = build_model(n_jobs, "logloss", **DEFAULT_PARAMS)
    y_pred_cv, y_prob_cv = cross_validate(base_clf, X_train, y_train)
    cv_metrics = compute_metrics(y_train.values, y_pred_cv, y_prob_cv)
    print_metrics("CV (5-fold)", cv_metrics)

    print()
    print(f"Fixed params: {DEFAULT_PARAMS}")

    clf_final = build_model(n_jobs, "logloss", **DEFAULT_PARAMS)
    clf_final.fit(X_train, y_train)

    y_prob_val = clf_final.predict_proba(X_val)[:, 1]
    thresh, thresh_f1 = best_threshold(y_val, y_prob_val, THRESHOLD_GRID)
    print(f"Best threshold (val F1={thresh_f1:.4f}): {thresh}")

    val_metrics = compute_metrics(y_val, (y_prob_val >= thresh).astype(int), y_prob_val)
    print_metrics("VAL", val_metrics)

    X_test = test[feature_cols]
    y_true = test[target_col].values
    y_prob_test = clf_final.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= thresh).astype(int)

    test_metrics = compute_metrics(y_true, y_pred_test, y_prob_test)
    print_metrics("TEST", test_metrics)

    target_multi = meta["target_multiclass"]
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(train[target_multi].astype(str))
    class_names = sorted(le.classes_)

    clf_multi = build_model(n_jobs, "mlogloss", **DEFAULT_PARAMS)
    clf_multi.fit(X_train, y_train_encoded)
    y_pred_multi = le.inverse_transform(clf_multi.predict(X_test).astype(int))

    multi_metrics = compute_multiclass_metrics(
        test[target_multi].astype(str).values, y_pred_multi, class_names
    )
    print_multiclass_metrics("TEST (multiclass)", multi_metrics)

    return {
        "strategy": "xgboost",
        "dataset": meta["dataset"],
        "params": DEFAULT_PARAMS,
        "best_threshold": thresh,
        "cv": cv_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "test_multiclass": multi_metrics,
    }
