import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold

from ._eval import (
    CV_FOLDS,
    RANDOM_STATE,
    best_threshold,
    compute_metrics,
    print_metrics,
)

DEFAULT_PARAMS = {"n_neighbors": 20, "contamination": 0.1}
THRESHOLD_PERCENTILES = np.arange(5, 100, 5)
MAX_FIT_SAMPLES = 30_000


def _subsample_benign(X, y, max_samples, rng):
    benign = X[y == 0]
    if len(benign) <= max_samples:
        return benign
    idx = rng.choice(len(benign), max_samples, replace=False)
    return benign.iloc[idx]


def _build_lof(n_jobs):
    return LocalOutlierFactor(
        n_neighbors=DEFAULT_PARAMS["n_neighbors"],
        contamination=DEFAULT_PARAMS["contamination"],
        novelty=True,
        n_jobs=n_jobs,
    )


def run(train, val, test, meta):
    feature_cols = meta["feature_columns"]
    target_col = meta["target_binary"]
    target_multi = meta["target_multiclass"]
    n_jobs = int(meta.get("n_jobs", 2))
    rng = np.random.RandomState(RANDOM_STATE)

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col].values

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = np.empty_like(y_train.values)
    y_scores_cv = np.empty(len(y_train), dtype=float)

    for i, (fold_train, fold_val) in enumerate(cv.split(X_train, y_train)):
        print(f"  CV fold {i + 1}/{CV_FOLDS} ...", flush=True)
        X_ft, y_ft = X_train.iloc[fold_train], y_train.iloc[fold_train]
        X_fv = X_train.iloc[fold_val]

        benign_sample = _subsample_benign(X_ft, y_ft.values, MAX_FIT_SAMPLES, rng)
        print(f"    fit on {len(benign_sample)} benign samples", flush=True)

        clf = _build_lof(n_jobs)
        clf.fit(benign_sample)

        y_pred_cv[fold_val] = (clf.predict(X_fv) == -1).astype(int)
        y_scores_cv[fold_val] = -clf.decision_function(X_fv)

    cv_metrics = compute_metrics(y_train.values, y_pred_cv, y_scores_cv)
    print_metrics("CV (5-fold)", cv_metrics)

    print()
    print(f"Fixed params: {DEFAULT_PARAMS}")
    print(f"Max fit samples: {MAX_FIT_SAMPLES}")

    benign_sample = _subsample_benign(X_train, y_train.values, MAX_FIT_SAMPLES, rng)
    print(f"Final fit on {len(benign_sample)} benign samples", flush=True)

    clf = _build_lof(n_jobs)
    clf.fit(benign_sample)

    val_scores = -clf.decision_function(X_val)
    thresholds = np.percentile(val_scores, THRESHOLD_PERCENTILES)
    thresh, thresh_f1 = best_threshold(y_val, val_scores, thresholds)
    print(f"Best threshold: {thresh:.6f} (val F1={thresh_f1:.4f})")

    val_metrics = compute_metrics(y_val, (val_scores >= thresh).astype(int), val_scores)
    print_metrics("VAL (calibrated)", val_metrics)

    X_test = test[feature_cols]
    y_true = test[target_col].values
    test_scores = -clf.decision_function(X_test)
    y_pred_test = (test_scores >= thresh).astype(int)

    test_metrics = compute_metrics(y_true, y_pred_test, test_scores)
    print_metrics("TEST", test_metrics)

    attack_types = test[target_multi].astype(str).to_numpy()
    per_attack = {}
    print()
    print("Per-attack detection rate:")
    for atype in sorted(set(attack_types)):
        mask = attack_types == atype
        support = int(mask.sum())
        detected = int(y_pred_test[mask].sum())
        recall = round(detected / support, 4) if support > 0 else 0.0
        per_attack[atype] = {"support": support, "detected": detected, "recall": recall}
        print(f"{atype}: support={support}, detected={detected}, recall={recall:.4f}")

    return {
        "strategy": "lof",
        "strategy_type": "anomaly_novelty",
        "dataset": meta["dataset"],
        "params": {**DEFAULT_PARAMS, "max_fit_samples": MAX_FIT_SAMPLES},
        "best_threshold": thresh,
        "cv": cv_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "per_attack_detection": per_attack,
    }
