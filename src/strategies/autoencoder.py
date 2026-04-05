import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold

from ._eval import (
    CV_FOLDS,
    RANDOM_STATE,
    best_threshold,
    compute_metrics,
    print_metrics,
)

DEFAULT_PARAMS = {
    "hidden_layer_sizes": (64, 16, 64),
    "max_iter": 100,
    "early_stopping": True,
}
THRESHOLD_PERCENTILES = np.arange(5, 100, 5)


def _build_ae(**params):
    return MLPRegressor(random_state=RANDOM_STATE, **params)


def _reconstruction_error(model, X):
    X_arr = np.asarray(X, dtype=np.float64)
    X_pred = model.predict(X_arr)
    return np.mean((X_arr - X_pred) ** 2, axis=1)


def run(train, val, test, meta):
    feature_cols = meta["feature_columns"]
    target_col = meta["target_binary"]
    target_multi = meta["target_multiclass"]

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

        benign = X_ft[y_ft == 0]
        ae = _build_ae(**DEFAULT_PARAMS)
        ae.fit(benign, benign)

        scores = _reconstruction_error(ae, X_fv)
        y_scores_cv[fold_val] = scores
        thresh_cv = np.percentile(scores, 90)
        y_pred_cv[fold_val] = (scores >= thresh_cv).astype(int)

    cv_metrics = compute_metrics(y_train.values, y_pred_cv, y_scores_cv)
    print_metrics("CV (5-fold)", cv_metrics)

    print()
    print(f"Fixed params: {DEFAULT_PARAMS}")

    benign_train = X_train[y_train == 0]
    print(f"Final fit on {len(benign_train)} benign samples", flush=True)

    ae = _build_ae(**DEFAULT_PARAMS)
    ae.fit(benign_train, benign_train)

    val_scores = _reconstruction_error(ae, X_val)
    thresholds = np.percentile(val_scores, THRESHOLD_PERCENTILES)
    thresh, thresh_f1 = best_threshold(y_val, val_scores, thresholds)
    print(f"Best threshold: {thresh:.6f} (val F1={thresh_f1:.4f})")

    val_metrics = compute_metrics(y_val, (val_scores >= thresh).astype(int), val_scores)
    print_metrics("VAL (calibrated)", val_metrics)

    X_test = test[feature_cols]
    y_true = test[target_col].values
    test_scores = _reconstruction_error(ae, X_test)
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
        "strategy": "autoencoder",
        "strategy_type": "anomaly_reconstruction",
        "dataset": meta["dataset"],
        "params": {k: str(v) if isinstance(v, tuple) else v for k, v in DEFAULT_PARAMS.items()},
        "best_threshold": thresh,
        "cv": cv_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "per_attack_detection": per_attack,
    }
