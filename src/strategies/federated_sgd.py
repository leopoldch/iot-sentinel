import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

from ._eval import (
    CV_FOLDS,
    RANDOM_STATE,
    best_threshold,
    compute_metrics,
    compute_multiclass_metrics,
    print_metrics,
    print_multiclass_metrics,
)

DEFAULT_PARAMS = {
    "loss": "log_loss",
    "learning_rate": "constant",
    "eta0": 0.01,
    "max_iter": 1,
    "tol": None,
}
NUM_CLIENTS = 5
NUM_ROUNDS = 20
THRESHOLD_GRID = np.arange(0.1, 0.91, 0.05)


def _split_clients_iid(X, y, n_clients, rng):
    """Stratified IID split: each client gets a balanced shard."""
    skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=int(rng.randint(2**31)))
    clients = []
    for _, shard_idx in skf.split(X, y):
        clients.append((X.iloc[shard_idx], y.iloc[shard_idx]))
    return clients


def _build_model():
    return SGDClassifier(random_state=RANDOM_STATE, **DEFAULT_PARAMS)


def _fedavg(global_model, client_models, client_sizes):
    """Average client weights proportional to their data size."""
    total = sum(client_sizes)
    avg_coef = sum(m.coef_ * (n / total) for m, n in zip(client_models, client_sizes))
    avg_intercept = sum(m.intercept_ * (n / total) for m, n in zip(client_models, client_sizes))
    global_model.coef_ = avg_coef
    global_model.intercept_ = avg_intercept
    return global_model


def _train_federated(X_train, y_train, classes, rng):
    clients = _split_clients_iid(X_train, y_train, NUM_CLIENTS, rng)
    print(f"  {NUM_CLIENTS} clients (stratified IID), sizes: {[len(c[1]) for c in clients]}", flush=True)

    global_model = _build_model()
    global_model.partial_fit(clients[0][0][:1], clients[0][1][:1], classes=classes)

    for r in range(NUM_ROUNDS):
        client_models = []
        client_sizes = []

        for X_c, y_c in clients:
            local = _build_model()
            local.partial_fit(X_c[:1], y_c[:1], classes=classes)
            local.coef_ = global_model.coef_.copy()
            local.intercept_ = global_model.intercept_.copy()
            local.partial_fit(X_c, y_c)
            client_models.append(local)
            client_sizes.append(len(y_c))

        global_model = _fedavg(global_model, client_models, client_sizes)

        if (r + 1) % 5 == 0:
            print(f"  Round {r + 1}/{NUM_ROUNDS} done", flush=True)

    return global_model


def run(train, val, test, meta):
    feature_cols = meta["feature_columns"]
    target_col = meta["target_binary"]
    rng = np.random.RandomState(RANDOM_STATE)

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col].values

    classes = np.sort(y_train.unique())

    # --- CV (5-fold, each fold trained federatedly) ---
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = np.empty_like(y_train.values)
    y_prob_cv = np.empty(len(y_train), dtype=float)

    for i, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train)):
        print(f"CV fold {i + 1}/{CV_FOLDS}", flush=True)
        fold_model = _train_federated(
            X_train.iloc[tr_idx], y_train.iloc[tr_idx], classes, rng,
        )
        probs = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1]
        y_prob_cv[va_idx] = probs
        y_pred_cv[va_idx] = (probs >= 0.5).astype(int)

    cv_metrics = compute_metrics(y_train.values, y_pred_cv, y_prob_cv)
    print_metrics("CV (5-fold federated)", cv_metrics)

    # --- Final federated training ---
    print()
    fed_params = {
        "loss": "log_loss",
        "learning_rate": "constant",
        "eta0": 0.01,
        "num_clients": NUM_CLIENTS,
        "num_rounds": NUM_ROUNDS,
        "client_split": "stratified_iid",
    }
    print(f"Fixed params: {fed_params}")
    print("Training final global model ...", flush=True)

    global_model = _train_federated(X_train, y_train, classes, rng)

    val_probs = global_model.predict_proba(X_val)[:, 1]
    thresh, thresh_f1 = best_threshold(y_val, val_probs, THRESHOLD_GRID)
    thresh = round(thresh, 2)
    print(f"Best threshold (val F1={thresh_f1:.4f}): {thresh}")

    val_metrics = compute_metrics(y_val, (val_probs >= thresh).astype(int), val_probs)
    print_metrics("VAL", val_metrics)

    X_test = test[feature_cols]
    y_true = test[target_col].values
    y_prob = global_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= thresh).astype(int)

    test_metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics("TEST", test_metrics)

    # --- Multiclass ---
    target_multi = meta["target_multiclass"]
    class_names = sorted(
        set(train[target_multi].astype(str).unique())
        | set(test[target_multi].astype(str).unique())
    )
    multi_classes = np.array(class_names)

    print("\nTraining multiclass global model ...", flush=True)
    multi_clients = _split_clients_iid(X_train, train[target_multi].astype(str), NUM_CLIENTS, rng)

    global_multi = _build_model()
    global_multi.partial_fit(multi_clients[0][0][:1], multi_clients[0][1][:1], classes=multi_classes)

    for r in range(NUM_ROUNDS):
        client_models = []
        client_sizes = []
        for X_c, y_c in multi_clients:
            local = _build_model()
            local.partial_fit(X_c[:1], y_c[:1], classes=multi_classes)
            local.coef_ = global_multi.coef_.copy()
            local.intercept_ = global_multi.intercept_.copy()
            local.partial_fit(X_c, y_c)
            client_models.append(local)
            client_sizes.append(len(y_c))
        global_multi = _fedavg(global_multi, client_models, client_sizes)
        if (r + 1) % 5 == 0:
            print(f"  Round {r + 1}/{NUM_ROUNDS} done", flush=True)

    multi_metrics = compute_multiclass_metrics(
        test[target_multi].astype(str).values,
        global_multi.predict(test[feature_cols]),
        class_names,
    )
    print_multiclass_metrics("TEST multiclass", multi_metrics)

    return {
        "strategy": "federated_sgd",
        "dataset": meta["dataset"],
        "params": fed_params,
        "best_threshold": thresh,
        "cv": cv_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "test_multiclass": multi_metrics,
    }
