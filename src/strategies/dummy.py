from sklearn.dummy import DummyClassifier

from ._eval import (
    RANDOM_STATE,
    compute_metrics,
    compute_multiclass_metrics,
    cross_validate,
    print_metrics,
    print_multiclass_metrics,
)


def run(train, val, test, meta):
    feature_cols = meta["feature_columns"]
    target_col = meta["target_binary"]

    X_train = train[feature_cols]
    y_train = train[target_col]

    clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    y_pred_cv, y_prob_cv = cross_validate(clf, X_train, y_train)
    cv_metrics = compute_metrics(y_train.values, y_pred_cv, y_prob_cv)
    print_metrics("CV (5-fold)", cv_metrics)

    clf.fit(X_train, y_train)

    X_val = val[feature_cols]
    val_metrics = compute_metrics(
        val[target_col].values,
        clf.predict(X_val),
        clf.predict_proba(X_val)[:, 1],
    )
    print_metrics("VAL", val_metrics)

    X_test = test[feature_cols]
    test_metrics = compute_metrics(
        test[target_col].values,
        clf.predict(X_test),
        clf.predict_proba(X_test)[:, 1],
    )
    print_metrics("TEST", test_metrics)

    # Multiclass
    target_multi = meta["target_multiclass"]
    y_train_multi = train[target_multi].astype(str)
    class_names = sorted(y_train_multi.unique())

    clf_multi = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    clf_multi.fit(X_train, y_train_multi)

    multi_metrics = compute_multiclass_metrics(
        test[target_multi].astype(str).values,
        clf_multi.predict(X_test),
        class_names,
    )
    print_multiclass_metrics("TEST MULTICLASS", multi_metrics)

    return {
        "strategy": "dummy",
        "dataset": meta["dataset"],
        "cv": cv_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "test_multiclass": multi_metrics,
    }
