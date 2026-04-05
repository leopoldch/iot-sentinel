from . import autoencoder, dummy, extra_trees, federated_sgd, gaussian_nb, hist_gb, isolation_forest, lof, logistic_regression, mlp_clf, random_forest, sgd_clf, xgboost_clf

STRATEGIES = {
    "autoencoder": autoencoder,
    "dummy": dummy,
    "extra_trees": extra_trees,
    "federated_sgd": federated_sgd,
    "gaussian_nb": gaussian_nb,
    "hist_gb": hist_gb,
    "isolation_forest": isolation_forest,
    "lof": lof,
    "logistic_regression": logistic_regression,
    "mlp": mlp_clf,
    "random_forest": random_forest,
    "sgd": sgd_clf,
    "xgboost": xgboost_clf,
}

STRATEGY_NAMES = list(STRATEGIES)
