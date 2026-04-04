from . import dummy, isolation_forest, random_forest, xgboost_clf

STRATEGIES = {
    "dummy": dummy,
    "isolation_forest": isolation_forest,
    "random_forest": random_forest,
    "xgboost": xgboost_clf,
}

STRATEGY_NAMES = list(STRATEGIES)
