import json
from pathlib import Path

import pandas as pd

from src.preprocessing import PREPROCESSORS
from src.preprocessing.common import (
    PREPROCESS_VERSION,
    count_overlaps,
    validate_metadata,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

AVAILABLE = {
    "edge_iiotset": DATA_DIR / "edge_iiotset",
    "ciciot2023": DATA_DIR / "ciciot2023",
    "ton_iot": DATA_DIR / "ton_iot",
}

REQUIRED_FILES = ["train.parquet", "val.parquet", "test.parquet", "metadata.json"]


def read_metadata(base):
    with (base / "metadata.json").open() as f:
        return json.load(f)


def is_ready(base):
    if any(not (base / name).exists() for name in REQUIRED_FILES):
        return False
    try:
        meta = read_metadata(base)
        validate_metadata(meta)
    except Exception:
        return False
    return meta["preprocess_version"] == PREPROCESS_VERSION


def validate_loaded_dataset(train, val, test, meta):
    feature_cols = meta["feature_columns"]
    target_cols = [
        meta["target_binary"],
        meta["target_multiclass"],
        meta["target_family"],
    ]

    missing = [col for col in feature_cols + target_cols if col not in train.columns]
    if missing:
        raise ValueError(f"Train missing columns: {missing}")

    train_cols = train.columns.tolist()
    for split_name, df in [("val", val), ("test", test)]:
        if df.columns.tolist() != train_cols:
            raise ValueError(f"{split_name} columns don't match train")

    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        if len(df) != meta["split_rows"][split_name]:
            raise ValueError(f"{split_name} row count doesn't match metadata")
        if df[feature_cols].isna().any().any():
            raise ValueError(f"{split_name} has NaN in features")
        for target_col in target_cols:
            if df[target_col].isna().any():
                raise ValueError(f"{split_name} has NaN in target {target_col}")

    overlaps = count_overlaps(train, val, test, feature_cols)
    nonzero = {name: count for name, count in overlaps.items() if count > 0}
    if nonzero:
        raise ValueError(f"Feature overlaps between splits: {nonzero}")


def load_dataset(name, force_preprocess=False, validate=True):
    if name not in AVAILABLE:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(AVAILABLE)}")

    base = AVAILABLE[name]
    if force_preprocess or not is_ready(base):
        reason = "forced" if force_preprocess else "missing/stale"
        print(f"Preprocessed data {reason} for '{name}', running preprocessing...")
        PREPROCESSORS[name].run()

    meta = read_metadata(base)
    validate_metadata(meta)

    train = pd.read_parquet(base / "train.parquet")
    val = pd.read_parquet(base / "val.parquet")
    test = pd.read_parquet(base / "test.parquet")

    if validate:
        validate_loaded_dataset(train, val, test, meta)

    return train, val, test, meta
