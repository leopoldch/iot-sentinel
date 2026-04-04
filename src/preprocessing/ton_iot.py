from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    build_metadata,
    drop_conflicting_rows,
    encode_labels,
    encode_onehot,
    export_splits,
    remove_overlaps,
    scale_features,
    split_data,
    validate_output,
)
from .taxonomy import map_ton_family

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "ton_iot" / "processed_iot"
OUTPUT_DIR = DATA_DIR / "processed" / "ton_iot"

TARGET_LABEL = "label"
TARGET_TYPE = "type"
TARGET_BINARY = "label_binary"
TARGET_MULTI = "label_multi"
TARGET_FAMILY = "label_family"
TARGET_COLS = [TARGET_BINARY, TARGET_MULTI, TARGET_FAMILY]

SAMPLE_N = 1_000_000
RANDOM_STATE = 42


def get_csv_files():
    return sorted(RAW_DIR.glob("IoT_*.csv"))


def load(sample_n=SAMPLE_N, full=False):
    parts = []
    for path in get_csv_files():
        df = pd.read_csv(path, low_memory=False)
        df.columns = [col.strip().replace("\ufeff", "") for col in df.columns]
        df["device"] = path.stem.removeprefix("IoT_").lower()
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)

    if full or sample_n is None or len(df) <= sample_n:
        return df

    labels = df[TARGET_TYPE].astype(str).str.strip()
    label_counts = labels.value_counts()
    total = int(label_counts.sum())

    sampled = []
    for label, count in label_counts.items():
        quota = max(1, int(sample_n * count / total))
        group = df[labels == label]
        if len(group) <= quota:
            sampled.append(group)
        else:
            sampled.append(group.sample(n=quota, random_state=RANDOM_STATE))

    result = pd.concat(sampled, ignore_index=True)
    return result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def run(sample_n=SAMPLE_N, full=False, encoding="onehot"):
    df = load(sample_n=sample_n, full=full)
    n_rows_before = len(df)
    df[TARGET_BINARY] = pd.to_numeric(df[TARGET_LABEL], errors="coerce").astype(int)
    df[TARGET_MULTI] = df[TARGET_TYPE].astype(str).str.strip().str.lower()
    df[TARGET_FAMILY] = df[TARGET_MULTI].map(map_ton_family)

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(
            df[col]
        ):
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "nan", "None", "<NA>"]), col] = pd.NA

    date_text = df["date"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    time_text = df["time"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    dt = pd.to_datetime(
        date_text + " " + time_text, format="%d-%b-%y %H:%M:%S", errors="coerce"
    )
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["second"] = dt.dt.second
    df = df.drop(columns=["date", "time", TARGET_LABEL, TARGET_TYPE])

    categorical_cols = [
        col
        for col in df.columns
        if col not in TARGET_COLS and not pd.api.types.is_numeric_dtype(df[col])
    ]
    numeric_cols = [
        col
        for col in df.columns
        if col not in TARGET_COLS and col not in categorical_cols
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_cols:
        df[col] = df[col].fillna("__missing__")

    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()

    feature_cols_pre = [col for col in df.columns if col not in TARGET_COLS]
    df = drop_conflicting_rows(df, feature_cols_pre, TARGET_COLS)

    train, val, test = split_data(df, label_col=TARGET_BINARY)

    medians = train[numeric_cols].median()
    for split_df in (train, val, test):
        split_df[numeric_cols] = split_df[numeric_cols].fillna(medians)
    train = train.dropna(subset=numeric_cols).reset_index(drop=True)
    val = val.dropna(subset=numeric_cols).reset_index(drop=True)
    test = test.dropna(subset=numeric_cols).reset_index(drop=True)

    train, val, test = remove_overlaps(train, val, test, feature_cols_pre)

    if encoding == "onehot":
        train, val, test, encoding_details = encode_onehot(
            train, val, test, categorical_cols
        )
        metadata_key = "onehot_columns"
    else:
        train, val, test, encoding_details = encode_labels(
            train, val, test, categorical_cols
        )
        metadata_key = "categorical_encoders"

    feature_cols = [col for col in train.columns if col not in TARGET_COLS]
    train, val, test, scaler_params = scale_features(train, val, test, feature_cols)
    train, val, test = remove_overlaps(train, val, test, feature_cols)

    metadata = build_metadata(
        dataset="TON-IoT",
        feature_cols=feature_cols,
        target_binary=TARGET_BINARY,
        target_multiclass=TARGET_MULTI,
        target_family=TARGET_FAMILY,
        train=train,
        val=val,
        test=test,
        scaler_params=scaler_params,
        binary_mapping={"0": "Normal", "1": "Attack"},
        categorical_columns=categorical_cols,
        dropped_columns=["date", "time", TARGET_LABEL, TARGET_TYPE],
        encoding_method=encoding,
        extra={
            "source_dir": str(RAW_DIR),
            "source_files": [p.name for p in get_csv_files()],
            "sampling_mode": "full" if full else "label_aware_full_load",
            "sample_target": "all" if full else sample_n,
            "n_rows_before_cleaning": n_rows_before,
            "n_rows_after_cleaning": len(train) + len(val) + len(test),
        },
    )
    metadata[metadata_key] = encoding_details

    export_splits(train, val, test, OUTPUT_DIR, metadata)
    validate_output(OUTPUT_DIR, feature_cols, TARGET_COLS, metadata)
    return OUTPUT_DIR
