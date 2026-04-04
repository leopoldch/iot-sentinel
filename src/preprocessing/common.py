import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .taxonomy import CANONICAL_FAMILY_CLASSES

PREPROCESS_VERSION = 5
ARTIFACT_SCHEMA = "iot_sentinel.preprocessed.v1"
SPLIT_TEST_SIZE = 0.15
SPLIT_VAL_SIZE = 0.15
RANDOM_STATE = 42


def proportional_rates(counts, sample_n):
    total = sum(counts.values())
    return {
        label: min(1.0, max(1, int(sample_n * count / total)) / count)
        for label, count in counts.items()
    }


def clean_infinities(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def hash_rows(df, cols):
    return pd.util.hash_pandas_object(df[cols], index=False)


def count_overlaps(train, val, test, feature_cols):
    h_train = set(hash_rows(train, feature_cols))
    h_val = set(hash_rows(val, feature_cols))
    h_test = set(hash_rows(test, feature_cols))
    return {
        "train_val": len(h_train & h_val),
        "train_test": len(h_train & h_test),
        "val_test": len(h_val & h_test),
    }


def drop_conflicting_rows(df, feature_cols, target_cols):
    hashes = hash_rows(df, feature_cols)
    signatures = df[target_cols].astype(str).fillna("__nan__").agg("||".join, axis=1)
    tmp = pd.DataFrame({"hash": hashes, "sig": signatures})
    bad_hashes = set(tmp.groupby("hash")["sig"].nunique().loc[lambda s: s > 1].index)

    if not bad_hashes:
        return df

    keep = ~hashes.isin(bad_hashes)
    print(f"Removed {int((~keep).sum())} conflicting rows")
    return df.loc[keep].reset_index(drop=True)


def remove_overlaps(train, val, test, feature_cols):
    h_train = set(hash_rows(train, feature_cols))
    val_keep = ~hash_rows(val, feature_cols).isin(h_train)
    val = val.loc[val_keep].reset_index(drop=True)

    h_val = set(hash_rows(val, feature_cols))
    h_test = hash_rows(test, feature_cols)
    test_keep = ~h_test.isin(h_train) & ~h_test.isin(h_val)
    test = test.loc[test_keep].reset_index(drop=True)

    removed = int((~val_keep).sum()) + int((~test_keep).sum())
    if removed > 0:
        print(f"Overlap cleanup: {removed} rows removed from val/test")

    return train, val, test


def encode_labels(train, val, test, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        for df in (train, val, test):
            df[col] = df[col].astype(str)

        le = LabelEncoder()
        le.fit(train[col])
        encoders[col] = list(le.classes_)
        known = set(le.classes_)

        train[col] = le.transform(train[col]).astype(int)

        for df in (val, test):
            encoded = pd.Series(-1, index=df.index, dtype=int)
            mask = df[col].isin(known)
            if mask.any():
                encoded[mask] = le.transform(df.loc[mask, col]).astype(int)
            df[col] = encoded

    return train, val, test, encoders


def sanitize_columns(df):
    rename = {
        col: re.sub(r"[\[\]<>]", "_", col)
        for col in df.columns
        if re.search(r"[\[\]<>]", col)
    }
    return df.rename(columns=rename) if rename else df


def align_to_train(train, *others):
    return [df.reindex(columns=train.columns, fill_value=0) for df in others]


def encode_onehot(train, val, test, categorical_cols):
    for col in categorical_cols:
        for df in (train, val, test):
            df[col] = df[col].astype(str)

    train = sanitize_columns(pd.get_dummies(train, columns=categorical_cols))
    dummy_cols = [
        c
        for c in train.columns
        if any(c.startswith(f"{cat}_") for cat in categorical_cols)
    ]

    val, test = align_to_train(
        train,
        sanitize_columns(pd.get_dummies(val, columns=categorical_cols)),
        sanitize_columns(pd.get_dummies(test, columns=categorical_cols)),
    )

    return train, val, test, dummy_cols


def scale_features(train, val, test, feature_cols):
    scaler = StandardScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    val[feature_cols] = scaler.transform(val[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    return (
        train,
        val,
        test,
        {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "columns": feature_cols,
        },
    )


def split_data(df, label_col):
    train_val, test = train_test_split(
        df,
        test_size=SPLIT_TEST_SIZE,
        stratify=df[label_col],
        random_state=RANDOM_STATE,
    )
    relative_val = SPLIT_VAL_SIZE / (1 - SPLIT_TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label_col],
        random_state=RANDOM_STATE,
    )
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def export_splits(train, val, test, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def collect_classes(train, val, test, column):
    return sorted(
        set(train[column].astype(str).unique())
        | set(val[column].astype(str).unique())
        | set(test[column].astype(str).unique())
    )


def build_metadata(
    dataset,
    feature_cols,
    target_binary,
    target_multiclass,
    target_family,
    train,
    val,
    test,
    scaler_params,
    binary_mapping,
    categorical_columns=None,
    dropped_columns=None,
    encoding_method=None,
    extra=None,
):
    """Build the standard metadata dict for a preprocessed dataset."""
    metadata = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "preprocess_version": PREPROCESS_VERSION,
        "dataset": dataset,
        "feature_view": "native",
        "cross_dataset_ready": False,
        "default_target": "binary",
        "target_binary": target_binary,
        "target_multiclass": target_multiclass,
        "target_family": target_family,
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "categorical_columns": categorical_columns or [],
        "dropped_columns": dropped_columns or [],
        "encoding_method": encoding_method,
        "scaler": scaler_params,
        "binary_mapping": binary_mapping,
        "family_classes": collect_classes(train, val, test, target_family),
        "multiclass_classes": collect_classes(train, val, test, target_multiclass),
        "split_ratio": "70/15/15",
        "split_rows": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "integrity": {
            "overlap_policy": "exact_feature_rows_removed_across_splits",
            "conflicting_feature_policy": "drop_identical_features_with_multiple_targets",
        },
    }
    if extra:
        metadata.update(extra)
    return metadata


def validate_metadata(metadata):
    required = {
        "artifact_schema",
        "preprocess_version",
        "dataset",
        "target_binary",
        "target_multiclass",
        "target_family",
        "feature_columns",
        "split_rows",
        "family_classes",
        "multiclass_classes",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Metadata missing keys: {missing}")
    if metadata["artifact_schema"] != ARTIFACT_SCHEMA:
        raise ValueError(f"Bad artifact schema: {metadata['artifact_schema']}")
    if metadata["preprocess_version"] != PREPROCESS_VERSION:
        raise ValueError(f"Bad preprocess version: {metadata['preprocess_version']}")
    unknown = sorted(
        set(metadata.get("family_classes", [])) - set(CANONICAL_FAMILY_CLASSES)
    )
    if unknown:
        raise ValueError(f"Unknown family classes: {unknown}")


def validate_output(output_dir, feature_cols, target_cols, metadata=None):
    errors = []
    if metadata is None:
        with (output_dir / "metadata.json").open() as f:
            metadata = json.load(f)
    try:
        validate_metadata(metadata)
    except ValueError as e:
        errors.append(str(e))

    splits = {}
    for name in ["train", "val", "test"]:
        path = output_dir / f"{name}.parquet"
        if not path.exists():
            errors.append(f"Missing {path}")
            continue

        df = pd.read_parquet(path)
        splits[name] = df

        for t in target_cols:
            if t in feature_cols:
                errors.append(f"{name}: target '{t}' in feature columns")

        feat = df[feature_cols]
        if feat.isna().sum().sum() > 0:
            errors.append(f"{name}: NaN in features")
        if np.isinf(feat.select_dtypes(include=[np.number])).sum().sum() > 0:
            errors.append(f"{name}: inf in features")

        for t in target_cols:
            if t not in df.columns:
                errors.append(f"{name}: missing target '{t}'")
            elif df[t].isna().any():
                errors.append(f"{name}: null in target '{t}'")

    if len(splits) == 3:
        train_cols = splits["train"].columns.tolist()
        for name in ["val", "test"]:
            if splits[name].columns.tolist() != train_cols:
                errors.append(f"{name}: columns differ from train")

        overlaps = count_overlaps(
            splits["train"], splits["val"], splits["test"], feature_cols
        )
        for key, count in overlaps.items():
            if count > 0:
                errors.append(f"non-zero feature overlap for {key}: {count}")

        for name, df in splits.items():
            expected = metadata["split_rows"][name]
            if len(df) != expected:
                errors.append(f"{name}: {len(df)} rows, expected {expected}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        raise ValueError(f"Validation failed with {len(errors)} errors")
    print("Validation passed")
