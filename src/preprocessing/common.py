import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def clean_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN, then drop rows with NaN."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df


def encode_categoricals(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Label-encode categorical columns. Fit on train only."""
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        
        df_train[col] = df_train[col].astype(str)
        df_val[col] = df_val[col].astype(str)
        df_test[col] = df_test[col].astype(str)

        le.fit(df_train[col])
        encoders[col] = list(le.classes_)

        df_train[col] = le.transform(df_train[col]).astype(int)

        for split_df in [df_val, df_test]:
            encoded = pd.Series(-1, index=split_df.index, dtype=int)
            mask = split_df[col].isin(le.classes_)
            if mask.any():
                encoded[mask] = le.transform(split_df.loc[mask, col]).astype(int)
            split_df[col] = encoded

    return df_train, df_val, df_test, encoders


def scale_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """StandardScaler fit on train only."""
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "columns": feature_cols,
    }
    return df_train, df_val, df_test, scaler_params


def split_data(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label_col],
        random_state=random_state,
    )
    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def export_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Export train/val/test as parquet + metadata JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def validate_output(output_dir: Path, feature_cols: list[str], target_cols: list[str]):
    """Lightweight validation of exported data."""
    errors = []

    for name in ["train", "val", "test"]:
        path = output_dir / f"{name}.parquet"
        if not path.exists():
            errors.append(f"Missing {path}")
            continue

        df = pd.read_parquet(path)

        # Check no target in features
        for t in target_cols:
            if t in feature_cols:
                errors.append(f"{name}: target '{t}' found in feature columns")

        # Check no NaN/inf in features
        feat_data = df[feature_cols]
        n_nan = feat_data.isna().sum().sum()
        n_inf = np.isinf(feat_data.select_dtypes(include=[np.number])).sum().sum()
        if n_nan > 0:
            errors.append(f"{name}: {n_nan} NaN values in features")
        if n_inf > 0:
            errors.append(f"{name}: {n_inf} inf values in features")

        # Check targets exist
        for t in target_cols:
            if t not in df.columns:
                errors.append(f"{name}: missing target column '{t}'")

    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        raise ValueError(f"Validation failed with {len(errors)} errors")
    else:
        print("  Validation passed")
