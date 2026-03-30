from pathlib import Path

import pandas as pd

from .common import (
    clean_infinities,
    drop_duplicates,
    encode_categoricals,
    export_splits,
    scale_features,
    split_data,
    validate_output,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "ton_iot" / "processed_iot"
OUTPUT_DIR = DATA_DIR / "processed" / "ton_iot"

TARGET_LABEL = "label"
TARGET_TYPE = "type"
TARGET_BINARY = "label_binary"
TARGET_MULTI = "label_multi"
TARGET_COLS = [TARGET_BINARY, TARGET_MULTI]

DEFAULT_SAMPLE_N = 1_000_000


def _get_csv_files() -> list[Path]:
    return sorted(RAW_DIR.glob("IoT_*.csv"))


def _device_name(path: Path) -> str:
    return path.stem.removeprefix("IoT_").lower()


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
            df.loc[df[col].isin(["", "nan", "None", "<NA>"]), col] = pd.NA
    return df


def _load_full() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for csv_path in _get_csv_files():
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = [col.strip().replace("\ufeff", "") for col in df.columns]
        df["device"] = _device_name(csv_path)
        parts.append(df)

    return pd.concat(parts, ignore_index=True)


def _sample_label_aware(df: pd.DataFrame, sample_n: int) -> pd.DataFrame:
    if sample_n is None or len(df) <= sample_n:
        return df

    label_counts = df[TARGET_TYPE].astype(str).str.strip().value_counts()
    total = int(label_counts.sum())

    parts: list[pd.DataFrame] = []
    for label, count in label_counts.items():
        quota = max(1, int(sample_n * count / total))
        group = df[df[TARGET_TYPE].astype(str).str.strip() == label]
        if len(group) <= quota:
            parts.append(group)
        else:
            parts.append(group.sample(n=quota, random_state=42))

    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    return sampled


def load(sample_n: int | None = DEFAULT_SAMPLE_N, full: bool = False) -> pd.DataFrame:
    df = _load_full()
    if full:
        return df
    return _sample_label_aware(df, sample_n)


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_BINARY] = pd.to_numeric(df[TARGET_LABEL], errors="coerce").astype(int)
    df[TARGET_MULTI] = df[TARGET_TYPE].astype(str).str.strip().str.lower()
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = _normalize_strings(df)

    dt = pd.to_datetime(
        df["date"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        + " "
        + df["time"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        format="%d-%b-%y %H:%M:%S",
        errors="coerce",
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

    numeric_cols = [col for col in df.columns if col not in TARGET_COLS + categorical_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna("__missing__")

    df = clean_infinities(df)
    df = drop_duplicates(df)
    return df, categorical_cols


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in TARGET_COLS]


def run(sample_n: int | None = DEFAULT_SAMPLE_N, full: bool = False):
    if full:
        df = load(full=True)
        sampling_mode = "full"
        sample_target = "all"
    else:
        df = load(sample_n=sample_n)
        sampling_mode = "label_aware_full_load"
        sample_target = sample_n

    n_rows_before_cleaning = len(df)

    df = build_targets(df)
    df, categorical_cols = clean(df)

    train, val, test = split_data(df, label_col=TARGET_BINARY)

    train, val, test, encoders = encode_categoricals(train, val, test, categorical_cols)

    feature_cols = get_feature_cols(train)
    train, val, test, scaler_params = scale_features(train, val, test, feature_cols)

    metadata = {
        "dataset": "TON-IoT",
        "source_dir": str(RAW_DIR),
        "source_files": [p.name for p in _get_csv_files()],
        "sampling_mode": sampling_mode,
        "sample_target": sample_target,
        "n_rows_before_cleaning": n_rows_before_cleaning,
        "n_rows_after_cleaning": len(train) + len(val) + len(test),
        "default_target": "binary",
        "target_binary": TARGET_BINARY,
        "target_multiclass": TARGET_MULTI,
        "feature_columns": feature_cols,
        "categorical_columns": categorical_cols,
        "dropped_columns": ["date", "time", TARGET_LABEL, TARGET_TYPE],
        "categorical_encoders": encoders,
        "scaler": scaler_params,
        "binary_mapping": {"0": "Normal", "1": "Attack"},
        "split_ratio": "70/15/15",
    }

    export_splits(train, val, test, OUTPUT_DIR, metadata)
    validate_output(OUTPUT_DIR, feature_cols, TARGET_COLS)

    return OUTPUT_DIR
