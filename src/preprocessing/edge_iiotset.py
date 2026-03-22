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
RAW_PATH = (
    DATA_DIR
    / "edge-iiotset"
    / "Edge-IIoTset dataset"
    / "Selected dataset for ML and DL"
    / "DNN-EdgeIIoT-dataset.csv"
)
OUTPUT_DIR = DATA_DIR / "processed" / "edge_iiotset"

DEFAULT_SAMPLE_N = 500_000
CHUNK_SIZE = 100_000

COLS_TO_DROP = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4",
    "http.file_data",
    "http.request.uri.query",
    "http.request.full_uri",
    "icmp.transmit_timestamp",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "mqtt.msg",
    "mqtt.msg_decoded_as",
    "dns.qry.name",
]

def _usecols() -> list[str] | None:
    header = pd.read_csv(RAW_PATH, nrows=0).columns.tolist()
    return [c for c in header if c not in COLS_TO_DROP]


TARGET_BINARY = "Attack_label"
TARGET_MULTI = "Attack_type"
TARGET_COLS = [TARGET_BINARY, TARGET_MULTI]

CATEGORICAL_COLS = [
    "http.request.method",
    "http.referer",
    "http.request.version",
    "mqtt.protoname",
    "mqtt.topic",
]


def _count_labels() -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in pd.read_csv(RAW_PATH, usecols=[TARGET_MULTI], chunksize=CHUNK_SIZE):
        for label, n in chunk[TARGET_MULTI].value_counts().items():
            counts[label] = counts.get(label, 0) + n
    return counts


def _load_sampled(sample_n: int) -> pd.DataFrame:
    label_counts = _count_labels()
    total = sum(label_counts.values())

    quotas = {
        label: max(1, int(sample_n * count / total))
        for label, count in label_counts.items()
    }
    filled: dict[str, int] = {label: 0 for label in quotas}
    collected: list[pd.DataFrame] = []

    usecols = _usecols()

    for chunk in pd.read_csv(
        RAW_PATH, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False
    ):
        for label, group in chunk.groupby(TARGET_MULTI):
            if label not in quotas:
                continue
            remaining = quotas[label] - filled[label]
            if remaining <= 0:
                continue
            if len(group) <= remaining:
                collected.append(group)
                filled[label] += len(group)
            else:
                collected.append(group.sample(n=remaining, random_state=42))
                filled[label] += remaining

        if all(filled[l] >= quotas[l] for l in quotas):
            break

    df = pd.concat(collected, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def _load_full() -> pd.DataFrame:
    usecols = _usecols()
    df = pd.read_csv(RAW_PATH, usecols=usecols, low_memory=False)
    return df


def load(sample_n: int | None = DEFAULT_SAMPLE_N, full: bool = False) -> pd.DataFrame:
    if full:
        return _load_full()
    else:
        return _load_sampled(sample_n)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    existing_drops = [c for c in COLS_TO_DROP if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    feature_cols = [c for c in df.columns if c not in TARGET_COLS + CATEGORICAL_COLS]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col] == "0.0", col] = "0"

    df = clean_infinities(df)
    df = drop_duplicates(df)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TARGET_COLS]


def run(sample_n: int | None = DEFAULT_SAMPLE_N, full: bool = False):
    """Full preprocessing pipeline for Edge-IIoTset"""
    
    if full:
        df = load(full=True)
    else:
        df = load(sample_n=sample_n)

    df = clean(df)

    train, val, test = split_data(df, label_col=TARGET_BINARY)

    cat_cols_present = [c for c in CATEGORICAL_COLS if c in train.columns]
    train, val, test, encoders = encode_categoricals(train, val, test, cat_cols_present)

    feature_cols = get_feature_cols(train)
    train, val, test, scaler_params = scale_features(train, val, test, feature_cols)

    metadata = {
        "dataset": "Edge-IIoTset",
        "source_file": str(RAW_PATH),
        "sampling_mode": "full" if full else "label_aware_chunked",
        "sample_target": "all" if full else sample_n,
        "n_rows_original": 2219201,
        "n_rows_after_cleaning": len(train) + len(val) + len(test),
        "default_target": "binary",
        "target_binary": TARGET_BINARY,
        "target_multiclass": TARGET_MULTI,
        "feature_columns": feature_cols,
        "categorical_columns": cat_cols_present,
        "dropped_columns": COLS_TO_DROP,
        "categorical_encoders": encoders,
        "scaler": scaler_params,
        "binary_mapping": {"0": "Normal", "1": "Attack"},
        "split_ratio": "70/15/15",
    }

    export_splits(train, val, test, OUTPUT_DIR, metadata)
    validate_output(OUTPUT_DIR, feature_cols, TARGET_COLS)

    return OUTPUT_DIR
