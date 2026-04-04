from pathlib import Path

import pandas as pd

from .common import (
    build_metadata,
    clean_infinities,
    drop_conflicting_rows,
    encode_labels,
    encode_onehot,
    export_splits,
    proportional_rates,
    remove_overlaps,
    scale_features,
    split_data,
    validate_output,
)
from .taxonomy import map_edge_family

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PATH = (
    DATA_DIR
    / "edge-iiotset"
    / "Edge-IIoTset dataset"
    / "Selected dataset for ML and DL"
    / "DNN-EdgeIIoT-dataset.csv"
)
OUTPUT_DIR = DATA_DIR / "processed" / "edge_iiotset"

SAMPLE_N = 500_000
CHUNK_SIZE = 100_000
RANDOM_STATE = 42

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

TARGET_BINARY = "Attack_label"
TARGET_MULTI = "Attack_type"
TARGET_FAMILY = "label_family"
TARGET_COLS = [TARGET_BINARY, TARGET_MULTI, TARGET_FAMILY]

CATEGORICAL_COLS = [
    "http.request.method",
    "http.referer",
    "http.request.version",
    "mqtt.protoname",
    "mqtt.topic",
]


def load(sample_n=SAMPLE_N, full=False):
    header = pd.read_csv(RAW_PATH, nrows=0).columns.tolist()
    usecols = [c for c in header if c not in COLS_TO_DROP]

    if full:
        return pd.read_csv(RAW_PATH, usecols=usecols, low_memory=False)

    counts = {}
    for chunk in pd.read_csv(RAW_PATH, usecols=[TARGET_MULTI], chunksize=CHUNK_SIZE):
        for label, n in chunk[TARGET_MULTI].value_counts().items():
            counts[label] = counts.get(label, 0) + n

    rates = proportional_rates(counts, sample_n)

    parts = []
    for chunk in pd.read_csv(
        RAW_PATH, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False
    ):
        for label, group in chunk.groupby(TARGET_MULTI):
            if label not in rates:
                continue
            if rates[label] >= 1.0:
                parts.append(group)
            else:
                n = max(1, int(len(group) * rates[label]))
                parts.append(group.sample(n=n, random_state=RANDOM_STATE))

    df = pd.concat(parts, ignore_index=True)
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def run(sample_n=SAMPLE_N, full=False, encoding="onehot"):
    df = load(sample_n=sample_n, full=full)
    df[TARGET_FAMILY] = df[TARGET_MULTI].map(map_edge_family)

    existing_drops = [c for c in COLS_TO_DROP if c in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)

    feature_cols = [
        c for c in df.columns if c not in TARGET_COLS and c not in CATEGORICAL_COLS
    ]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col] == "0.0", col] = "0"

    df = clean_infinities(df.drop_duplicates())

    feature_cols_pre = [c for c in df.columns if c not in TARGET_COLS]
    df = drop_conflicting_rows(df, feature_cols_pre, TARGET_COLS)

    train, val, test = split_data(df, label_col=TARGET_BINARY)
    train, val, test = remove_overlaps(train, val, test, feature_cols_pre)

    cat_cols = [c for c in CATEGORICAL_COLS if c in train.columns]
    if encoding == "onehot":
        train, val, test, encoding_details = encode_onehot(train, val, test, cat_cols)
        metadata_key = "onehot_columns"
    else:
        train, val, test, encoding_details = encode_labels(train, val, test, cat_cols)
        metadata_key = "categorical_encoders"

    feature_cols = [c for c in train.columns if c not in TARGET_COLS]
    train, val, test, scaler_params = scale_features(train, val, test, feature_cols)
    train, val, test = remove_overlaps(train, val, test, feature_cols)

    metadata = build_metadata(
        dataset="Edge-IIoTset",
        feature_cols=feature_cols,
        target_binary=TARGET_BINARY,
        target_multiclass=TARGET_MULTI,
        target_family=TARGET_FAMILY,
        train=train,
        val=val,
        test=test,
        scaler_params=scaler_params,
        binary_mapping={"0": "Normal", "1": "Attack"},
        categorical_columns=cat_cols,
        dropped_columns=COLS_TO_DROP,
        encoding_method=encoding,
        extra={
            "source_file": str(RAW_PATH),
            "sampling_mode": "full" if full else "label_aware_chunked",
            "sample_target": "all" if full else sample_n,
            "n_rows_original": 2219201,
            "n_rows_after_cleaning": len(train) + len(val) + len(test),
        },
    )
    metadata[metadata_key] = encoding_details

    export_splits(train, val, test, OUTPUT_DIR, metadata)
    validate_output(OUTPUT_DIR, feature_cols, TARGET_COLS, metadata)
    return OUTPUT_DIR
