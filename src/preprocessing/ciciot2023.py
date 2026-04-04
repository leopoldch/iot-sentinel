from pathlib import Path

import pandas as pd

from .common import (
    build_metadata,
    clean_infinities,
    drop_conflicting_rows,
    export_splits,
    proportional_rates,
    remove_overlaps,
    scale_features,
    split_data,
    validate_output,
)
from .taxonomy import map_ciciot_family

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MERGED_DIR = DATA_DIR / "ciciot2023" / "CIC_IOT_Dataset2023" / "MERGED_CSV"
OUTPUT_DIR = DATA_DIR / "processed" / "ciciot2023"

TARGET_LABEL = "Label"
TARGET_BINARY = "label_binary"
TARGET_MULTI = "label_multi"
TARGET_FAMILY = "label_family"
TARGET_COLS = [TARGET_BINARY, TARGET_MULTI, TARGET_FAMILY]

SAMPLE_N = 2_000_000
RANDOM_STATE = 42

FEATURE_COLS = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]


def count_labels(csv_files):
    counts = {}
    for f in csv_files:
        chunk = pd.read_csv(f, usecols=[TARGET_LABEL])
        for label, n in chunk[TARGET_LABEL].value_counts().items():
            counts[label] = counts.get(label, 0) + n
    return counts


def sample_by_label(csv_files, sample_n, label_counts):
    rates = proportional_rates(label_counts, sample_n)

    parts = []
    for f in csv_files:
        chunk = pd.read_csv(f)
        for label, group in chunk.groupby(TARGET_LABEL):
            if label not in rates:
                continue
            if rates[label] >= 1.0:
                parts.append(group)
            else:
                n = max(1, int(len(group) * rates[label]))
                parts.append(group.sample(n=n, random_state=RANDOM_STATE))

    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def load(sample_n=SAMPLE_N, full=False):
    csv_files = sorted(MERGED_DIR.glob("Merged*.csv"))
    if full:
        return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    return sample_by_label(csv_files, sample_n, count_labels(csv_files))


def run(sample_n=SAMPLE_N, full=False):
    df = load(sample_n=sample_n, full=full)
    df[TARGET_BINARY] = (df[TARGET_LABEL] != "BENIGN").astype(int)
    df[TARGET_MULTI] = df[TARGET_LABEL]
    df[TARGET_FAMILY] = df[TARGET_LABEL].map(map_ciciot_family)
    df = df.drop(columns=[TARGET_LABEL])

    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = clean_infinities(df.drop_duplicates())

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[feature_cols + TARGET_COLS]
    df = drop_conflicting_rows(df, feature_cols, TARGET_COLS)

    train, val, test = split_data(df, label_col=TARGET_BINARY)
    train, val, test = remove_overlaps(train, val, test, feature_cols)
    train, val, test, scaler_params = scale_features(train, val, test, feature_cols)
    train, val, test = remove_overlaps(train, val, test, feature_cols)

    metadata = build_metadata(
        dataset="CICIoT2023",
        feature_cols=feature_cols,
        target_binary=TARGET_BINARY,
        target_multiclass=TARGET_MULTI,
        target_family=TARGET_FAMILY,
        train=train,
        val=val,
        test=test,
        scaler_params=scaler_params,
        binary_mapping={"0": "BENIGN", "1": "Attack"},
        extra={
            "source_dir": str(MERGED_DIR),
            "sampling_mode": "full" if full else "label_aware_approximate",
            "sample_target": "all" if full else sample_n,
            "n_rows_after_cleaning": len(train) + len(val) + len(test),
        },
    )

    export_splits(train, val, test, OUTPUT_DIR, metadata)
    validate_output(OUTPUT_DIR, feature_cols, TARGET_COLS, metadata)
    return OUTPUT_DIR
