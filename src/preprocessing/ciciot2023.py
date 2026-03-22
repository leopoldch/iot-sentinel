from pathlib import Path

import pandas as pd

from .common import (
    clean_infinities,
    drop_duplicates,
    export_splits,
    scale_features,
    split_data,
    validate_output,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MERGED_DIR = DATA_DIR / "ciciot2023" / "CIC_IOT_Dataset2023" / "MERGED_CSV"
OUTPUT_DIR = DATA_DIR / "processed" / "ciciot2023"

TARGET_LABEL = "Label"
TARGET_BINARY = "label_binary"
TARGET_MULTI = "label_multi"
TARGET_COLS = [TARGET_BINARY, TARGET_MULTI]

DEFAULT_SAMPLE_N = 2_000_000

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


def _get_csv_files() -> list[Path]:
    return sorted(MERGED_DIR.glob("Merged*.csv"))


def _count_labels(csv_files: list[Path]) -> dict[str, int]:
    """count labels across all files"""
    counts: dict[str, int] = {}
    for f in csv_files:
        chunk = pd.read_csv(f, usecols=[TARGET_LABEL])
        for label, n in chunk[TARGET_LABEL].value_counts().items():
            counts[label] = counts.get(label, 0) + n
    return counts


def _sample_label_aware(
    csv_files: list[Path], sample_n: int, label_counts: dict[str, int]
) -> pd.DataFrame:
    """Sample rows proportional to full label distribution"""
    total = sum(label_counts.values())

    quotas = {}
    for label, count in label_counts.items():
        quota = max(1, int(sample_n * count / total))
        quotas[label] = quota

    collected: dict[str, list[pd.DataFrame]] = {label: [] for label in quotas}
    filled: dict[str, int] = {label: 0 for label in quotas}

    for f in csv_files:
        chunk = pd.read_csv(f)
        for label, group in chunk.groupby(TARGET_LABEL):
            if label not in quotas:
                continue
            remaining = quotas[label] - filled[label]
            if remaining <= 0:
                continue
            if len(group) <= remaining:
                collected[label].append(group)
                filled[label] += len(group)
            else:
                collected[label].append(
                    group.sample(n=remaining, random_state=42)
                )
                filled[label] += remaining

        if all(filled[l] >= quotas[l] for l in quotas):
            break

    parts = []
    for label in collected:
        if collected[label]:
            parts.append(pd.concat(collected[label], ignore_index=True))

    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)
    return df


def _load_full(csv_files: list[Path]) -> pd.DataFrame:
    """Load all merged CSVs into memory. Requires ~32GB RAM."""
    chunks = []
    for f in csv_files:
        chunks.append(pd.read_csv(f))

    return pd.concat(chunks, ignore_index=True)


def load(sample_n: int | None = DEFAULT_SAMPLE_N, full: bool = False) -> pd.DataFrame:
    csv_files = _get_csv_files()

    if full:
        df = _load_full(csv_files)
    else:
        label_counts = _count_labels(csv_files)
        df = _sample_label_aware(csv_files, sample_n, label_counts)

    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary and multiclass target columns from Label."""
    df[TARGET_BINARY] = (df[TARGET_LABEL] != "BENIGN").astype(int)

    df[TARGET_MULTI] = df[TARGET_LABEL]

    df = df.drop(columns=[TARGET_LABEL])
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset."""
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = clean_infinities(df)
    df = drop_duplicates(df)
    return df


def run(sample_n: int | None = DEFAULT_SAMPLE_N, full: bool = False):
    """Full preprocessing pipeline for CICIoT2023"""

    if full:
        df = load(full=True)
    else:
        df = load(sample_n=sample_n)

    df = build_targets(df)
    df = clean(df)

    train, val, test = split_data(df, label_col=TARGET_BINARY)

    feature_cols = [c for c in FEATURE_COLS if c in train.columns]
    train, val, test, scaler_params = scale_features(train, val, test, feature_cols)

    metadata = {
        "dataset": "CICIoT2023",
        "source_dir": str(MERGED_DIR),
        "sampling_mode": "full" if full else "label_aware_approximate",
        "sample_target": "all" if full else sample_n,
        "n_rows_after_cleaning": len(train) + len(val) + len(test),
        "default_target": "binary",
        "target_binary": TARGET_BINARY,
        "target_multiclass": TARGET_MULTI,
        "feature_columns": feature_cols,
        "categorical_columns": [],
        "dropped_columns": [],
        "scaler": scaler_params,
        "binary_mapping": {"0": "BENIGN", "1": "Attack"},
        "split_ratio": "70/15/15",
    }

    export_splits(train, val, test, OUTPUT_DIR, metadata)

    validate_output(OUTPUT_DIR, feature_cols, TARGET_COLS)

    return OUTPUT_DIR
