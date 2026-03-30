import json
from pathlib import Path
import importlib
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

AVAILABLE = {
    "edge_iiotset": DATA_DIR / "edge_iiotset",
    "ciciot2023": DATA_DIR / "ciciot2023",
    "ton_iot": DATA_DIR / "ton_iot",
}

PREPROCESSORS = {
    "edge_iiotset": "src.preprocessing.edge_iiotset",
    "ciciot2023": "src.preprocessing.ciciot2023",
    "ton_iot": "src.preprocessing.ton_iot",
}


def _is_ready(base: Path) -> bool:
    return all((base / f).exists() for f in ["train.parquet", "val.parquet", "test.parquet", "metadata.json"])


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if name not in AVAILABLE:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(AVAILABLE.keys())}")

    base = AVAILABLE[name]

    if not _is_ready(base):
        print(f"Preprocessed data not found for '{name}', running preprocessing...")
        mod = importlib.import_module(PREPROCESSORS[name])
        mod.run()

    meta = json.load(open(base / "metadata.json"))
    train = pd.read_parquet(base / "train.parquet")
    val = pd.read_parquet(base / "val.parquet")
    test = pd.read_parquet(base / "test.parquet")
    return train, val, test, meta
