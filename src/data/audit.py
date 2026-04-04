import json
import sys

from src.datasets import AVAILABLE, load_dataset
from src.preprocessing.common import count_overlaps


def main():
    failed = []

    for name in AVAILABLE:
        train, val, test, meta = load_dataset(name, validate=True)
        overlaps = count_overlaps(train, val, test, meta["feature_columns"])

        print(json.dumps({
            "dataset": meta["dataset"],
            "preprocess_version": meta["preprocess_version"],
            "feature_count": meta["feature_count"],
            "split_rows": meta["split_rows"],
            "family_classes": meta["family_classes"],
            "overlaps": overlaps,
        }, indent=2))

        nonzero = {k: v for k, v in overlaps.items() if v > 0}
        if nonzero:
            failed.append((name, nonzero))

    if failed:
        print("\nAUDIT FAILED - non-zero overlaps:")
        for name, ov in failed:
            print(f"  {name}: {ov}")
        sys.exit(1)

    print("\nAUDIT PASSED - all datasets have 0/0/0 overlaps.")


if __name__ == "__main__":
    main()
