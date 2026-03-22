import argparse
import json
from pathlib import Path

from src.datasets import load_dataset
from src.strategies import isolation_forest, xgboost_clf

STRATEGIES = {
    "isolation_forest": isolation_forest,
    "xgboost": xgboost_clf,
}

RESULTS_DIR = Path("results")

def main():
    parser = argparse.ArgumentParser(description="Run a ML strategy on a dataset")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGIES.keys()),
        help="Strategy to run",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Preprocessed dataset name (e.g. edge_iiotset, ciciot2023)",
    )
    args = parser.parse_args()

    print(f"Strategy: {args.strategy}")
    print(f"Dataset:  {args.dataset}")
    print()

    train, val, test, meta = load_dataset(args.dataset)
    strategy = STRATEGIES[args.strategy]
    results = strategy.run(train, val, test, meta)

    out_dir = RESULTS_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.strategy}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
