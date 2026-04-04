import argparse
import json
import os
from pathlib import Path

from src.datasets import AVAILABLE, load_dataset
from src.strategies import STRATEGIES, STRATEGY_NAMES

RESULTS_DIR = Path("results")
ALL_DATASETS = list(AVAILABLE)
DEFAULT_JOBS = 2
MIN_COMMON_FEATURES = 5
THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def set_thread_limits(jobs):
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(jobs)


def save_results(dataset_name, filename, results):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    return out_path


def run_strategy(
    strategy_name, dataset_name, force_preprocess=False, jobs=DEFAULT_JOBS
):
    print(f"Strategy: {strategy_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Jobs: {jobs}\n")

    train, val, test, meta = load_dataset(
        dataset_name, force_preprocess=force_preprocess
    )
    meta["n_jobs"] = jobs

    results = STRATEGIES[strategy_name].run(train, val, test, meta)
    out_path = save_results(dataset_name, f"{strategy_name}.json", results)
    print(f"\nResults saved to {out_path}")


def run_cross_dataset(
    strategy_name, source_name, force_preprocess=False, jobs=DEFAULT_JOBS
):
    print(f"\nCross-dataset evaluation: train on {source_name}")
    print("Current implementation requires a shared post-preprocessing feature space.")

    train_src, val_src, _, meta_src = load_dataset(
        source_name,
        force_preprocess=force_preprocess,
    )
    meta_src["n_jobs"] = jobs
    source_features = set(meta_src["feature_columns"])
    source_binary = meta_src["target_binary"]
    source_multi = meta_src["target_multiclass"]
    strategy = STRATEGIES[strategy_name]
    skipped = 0

    for target_name in ALL_DATASETS:
        if target_name == source_name:
            continue

        print(f"\nTesting on {target_name}")
        _, _, test_tgt, meta_tgt = load_dataset(
            target_name,
            force_preprocess=force_preprocess,
        )
        common_features = sorted(source_features & set(meta_tgt["feature_columns"]))

        if len(common_features) < MIN_COMMON_FEATURES:
            print(f"Skipping: only {len(common_features)} common features")
            skipped += 1
            continue

        print(
            f"Using {len(common_features)} common features "
            f"(out of {len(meta_src['feature_columns'])} source, "
            f"{len(meta_tgt['feature_columns'])} target)"
        )

        rename = {}
        if meta_tgt["target_binary"] != source_binary:
            rename[meta_tgt["target_binary"]] = source_binary
        if meta_tgt["target_multiclass"] != source_multi:
            rename[meta_tgt["target_multiclass"]] = source_multi
        if rename:
            test_tgt = test_tgt.rename(columns=rename)

        cross_meta = meta_src.copy()
        cross_meta["feature_columns"] = common_features
        cross_meta["dataset"] = f"{meta_src['dataset']} -> {meta_tgt['dataset']}"

        results = strategy.run(train_src, val_src, test_tgt, cross_meta)
        results["cross_dataset"] = {
            "train_on": source_name,
            "test_on": target_name,
            "common_features": len(common_features),
        }

        out_path = save_results(
            source_name,
            f"cross_{target_name}_{strategy_name}.json",
            results,
        )
        print(f"Results saved to {out_path}")

    if skipped == len(ALL_DATASETS) - 1:
        print(
            "\nNo cross-dataset runs were executed. "
            "The current dataset-specific feature spaces have no usable intersection."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="IoT Sentinel")
    parser.add_argument("--strategy", required=True, choices=STRATEGY_NAMES)
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cross-dataset", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("--dataset is required unless --all is set")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    return args


def main():
    args = parse_args()
    set_thread_limits(args.jobs)

    dataset_names = ALL_DATASETS if args.all else [args.dataset]
    for dataset_name in dataset_names:
        run_strategy(
            args.strategy,
            dataset_name,
            force_preprocess=args.force_preprocess,
            jobs=args.jobs,
        )
        if args.cross_dataset:
            run_cross_dataset(
                args.strategy,
                dataset_name,
                force_preprocess=args.force_preprocess,
                jobs=args.jobs,
            )


if __name__ == "__main__":
    main()
