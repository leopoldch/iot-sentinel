# IoT Sentinel

An ML-based intrusion detection system (IDS) evaluation framework for IoT networks.
Benchmarks multiple detection strategies across CICIoT2023, Edge-IIoTset, and TON-IoT
datasets with support for both binary and multiclass classification.

## Features

- Multi-dataset support (CICIoT2023, Edge-IIoTset, TON-IoT)
- Binary and multiclass evaluation
- Fixed-parameter baselines with validation-based threshold calibration
- Threshold calibration
- Conservative runtime controls via `--jobs`
- Experimental cross-dataset evaluation hook
- Versioned preprocessing artifacts with strict validation and overlap checks
- One-hot and label encoding support for categorical features
- Harmonized `label_family` target across datasets
- Intentionally simple code paths for preprocessing, loading, and evaluation

## Datasets

| Dataset | Description |
|---------|-------------|
| **CICIoT2023** | Large-scale IoT attack dataset covering 33 attack types across 7 categories, generated from a realistic IoT topology. |
| **Edge-IIoTset** | Cybersecurity dataset targeting IoT and IIoT edge environments, with 14 attack types including DDoS, injection, and malware. |
| **TON-IoT** | Telemetry dataset from IoT/IIoT services including weather, fridge, and garage door sensors, with diverse attack scenarios. |

## Strategies

### Supervised classifiers

| Strategy | CLI name | Description |
|----------|----------|-------------|
| **Dummy** | `dummy` | Baseline classifier for sanity checking. |
| **Logistic Regression** | `logistic_regression` | Linear discriminative classifier. |
| **SGDClassifier (hinge)** | `sgd` | Large-scale linear SVM via stochastic gradient descent, calibrated for probabilities. |
| **GaussianNB** | `gaussian_nb` | Probabilistic generative classifier (naive Bayes). |
| **MLPClassifier** | `mlp` | Two-layer dense neural network (128, 64). |
| **Random Forest** | `random_forest` | Ensemble of bagged decision trees. |
| **Extra Trees** | `extra_trees` | Extremely randomized trees ensemble. |
| **HistGradientBoosting** | `hist_gb` | Native sklearn histogram-based gradient boosting. |
| **XGBoost** | `xgboost` | Gradient-boosted tree classifier. |

### Anomaly detection

| Strategy | CLI name | Description |
|----------|----------|-------------|
| **Isolation Forest** | `isolation_forest` | Unsupervised anomaly detection (tree-based isolation). |
| **Autoencoder** | `autoencoder` | Reconstruction-based anomaly detection (MLPRegressor). |
| **LOF** | `lof` | Local Outlier Factor novelty detection (subsampled fit). |

### Federated learning

| Strategy | CLI name | Description |
|----------|----------|-------------|
| **Federated SGD** | `federated_sgd` | FedAvg with 5 stratified IID clients, SGD log_loss, 20 rounds. |

## Installation

Requires **Python 3.13+** and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Usage

### Download datasets

```bash
uv run python -m src.data.download
```

Datasets are downloaded from Kaggle and stored in the `data/` directory.

### Run a strategy

```bash
uv run python main.py --dataset edge_iiotset --strategy xgboost
```

**Options:**

| Flag | Description | Values |
|------|-------------|--------|
| `--dataset` | Target dataset | `ciciot2023`, `edge_iiotset`, `ton_iot` |
| `--strategy` | Detection strategy | See [Strategies](#strategies) for full list |
| `--all` | Run on all datasets | Flag |
| `--cross-dataset` | Experimental cross-dataset evaluation | Flag |
| `--force-preprocess` | Regenerate preprocessed splits (use after code changes) | Flag |
| `--jobs` | Maximum parallel workers for safe local runs | Integer |

### Run on all datasets

```bash
uv run python main.py --strategy xgboost --all --jobs 2
```

### Cross-dataset evaluation

```bash
uv run python main.py --dataset edge_iiotset --strategy xgboost --cross-dataset
```

Results are written as JSON to `results/<dataset>/<strategy>.json`.

If you later want parameter sweeps, prefer running W&B or similar orchestration on a remote server.
The local CLI is intentionally conservative and meant for safe baseline runs on a workstation that may not tolerate full CPU saturation.

### Audit processed artifacts

```bash
uv run python -m src.data.audit
```

## Methodology

### 1. Goal of the pipeline

The project is designed as a **controlled evaluation framework** for IoT intrusion detection on three public datasets.
The objective is not to reproduce each original paper line by line.
The objective is to apply a **single, explicit, reproducible evaluation protocol** to several datasets so the model comparisons inside this repository are fair and interpretable.

In practice, the pipeline answers this question:

> Given one processed dataset, which baseline or tree-based method performs best under the same split policy, the same validation policy, and the same metric family?

### 2. Common design choices

These choices are shared across the repository:

- **Same-dataset evaluation first**: every main result in this repo is train/validation/test on the same dataset.
- **Stratified split**: each dataset is split into `70/15/15` train/validation/test.
- **Fixed model parameters**: the current repo uses fixed model defaults and chooses only decision thresholds on validation data.
- **Untouched test split**: the test split is used only for the final report.
- **Versioned artifacts**: processed data are exported as parquet files plus a metadata JSON file.
- **Strict overlap control**: exact duplicate feature rows across splits are removed and then checked again after scaling.
- **Strict load-time validation**: `load_dataset()` refuses to load artifacts with wrong schema, wrong row counts, NaNs in features, or non-zero split overlap.

This makes the repository behave like a small benchmark harness rather than a loose collection of notebooks or one-off scripts.

### 3. Data pipeline shared structure

Every dataset-specific preprocessor follows the same high-level flow:

1. Load raw CSV data.
2. Build target columns.
3. Clean invalid values and duplicates.
4. Remove rows where the same feature vector maps to conflicting labels.
5. Split into train/validation/test.
6. Remove exact feature overlaps across splits.
7. Encode categorical variables if needed.
8. Scale features with a scaler fit on train only.
9. Remove residual overlaps again after scaling.
10. Export `train.parquet`, `val.parquet`, `test.parquet`, and `metadata.json`.
11. Validate the exported artifact immediately.

The second overlap cleanup is intentional.
It exists because two rows that were distinct before scaling can become identical after standardization because of floating-point precision.

### 4. Dataset-specific preprocessing

#### CICIoT2023

Current implementation:

- Reads files matching `Merged*.csv` from the CICIoT2023 merged directory.
- Uses **label-aware proportional sampling** by default with a target of `2,000,000` rows.
- Builds:
  - `label_binary`: `0 = BENIGN`, `1 = attack`
  - `label_multi`: original attack label
  - `label_family`: harmonized attack family
- Coerces the selected CICIoT feature columns to numeric.
- Removes rows with `inf`, `-inf`, or resulting `NaN`.
- Removes exact duplicate rows.
- Keeps only the selected feature columns plus the three target columns.
- Drops rows where the same feature vector points to conflicting targets.
- Splits stratified on the **binary** target.
- Removes exact feature overlap across `train`, `val`, and `test`.
- Fits `StandardScaler` on train only and applies it to validation and test.
- Removes residual overlap again after scaling.

Important detail:
the current artifact is a **repo-defined benchmark view** of CICIoT2023, not a literal reimplementation of the original CIC release protocol.

#### Edge-IIoTset

Current implementation:

- Reads the main `DNN-EdgeIIoT-dataset.csv`.
- Drops a fixed set of high-cardinality or text-heavy columns such as raw hosts, payload-like fields, and URI content.
- Uses **chunked proportional sampling** by attack type by default with a target of `500,000` rows.
- Builds:
  - existing `Attack_label` as binary label
  - existing `Attack_type` as multiclass label
  - derived `label_family` as harmonized family label
- Converts non-categorical feature columns to numeric.
- Normalizes a small set of categorical columns used for encoding.
- Removes rows with invalid numeric values, then removes duplicates.
- Drops rows where identical features map to different targets.
- Splits stratified on the binary target.
- Removes exact feature overlap across splits before encoding.
- Encodes categorical variables.
  - default: one-hot encoding
  - supported but not default: integer label encoding
- Fits `StandardScaler` on train only and applies it to validation and test.
- Removes residual overlap again after scaling.

This is close in spirit to the paper:
binary and multiclass labels, selected features, categorical encoding, and standardization are preserved.
However, this repo does **not** claim to exactly replay the full paper protocol step for step.

#### TON-IoT

Current implementation:

- Loads every `IoT_*.csv` file from the processed IoT folder.
- Adds a `device` column derived from the filename.
- Uses **label-aware proportional sampling** by attack type by default with a target of `1,000,000` rows.
- Builds:
  - `label_binary` from the original `label`
  - `label_multi` from the original `type`
  - `label_family` from the harmonized family mapping
- Normalizes string columns.
- Parses `date + time` into numeric time features:
  - `month`, `day`, `hour`, `minute`, `second`
- Drops the raw `date`, `time`, `label`, and `type` columns.
- Infers categorical vs numeric columns from the dataframe schema.
- Converts numeric columns to numeric explicitly.
- Fills missing categorical values with `__missing__`.
- Replaces `inf` and `-inf` with `NaN`.
- Removes duplicates.
- Drops conflicting feature rows.
- Splits stratified on the binary target.
- Imputes missing numeric values using **train medians only**.
- Drops rows still invalid after train-median imputation.
- Removes exact feature overlap across splits before encoding.
- Encodes categorical variables.
  - default: one-hot encoding
  - supported but not default: integer label encoding
- Fits `StandardScaler` on train only and applies it to validation and test.
- Removes residual overlap again after scaling.

Scientifically, this is a conservative choice:
median imputation is fit on train only, which avoids leaking information from validation or test into preprocessing.

### 5. Artifact contract

Each processed dataset exports:

- `train.parquet`
- `val.parquet`
- `test.parquet`
- `metadata.json`

The metadata records at least:

- artifact schema
- preprocess version
- dataset name
- feature column list
- target column names
- feature count
- split row counts
- family classes
- multiclass classes
- scaler parameters
- encoding method
- binary label mapping

This makes the processed dataset self-describing.
The model code does not need to guess which columns are targets or which feature set was used.

### 6. Loading and auditing

`load_dataset()` is intentionally strict.
It validates:

- required files exist
- metadata schema is correct
- artifact version matches the current preprocessor version
- expected columns are present
- split columns are identical
- split row counts match metadata
- feature columns contain no `NaN`
- target columns contain no `NaN`
- exact feature overlap across splits is zero

`uv run python -m src.data.audit` is the standalone integrity command.
It reloads each processed dataset, recomputes the overlap counts, prints a compact summary, and exits with a non-zero status if any dataset is contaminated.

This gives the project a simple but meaningful reproducibility checkpoint:
the repository can distinguish "artifact exists" from "artifact is scientifically usable."

### 7. Evaluation protocol

The evaluation protocol depends on the strategy.

#### Dummy baseline

The dummy baseline:

- runs 5-fold stratified cross-validation on the train split for the binary target
- fits on the full train split
- evaluates binary metrics on validation and test
- trains a separate multiclass dummy model on train
- evaluates multiclass metrics on test

Its role is not performance.
Its role is to show what happens when a model mostly predicts the dominant class.

#### Isolation Forest

The anomaly baseline:

- uses the binary target for evaluation
- trains the Isolation Forest on **normal samples only**
- computes 5-fold cross-validation predictions on the train split
- uses fixed parameters: `n_estimators=100`, `contamination=0.1`
- calibrates the anomaly threshold from the full validation split
- evaluates on test
- reports per-attack detection rates using the multiclass attack label

This is scientifically useful because it tests whether a one-class style anomaly detector can detect attacks without the same type of direct supervision as the supervised baselines.

#### Supervised classifiers (Random Forest, XGBoost, HistGB, Extra Trees, MLP, Logistic Regression, SGD, GaussianNB)

All supervised strategies follow the same evaluation logic:

- 5-fold stratified cross-validation on the train split for the binary task
- use fixed model parameters (no hyperparameter search)
- fit the final binary model on the full train split
- choose the binary decision threshold on the full validation split
- evaluate the final binary model on test
- train a separate multiclass model on the train split
- evaluate the multiclass model on test

Class names for multiclass evaluation are the union of train and test labels, ensuring rare classes are never silently dropped from metrics.

This is a practical engineering choice to keep the benchmark runnable on a workstation.
If a future study wants parameter search, it should be done explicitly outside this baseline CLI.

#### Autoencoder

The reconstruction-based anomaly detector:

- trains an MLPRegressor (64-16-64 bottleneck) to reconstruct benign samples only
- uses mean squared reconstruction error as the anomaly score
- calibrates the threshold on validation percentiles
- reports per-attack detection rates

#### LOF (Local Outlier Factor)

The novelty detection baseline:

- fits LOF in novelty mode on a subsample of benign training data (max 30k samples to control memory)
- uses negated decision function as anomaly score
- same threshold calibration and per-attack reporting as Isolation Forest

#### Federated SGD

The federated learning strategy:

- simulates 5 clients via stratified IID partition of the training data
- runs FedAvg for 20 communication rounds with SGD (log_loss, constant lr=0.01)
- each round: clients train 1 local epoch, then weights are averaged proportional to client size
- evaluation follows the same binary + multiclass protocol as centralized strategies

### 8. Metrics and why they were chosen

For binary classification, the code reports:

- accuracy
- balanced accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix

For multiclass classification, the code reports:

- accuracy
- balanced accuracy
- macro F1
- weighted F1
- per-class precision / recall / F1 / support
- confusion matrix

These choices are scientifically defensible for IDS work because:

- **accuracy alone is misleading** on imbalanced attack datasets
- **balanced accuracy** penalizes classifiers that ignore minority classes
- **precision and recall** expose false positive / false negative trade-offs
- **F1** is a compact summary for attack detection quality
- **PR-AUC** is important when attacks are a minority class
- **macro F1** prevents the multiclass result from being dominated by large classes
- **per-class metrics** show where the model fails on rare or fine-grained attacks

### 9. Harmonized family taxonomy

The repository creates a shared `label_family` target across all datasets.
This harmonized family target is useful for:

- semantic comparison across datasets
- metadata consistency
- future family-level experiments
- reasoning about groups of attacks rather than only dataset-specific labels

Current important limitation:
the main benchmark still trains on dataset-native features and dataset-native labels.
The harmonized family target improves organization and future extensibility, but it does **not** by itself make cross-dataset comparison valid.

### 10. Why this is scientifically viable

The pipeline is scientifically viable for the claims it actually makes because:

- **the protocol is explicit**: preprocessing, splitting, validation, thresholding, and reporting are defined in code
- **the test set is protected**: model selection is done on validation, not on test
- **data leakage is actively controlled**: train-only scaling, train-only imputation, overlap removal, and overlap validation are all enforced
- **baselines are meaningful**: the framework compares a naive classifier, anomaly detectors (Isolation Forest, Autoencoder, LOF), linear models, neural networks, tree ensembles, and a federated learning setup
- **results are reproducible**: processed artifacts are versioned and validated
- **metrics match the problem**: the reported metrics are suitable for imbalanced intrusion detection
- **limitations are visible**: the framework does not hide that some tasks are harder than others and does not claim solved generalization where none exists

### 11. What this repository can claim

Under the current `v5` pipeline, this repository can legitimately claim:

- a clean, validated, same-dataset IDS benchmark on three public IoT datasets
- reproducible processed artifacts with a strict loading contract
- fair within-repo comparison across 13 strategies: dummy baseline, linear classifiers, neural networks, tree ensembles, anomaly detectors, and federated learning
- meaningful binary and multiclass evaluation

### 12. What this repository does not claim

This repository does **not** currently claim:

- exact reproduction of each original paper's full experimental protocol
- meaningful cross-dataset generalization
- temporal robustness or concept-drift robustness
- deployment-readiness in a live IoT environment
- universal superiority of one model outside the benchmark protocol used here

These limits matter.
The benchmark is scientifically useful because its scope is controlled and explicit, not because it tries to claim more than the code actually supports.

## Experiments

### Verified data state

- Preprocessed artifacts are currently at version `5` for all three datasets.
- `uv run python -m src.data.audit` passes with `0/0/0` overlap for all three datasets.
- Exact overlap checks on `train/val/test` are enforced at load time (`load_dataset` raises on any non-zero overlap).
- Verified artifact sizes after cleaning:

| Dataset | Rows after cleaning | Artifact version |
|---------|---------------------|------------------|
| `CICIoT2023` | `1,206,334` | `5` |
| `Edge-IIoTset` | `451,508` | `5` |
| `TON-IoT` | `834,859` | `5` |

### Verified model runs

All 13 strategies have been run on all three datasets under the current fixed-parameter pipeline. Results are stored in `results/<dataset>/<strategy>.json`.

## Results

### Current benchmark snapshot

#### Edge-IIoTset

| Model | Binary F1 | Balanced Acc. | Multiclass Macro F1 |
|-------|-----------|---------------|----------------------|
| `dummy` | 0.0000 | 0.5000 | 0.0556 |
| `gaussian_nb` | 0.7173 | 0.7797 | 0.3182 |
| `federated_sgd` | 0.7849 | 0.8294 | 0.3739 |
| `autoencoder` | 0.7885 | 0.8290 | - |
| `sgd` | 0.7902 | 0.8342 | 0.4355 |
| `logistic_regression` | 0.7966 | 0.8384 | 0.5011 |
| `mlp` | 0.8377 | 0.8607 | 0.6840 |
| `extra_trees` | 0.8698 | 0.9111 | 0.5502 |
| `random_forest` | 0.9246 | 0.9317 | 0.8639 |
| `isolation_forest` | 0.4917 | 0.6095 | - |
| `hist_gb` | 0.9541 | 0.9564 | 0.8279 |
| `xgboost` | 0.9526 | 0.9549 | 0.8817 |

#### CICIoT2023

| Model | Binary F1 | Balanced Acc. | Multiclass Macro F1 |
|-------|-----------|---------------|----------------------|
| `dummy` | 0.9803 | 0.5000 | 0.0066 |
| `gaussian_nb` | 0.9810 | 0.9788 | 0.3684 |
| `isolation_forest` | 0.9834 | 0.8546 | - |
| `autoencoder` | 0.9861 | 0.8886 | - |
| `sgd` | 0.9888 | 0.8401 | 0.4105 |
| `federated_sgd` | 0.9888 | 0.8834 | 0.4728 |
| `logistic_regression` | 0.9889 | 0.8774 | 0.5268 |
| `mlp` | 0.9925 | 0.9124 | 0.5848 |
| `extra_trees` | 0.9928 | 0.9133 | 0.5923 |
| `hist_gb` | 0.9936 | 0.9141 | 0.5218 |
| `random_forest` | 0.9936 | 0.9294 | - |
| `xgboost` | 0.9936 | 0.9260 | 0.6154 |

#### TON-IoT

| Model | Binary F1 | Balanced Acc. | Multiclass Macro F1 |
|-------|-----------|---------------|----------------------|
| `dummy` | 0.0000 | 0.5000 | 0.1134 |
| `isolation_forest` | 0.3430 | 0.6044 | - |
| `autoencoder` | 0.3893 | 0.6733 | - |
| `sgd` | 0.6488 | 0.8659 | 0.1869 |
| `gaussian_nb` | 0.6657 | 0.8967 | 0.2635 |
| `federated_sgd` | 0.6725 | 0.8822 | 0.2046 |
| `logistic_regression` | 0.6725 | 0.9001 | 0.2330 |
| `mlp` | 0.9742 | 0.9891 | 0.9452 |
| `extra_trees` | 0.9803 | 0.9929 | 0.9171 |
| `random_forest` | 0.9844 | 0.9937 | 0.9485 |
| `hist_gb` | 0.9868 | 0.9955 | 0.8710 |
| `xgboost` | 0.9869 | 0.9958 | 0.9623 |

Anomaly detection strategies (isolation_forest, autoencoder, lof) report per-attack detection rates instead of multiclass macro F1.

### Interpretation

- **Tree ensembles dominate**: XGBoost, HistGB, Random Forest, and Extra Trees consistently lead across all datasets.
- **XGBoost remains the strongest default**: best or near-best on all three datasets for both binary and multiclass.
- **MLP is the best non-tree model**: competitive with tree ensembles on TON-IoT (0.9452 macro F1) and significantly better than linear models everywhere.
- **Linear models plateau quickly**: Logistic Regression, SGD, and GaussianNB cluster together in binary performance but fall behind on multiclass, especially on TON-IoT.
- **Federated SGD loses a few points vs centralized SGD**: expected with 5-client FedAvg on stratified IID partitions; the gap is small on CICIoT2023 but more visible on harder datasets.
- **Anomaly detectors are useful baselines but lag behind supervised methods**: Isolation Forest and Autoencoder detect volume attacks (DDoS) well but miss subtle attack types.
- **The hard part is multiclass**: binary F1 is above 0.98 for most supervised models on CICIoT2023, but fine-grained attack recognition remains much harder.

## Intuition

- **Tree ensembles dominate the current benchmark**: the datasets are tabular, mostly structured, and respond very well to boosted or bagged trees.
- **XGBoost is the strongest default**: it leads or ties on all three datasets for both binary and multiclass metrics.
- **MLP bridges the gap**: the neural network is the strongest non-tree model, competitive with tree ensembles on TON-IoT and significantly better than linear models on multiclass tasks.
- **Linear models have a hard ceiling**: Logistic Regression, SGD, and GaussianNB are fast but plateau quickly on multiclass recognition.
- **Federated learning works but loses a few points**: the 5-client FedAvg setup with stratified IID partitions produces results close to centralized SGD, demonstrating that federated training is viable for IoT IDS.
- **The hard part is not binary separation anymore**: the real weakness is fine-grained attack family recognition, especially on `Edge-IIoTset` and `CICIoT2023`.
- **Accuracy is not enough**: for imbalanced datasets like `CICIoT2023`, a dummy model can still show high accuracy while being useless for balanced detection.
- **Cross-dataset generalization is still unresolved**: the current dataset-specific preprocessing yields no shared feature intersection across datasets, so same-dataset scores should not be mistaken for deployment robustness.


## Cross-Dataset Status

The cross-dataset entry point in `main.py` is currently **experimental**.
With the present dataset-specific feature engineering, the pairwise feature intersections are empty after preprocessing, so no meaningful cross-dataset run is executed yet.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
