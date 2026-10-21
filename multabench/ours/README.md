# The method in detail: `cte_depth_top3`

Cross-fitted Text Experts (CTE) with a linear probe and depth probes on a frozen e5-small-v2 encoder. This page
covers what the method computes, how to run it, what it needs, and where each part lives in the code. For the
motivation and the results, see the [main README](../../README.md).

## What the learner receives

For a table with structured columns and text columns, the learner (any of the five MulTaBench learners) gets:

| Feature block | Per | What it is | Uses labels? |
|---|---|---|---|
| structured columns | table | upstream's preprocessing, unchanged | no |
| PCA-30 | text column | frozen e5 mean-pooled embedding of `"passage: <column>: <value>"`, PCA fit on training rows (exactly Joint Frozen) | no |
| lexical expert | table | TF-IDF of all text columns concatenated (`"col: val \| ..."`, word 1–2-grams + character 3–5-grams, 50k features each), then Ridge; the Ridge strength is picked from {0.3, 1, 3, 10} by the out-of-fold score on training rows | yes, cross-fitted |
| neighbour expert | text column (+ concatenation) | the label average of the 20 most similar training rows (cosine similarity of frozen embeddings, 1/distance weights) | yes, cross-fitted |
| probe | table | Ridge on the full 384-d final-layer embeddings of all text columns | yes, cross-fitted |
| depth probes | table | Ridge on each of the 12 layers' mean-pooled states; the 3 layers with the best out-of-fold training score are averaged | yes, cross-fitted |
| numeracy | text column | 20 form features: count / first / max / min of the numbers in the text, years, currency and amounts, percentages, length in characters / words / sentences, word length, shares of capitals, digits and punctuation, `!` and `?` counts, vocabulary richness, empty cell | no |

**Targets of the experts.** Regression: the rank-Gaussian transform of y (robust to heavy tails). Classification:
one-hot class indicators; the expert outputs class scores, with a column budget (binary: 1 column; 3–10 classes:
C−1; more than 10 classes: the scores of the 10 most frequent training classes + max score + entropy).

**Cross-fitting.** The learner's training rows are split into 5 folds (stratified by class when every class has at
least 5 rows). Each fold's expert columns come from
experts fit on the other 4 folds, so no training row's feature saw its own label. The learner's validation rows and
the test rows get experts fit on all training rows. The same holds for the probe and every depth probe.

**What is never done:** fine-tuning the encoder, fitting anything label-aware on validation or test rows, or tuning a
setting on test scores. TF-IDF vocabularies and PCA are fit on training rows only.

## Requirements

| What | Needed for | How |
|---|---|---|
| Python 3.11 + pinned packages | everything | `uv sync --frozen` (from `pyproject.toml` + `uv.lock`), or `source init.sh` |
| Kaggle API credentials | downloading MulTaBench datasets (`--dataset`) | `KAGGLE_USERNAME`, `KAGGLE_KEY` in `.env` (copy `.env.example`); not needed for `--csv` |
| Prior Labs API key | the TabPFN-2.5 learner only | accept the TabPFN-2.5 license at [ux.priorlabs.ai](https://ux.priorlabs.ai), copy the API key from Account, add `TABPFN_TOKEN=<key>` to `.env` |
| GPU | optional | everything runs on CPU; a GPU speeds up the e5 encoding and TabPFN / TabM |

The e5 encoder (`intfloat/e5-small-v2`) and the TabPFN-v2 weights are downloaded from Hugging Face on first use.
Secrets stay in `.env`, which is git-ignored.

## Train and predict

```bash
# a MulTaBench dataset and fold: the benchmark's own loading, subsampling and train / test split
.venv/bin/python scripts/train.py --dataset MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --out_dir models/ps

# any CSV with a target column: the same protocol (test 10% capped at 2,000 rows, training capped at 10,000)
.venv/bin/python scripts/train.py --csv my_table.csv --target price --out_dir models/my_table

# new rows: a CSV with the training feature columns (other columns are ignored); nothing is re-fitted
.venv/bin/python scripts/predict.py --model_dir models/ps --csv new_rows.csv --out predictions.csv
```

| `train.py` option | Default | Meaning |
|---|---|---|
| `--dataset` / `--csv` | (one is required) | a MulTaBench dataset id (`multabench/datasets/all_datasets.py`) or a CSV file |
| `--target` | | `--csv`: the target column |
| `--task` | `auto` | `binary`, `multiclass` or `regression`; `auto` infers it from the target |
| `--learner` | `catboost` | `catboost`, `lgbm`, `tabm`, `tabpfnv2`, `tabpfnv2p5` |
| `--method` | `cte_depth_top3` | the final method; `cte_probe` (no depth probes) and `cte` (experts only) are the earlier, simpler versions |
| `--fold` | 0 | which of the 5 benchmark splits is held out as test |
| `--train_examples` | 10000 | cap on training rows (the benchmark's) |
| `--out_dir` | (required) | the model folder |

**Output.** The model folder holds the fitted experts (`vectorizers.joblib`, `ridge.joblib`, `knn_index.npz`,
`probe.joblib`, `depth.joblib`), the PCA (`pca.joblib`), the learner in its own format (`catboost.cbm`,
`learner_model.joblib` for LightGBM / TabM, `learner_model.tabpfn_fit` for TabPFN), the upstream preprocessing state
(`learner.joblib`) and `meta.json`. Next to it: `metrics.json` (the test score and, for MulTaBench datasets, the
leaderboard's Joint TAR and Joint Frozen scores for the same learner and fold), `test_rows.csv` and
`test_predictions.csv`. `predict.py` writes `prediction` (regression) or one `proba_<class>` column per class.

### Choosing the learner

`--learner` picks the model that makes the final prediction from our features. Measured end to end (`train.py`
from scratch, including the e5 encoding, then `predict.py`) on Product Sentiment fold 0 (4,582 training rows, one text
column) on a laptop CPU (Intel i7-13700H, no GPU), and per job on a Colab A100 machine during the benchmark runs:

| `--learner` | Laptop CPU: train | Laptop CPU: predict 509 rows | Colab A100: median job (max) over the benchmark | Needs |
|---|---:|---:|---:|---|
| `catboost` (default) | 3.4 min | 19 s | 7.1 min (2.9 h) | – |
| `lgbm` | 3.6 min | 19 s | 0.9 min (10 min) | – |
| `tabm` | 8.2 min | 31 s | 0.8 min (7 min) | – |
| `tabpfnv2` | > 12 min (stopped) | – | 3.3 min (12 min) | a GPU in practice |
| `tabpfnv2p5` | > 12 min (stopped) | – | 0.5 min (2 min) | a GPU in practice; `TABPFN_TOKEN` |

- The A100 times are as run: the LightGBM, TabM and TabPFN-2.5 jobs reused features already built by an earlier
  learner for the same dataset and fold, so for a single new run expect the CatBoost / TabPFN-v2 times plus the
  learner's own fit. CatBoost's maximum is Spotify Genres, 114 classes: CatBoost's multiclass fit dominates there.
- TabPFN warns that it is not meant for CPU use on more than about 1,000 rows; on this laptop both versions passed
  12 minutes, so use a GPU for them.
- Building the features themselves takes about 2.7 minutes per job on the A100 machine's CPU (median over 90 jobs);
  see notebook 02, section 8.

## Run the benchmark evaluation

One job = (method, learner, dataset, fold), one JSON result per job, skipped if it already exists:

```bash
.venv/bin/python scripts/run_experiment.py --method cte_depth_top3 --learner tabm \
    --dataset REG_TEXT_BOOK_PRICE --fold 0 --out results_runs/
.venv/bin/python scripts/run_experiment.py --jobs_file jobs/p15.txt --out results_runs/   # a whole jobs file
```

Useful options: `--parallel_jobs` (default `auto`: half the CPU cores for CPU-bound methods; use 2 for TabPFN on one
GPU), `--job_timeout_min` (default 30; the final runs used 240), `--filter` (run matching jobs only), `--max_jobs`.
Each JSON records the score, the leaderboard references for the same fold, runtimes, the git commit and the engine
version. [`notebooks/colab_run.ipynb`](../../notebooks/colab_run.ipynb) runs a jobs file on a Colab GPU with results
on Google Drive. The final runs: `jobs/p14_A.txt` (CatBoost), `jobs/p15.txt` (LightGBM, TabM), `jobs/p16.txt`
(TabPFN-v2, TabPFN-2.5), `jobs/p14_B.txt` (36 further datasets). Reports: `scripts/p14_report.py`, `p15_report.py`,
`p16_report.py` (`--out <results root>`); `scripts/make_results.py` and `scripts/build_results_notebook.py` rebuild
`results/` and notebook 02.

## Code map

| File | What it holds |
|---|---|
| [`cte.py`](cte.py) | the frozen-embedding cache (`frozen_embeddings_cached`), numeracy features (`numeracy_features`), the lexical and neighbour experts with cross-fitting (`compute_cte_features`, `ridge_fit`, `neighbour_average`), the column budget, `run_cte` |
| [`p7.py`](p7.py) | the probe: cross-fitted Ridge on the full embeddings (`xfit_linear`, `run_p7`, method `cte_probe`); the file is named after the development round that introduced it |
| [`depth.py`](depth.py) | depth probes: every layer's pooled states (`layer_states_cached`), the per-layer cross-fitted Ridge (`xfit_linear_eig`), the top-3 aggregation (`run_depth`, method `cte_depth_top3`) |
| [`experts.py`](experts.py) | fold construction (`expert_folds`), the rank-Gaussian target transform, the feature cache |
| [`learners.py`](learners.py) | the plug-in point: wraps each upstream learner so that its text-encoder step returns our features (`make_learner`) |
| [`stacking.py`](stacking.py), [`variants.py`](variants.py), [`tar_engine.py`](tar_engine.py) | method dispatch, engine versions and metric helpers, the e5 encoding helpers (and our re-implementation of Joint TAR) |
| [`cte_model.py`](cte_model.py) | the saved model folder used by `train.py` / `predict.py` |
| [`archive/`](archive/) | earlier versions of the method, kept runnable |

## Tests

`.venv/bin/python -m pytest tests` covers:
- the leakage rules: out-of-fold features never see their own label; PCA, TF-IDF and target transforms are fit on
  training rows only
- determinism, and bit-for-bit golden features for the expert core
- the frozen Joint TAR re-implementation
- the train → save → predict round trip for all five learners
- the report scripts, on synthetic results

One known difference: a reloaded TabPFN model reproduces the fit-time probabilities only to about 0.01 (predicted
classes agree on ≥ 99% of rows), while TabPFN's own save / load is exact; the other learners reproduce them to 1e-5.
