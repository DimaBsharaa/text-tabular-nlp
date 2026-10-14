# text-tabular-nlp

> **Credit / upstream.** This repository is built on top of
> [alanarazi7/MulTaBench](https://github.com/alanarazi7/MulTaBench) (tag `paper_version`), the code
> accompanying *MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image*
> ([arXiv:2605.10616](https://arxiv.org/abs/2605.10616)). All benchmark, preprocessing and evaluation
> code is theirs and is kept unmodified; our additions for the Technion 097215 NLP final project
> (Track 1: Algorithm, text-tabular) live in `multabench/ours/`.

## Our method: Cross-fitted Text Experts (CTE) — quickstart

CTE adds no fine-tuning. For each text-tabular dataset, the learner (CatBoost) gets the structured columns, frozen
e5-small-v2 PCA-30 per text column, and **cross-fitted (out-of-fold) predictions of cheap text experts**: a TF-IDF +
Ridge lexical expert on all text columns, and a k-nearest-neighbour expert in the frozen embedding space. It also
gets label-free numeracy features. Code: `multabench/ours/cte.py`. Pre-registration and every result:
`PROGRESS.md`. Paper notes: `docs/paper_notes.md`.

```bash
uv sync --frozen                          # Python 3.11 env from pyproject.toml + uv.lock (or: source init.sh)
cp .env.example .env                      # then fill in KAGGLE_USERNAME / KAGGLE_KEY yourself (datasets download)

# train on one dataset + fold (any MulTaBench text dataset id) -> model.joblib + metrics.json (test score of the fold)
.venv/bin/python scripts/train.py --dataset REG_TEXT_BOOK_PRICE --fold 0 --out_dir models/book_price_f0

# predict on new rows (a CSV with the same feature columns; a target column, if present, is ignored)
.venv/bin/python scripts/predict.py --model_dir models/book_price_f0 --csv new_rows.csv --out predictions.csv

# reproduce the reports from a results folder (one JSON per method/learner/dataset/fold)
.venv/bin/python scripts/run_experiment.py --jobs_file jobs/cte_full.txt --out results_cte   # the full run (GPU advised)
.venv/bin/python scripts/cte_report.py --out results_cte          # claim C2 (+ subsets, LightGBM, ablations)
.venv/bin/python scripts/p6_report.py --out results_cte           # claim C3 (P6), after jobs/p6_secondary.txt
.venv/bin/python scripts/build_analysis_notebook.py --execute     # notebooks/project_analysis.ipynb + reports/figures
```

Colab: `notebooks/colab_run.ipynb` runs a jobs file with results on Google Drive. Tests:
`.venv/bin/python -m pytest tests`.

---

# MulTaBench

A benchmark for multimodal tabular learning: tables whose columns include images or free text, not
just numbers and categories. It is 20 image-tabular and 20 text-tabular curated datasets, with 40
further datasets released alongside them, 80 in total. The benchmark evaluates tabular learners
under a target-aware setting, where the image and text encoders are fine-tuned on the task rather
than used frozen.

**Paper**: [MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image](https://arxiv.org/abs/2605.10616)  
**Datasets**: [kaggle.com/chico89](https://www.kaggle.com/chico89/datasets)

## Getting started

```bash
source init.sh && source .venv/bin/activate
cp .env.example .env     # Weights & Biases, Hugging Face and Kaggle credentials

python benchmark.py --model light --dataset_name MUL_IMAGE_PETFINDER --fold 0 --multimodal_state all
```

`benchmark.py` is the single entry point: it evaluates one model on one dataset and logs the
result to Weights & Biases. `--help` lists the available models and the feature combinations each
run can use, from tabular-only through fully multimodal with fine-tuned encoders.

## Datasets

Every dataset is hosted on the [`chico89`](https://www.kaggle.com/chico89/datasets) Kaggle account
and downloaded on demand; the benchmark datasets are slugged `multabench-<name>` and the ones
released alongside them `multabench-full-<name>`.

| What you are looking for | Where it is |
|--------------------------|-------------|
| Which datasets are in the benchmark | `multabench/datasets/all_multabench_datasets.py` |
| A dataset's Kaggle slug and source | `multabench/benchmark/datasets/` |
| How a dataset was curated | `multabench/datasets/annotated/` |
| Size, task and feature counts per dataset | `multabench/leaderboard/results/datasets_summary{,_extra}.csv` |

The 40 datasets released alongside the benchmark were admitted on a weaker criterion and carry no
tier name of their own, so their scores must not be pooled with the 40 MulTaBench datasets.
