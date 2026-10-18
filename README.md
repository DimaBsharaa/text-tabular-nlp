# text-tabular-nlp

> **Credit / upstream.** This repository is built on top of
> [alanarazi7/MulTaBench](https://github.com/alanarazi7/MulTaBench) (tag `paper_version`), the code
> accompanying *MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image*
> ([arXiv:2605.10616](https://arxiv.org/abs/2605.10616)). All benchmark, preprocessing and evaluation
> code is theirs and is kept unmodified; our additions for the Technion 097215 NLP final project
> (Track 1: Algorithm, text-tabular) live in `multabench/ours/`.

## Our method: CTE + probe (`cte_probe`), Cross-fitted Text Experts

**Final method:** `cte_probe` (claim **C5**: +0.0141 vs leaderboard Joint TAR, 15/20 datasets, Wilcoxon p = 0.0023;
p = 0.014 after Holm over all 7 pre-registered tests). **No fine-tuning:** a frozen e5-small-v2 encoder, so **no
weights are uploaded to Hugging Face** (nothing is trained but the per-dataset experts and learner, which `train.py`
fits in seconds to minutes). The encoder is the public `intfloat/e5-small-v2`.

For a text-tabular table, CatBoost gets:
- the structured columns;
- frozen e5 PCA-30 per text column;
- **cross-fitted (out-of-fold) predictions of cheap text experts**: TF-IDF + Ridge on all text columns, kNN in the
  frozen embedding space, and a linear probe on the full 384-d embeddings;
- label-free numeracy features.

Every training row's expert features come from experts that never saw its label.

**One command** (train + evaluate on a MulTaBench dataset, or `--csv my.csv --target y` for any table):
```bash
uv sync --frozen && .venv/bin/python scripts/train.py --dataset MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --out_dir models/ps
```

Code:
- final method: [`multabench/ours/p7.py`](multabench/ours/p7.py) (`cte_probe`) on top of
  [`multabench/ours/cte.py`](multabench/ours/cte.py);
- packaged model: [`multabench/ours/cte_model.py`](multabench/ours/cte_model.py);
- label-free variant `tf_num_lsa` (claim C4, vs Joint Frozen): also in `p7.py`.

**Development path, kept for transparency, not needed to use the method:**
- `tar_engine.py` / `variants.py` (P1–P2, our TAR re-implementation and fine-tuning variants);
- `stacking.py` (P3, C1);
- `experts.py` (P5 v1 legacy);
- `p6.py` (C3);
- `p9.py` (P9 / C6).

Every claim, amendment and result is in [`PROGRESS.md`](PROGRESS.md). Results, figures and tables:
[`notebooks/project_analysis.ipynb`](notebooks/project_analysis.ipynb). Paper notes:
[`docs/paper_notes.md`](docs/paper_notes.md).

### Install, train, predict

```bash
uv sync --frozen        # Python 3.11 env from pyproject.toml + uv.lock (pinned); or: source init.sh
cp .env.example .env    # fill in KAGGLE_USERNAME / KAGGLE_KEY yourself (needed to download MulTaBench datasets)

# train on a MulTaBench dataset + fold (the benchmark's split) ...
.venv/bin/python scripts/train.py --dataset MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --out_dir models/ps_f0
# ... or on any CSV with a target column (same protocol: 10% test, capped at 2,000; train capped at 10,000)
.venv/bin/python scripts/train.py --csv my_table.csv --target price --out_dir models/my_table

# predict on new rows (a CSV with the training feature columns; other columns are ignored)
.venv/bin/python scripts/predict.py --model_dir models/ps_f0 --csv models/ps_f0/test_rows.csv --out predictions.csv
```

`train.py` writes a model folder:
- the fitted parts: TF-IDF vectorizers, Ridge, kNN index, PCA, CatBoost, the upstream preprocessing state, `meta.json`;
- `metrics.json`: the test-split score, plus the leaderboard's Joint TAR / Frozen for the same fold;
- `test_rows.csv` + `test_predictions.csv`.

`predict.py` re-fits nothing. Measured on a laptop CPU (i7-13700H, no GPU), Product Sentiment fold 0 (4,582 training
rows): training **36 s**, predicting the 509 test rows **15 s**, model folder ≈ 8 MB.

### Reproduce every table and figure from the exported results

The full runs ran on Colab ([`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb), jobs files in `jobs/`). Each
export zip holds one JSON per (method, learner, dataset, fold) under `results/`. Unzip each export into its own folder,
then:

```bash
.venv/bin/python scripts/phase0_diagnostics.py                       # reports/phase0/  (leaderboard CSVs only)
.venv/bin/python scripts/pilot_report.py --out <pilot export>/results --variant <run> ...  # reports/pilot/ (P1-P3)
.venv/bin/python scripts/full_report.py --out <E17 export>/results    # reports/full/c1_report.json  (claim C1)
.venv/bin/python scripts/cte_report.py  --out <E24 export>/results    # reports/cte/cte_report.json  (claim C2)
.venv/bin/python scripts/p6_report.py   --out <E25 export>/results    # reports/p6/p6_report.json    (claim C3)
.venv/bin/python scripts/collect_fold_scores.py --export <E17 export>/results --export <E24 export>/results \
    --export <E25 export>/results --out reports/tables/fold_scores.csv
.venv/bin/python scripts/build_analysis_notebook.py --execute        # notebook + reports/figures/ + reports/tables/
```

All the `reports/` outputs are committed, so the last command alone rebuilds the notebook, figures and tables.
Tests: `.venv/bin/python -m pytest tests`.

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
