# Cross-fitted Text Experts: target-aware text features for tabular learning, without fine-tuning

An extension of [MulTaBench](https://github.com/alanarazi7/MulTaBench), the benchmark for tables with free-text
columns ([arXiv:2605.10616](https://arxiv.org/abs/2605.10616)). The benchmark's strongest text pipeline, **Joint TAR**,
fine-tunes the text encoder on each task. This repository presents a method that makes the text features target-aware
**without fine-tuning the encoder**. A frozen encoder is read by a few small models that predict the target
("experts"), each fitted without the label of the row it describes. With all five MulTaBench learners, the method
scores higher than Joint TAR with the same learner.

## Results

The official MulTaBench evaluation on its 20 text-tabular datasets, 5 folds each (AUROC for classification, R² for
regression), compared with the published Joint TAR score of the **same learner, dataset and fold**:

| Learner | Datasets | Joint TAR | Ours | Mean gain [95% CI] | Datasets won | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 20 | 0.811 | 0.831 | +0.019 [0.012, 0.027] | 18 | < 0.0001 |
| LightGBM | 20 | 0.798 | 0.817 | +0.019 [0.005, 0.031] | 16 | 0.007 |
| TabM | 20 | 0.819 | 0.838 | +0.019 [0.011, 0.027] | 17 | < 0.0001 |
| TabPFN-v2 | 18 | 0.783 | 0.815 | +0.033 [0.017, 0.051] | 16 | 0.0003 |
| TabPFN-2.5 | 18 | 0.807 | 0.829 | +0.023 [0.013, 0.033] | 18 | < 0.0001 |

![Mean score per learner: Joint TAR vs ours, and the gain with its 95% interval](results/figures/fig1_mean_by_learner.png)

Left: mean scores (the axis starts at 0.70 and averages AUROC with R², as the benchmark's summaries do). Right: the
gain over Joint TAR with its 95% interval, the honest scale.

- **Significant with every learner** (two-sided Wilcoxon signed-rank test over datasets, each test specified before
  its run). With a Holm correction over the 24 pre-registered comparisons of the project, 4 of 5 remain significant
  (LightGBM: 0.095); over the five final comparisons alone, all 5.
- **Broad:** 83% of the 480 individual (learner, dataset, fold) runs beat Joint TAR; 16 of the 20 datasets are won
  with at least 4 of the 5 learners.
- **Jigsaw Toxicity**, where fine-tuning itself gains the most (+0.117 over the frozen encoder), was the one dataset
  Joint TAR won with every learner. The last expert added, an attention probe on the frozen token states, turns it
  into a win with every learner (+0.006 to +0.016).
- **One failure:** LightGBM fails on one fold of Spotify Genres (AUROC 0.536 against 0.906–0.915 on the other folds;
  0.913 when that job is re-run on an A100 GPU). It is reported as measured and makes LightGBM the one learner the
  attention probe does not help on average (−0.0006).
- **Checked with the version without the attention probe:** re-run with two other cross-fitting fold seeds
  (LightGBM), the gain is +0.017 and +0.018 (15 of 20 datasets each, Holm p = 0.019); on 36 further text datasets
  never used in development it is smaller (median +0.003, 24 of 36 won).
- TabPFN has no published baseline for 2 of the 20 datasets (Spotify Genres, Wine Review).

Full analysis, with every comparison and chart: [**notebook 02**](notebooks/02_results.ipynb). Scores per fold, dataset
and learner: [`results/`](results/); the result file of every run of the final method and of its version without
the attention probe (1,268 JSONs, with config, git commit and hardware): [`results/raw/`](results/raw/). The final
runs of CatBoost, LightGBM and TabM ran on a Tesla T4 GPU, those of TabPFN on an A100. Single scores differ slightly
between the two machines (GPU type, drivers, CPU threads; the attention probe trains on the GPU): 18 of the T4 jobs
re-run on an A100 differ by +0.0005 on average and by up to ±0.02 on the noisiest dataset (Video Games Sales) ([`reports/p18/a100_check.csv`](reports/p18/a100_check.csv)).

## Background

In a text-tabular dataset, some columns are numbers and categories and at least one is free text: a product
description, a review, a job posting. MulTaBench evaluates such data with one pipeline: each text column is embedded
with the e5-small-v2 encoder, the 384-dimensional embedding is reduced to 30 dimensions with PCA, and a tabular learner
(CatBoost, LightGBM, TabM, TabPFN-v2 or TabPFN-2.5) predicts from these and the structured columns. Two versions:

- **Joint Frozen:** e5 as published.
- **Joint TAR** (target-aware representations): e5's top layers are first fine-tuned with LoRA to predict the target
  from the text, then used for the embeddings. It is the stronger baseline.

## The problem with fine-tuning the encoder

Each one measured on the benchmark, while studying it or while developing the method:

1. **Over-trust.** The encoder is fine-tuned on the training rows and then embeds those same rows, so their text
   features look better than they will on new rows: TAR's prediction head scores +0.031 higher on its own training
   rows than on test rows (16 of 18 runs on 6 datasets, p = 0.0002). A downstream learner may trust them too much; like
   point 2, this is a motivation, not a measured cause.
2. **PCA does not look at the target.** 384 → 30 dimensions keeps the directions in which the texts vary most, not
   necessarily the ones that predict the target. This is a motivation, not a measured cause of Joint TAR's shortfall (the
   MulTaBench paper's Figure 6 finds Joint TAR ahead of Joint Frozen at 15, 30 and 60 components alike); a probe on the full 384
   dimensions adds +0.0035 on top of the experts (CatBoost, 20 datasets).
3. **Only the last layer is read.** The layer from which a linear probe best predicts the target varies from layer 2 to
   layer 12, and in 14 of the 20 datasets it is an inner layer, not the final one (out-of-fold scores on training rows,
   20 datasets × 5 folds). The largest gain is Book Readability at layer 3, where surface form matters.
4. **Exact words and numbers get blurred.** A sentence embedding smooths over the specific tokens ("Riesling",
   "$45", "2015") that often carry the signal: adding only the cross-fitted lexical expert (TF-IDF + Ridge) to the
   standard frozen features already matches Joint TAR (±0.000).
5. **Fine-tuning inside every run.** Upstream fine-tunes the encoder again inside every learner's run: for every
   dataset, fold and learner.

## The method

```mermaid
flowchart LR
    T["text columns"] --> E["frozen e5-small-v2<br/>(never fine-tuned)"]
    T --> L["lexical expert<br/>TF-IDF words + characters → Ridge"]
    E --> P["PCA-30<br/>(as Joint Frozen)"]
    E --> N["neighbour expert<br/>labels of the 20 most similar rows"]
    E --> F["probe<br/>Ridge on the full 384-d embedding"]
    E --> D["depth probes<br/>Ridge on each of 12 layers, best 3 averaged"]
    E --> A["attention probe<br/>learned token weights on the best layer"]
    T --> M["numeracy features<br/>numbers, years, currency, length"]
    L & N & F & D & A --> X["cross-fitting<br/>5-fold out-of-fold predictions"]
    S["structured columns"] --> G["any MulTaBench learner"]
    P --> G
    X --> G
    M --> G
```

The learner receives everything Joint Frozen gives it, plus a few columns that are **predictions of the target from
the text**, each from a different view: the exact words (lexical expert), similar rows (neighbour expert), the whole
final-layer embedding (probe), the best inner layers (depth probes) and the tokens that matter most (attention probe:
a small learned head that weights the frozen token states of the best layer, so one decisive word is not averaged
away). Each answers one problem above: cross-fitting (1), the probe on all 384 dimensions (2), the depth probes (3),
the lexical expert, the attention probe and the numeracy features (4), and features built once per dataset and fold
for all learners, with the encoder never fine-tuned (5).

**Cross-fitting** is what makes this work. The training rows are split into 5 parts; each part's expert predictions
come from experts trained on the other 4, so each training row's feature comes from a model that never saw that row's
label, as for a test row. Test rows get predictions from experts trained on all training rows (the attention probe
instead gives them the mean of its 5 fold models). The few selection
choices (the Ridge penalty from a small grid, the three depth layers) are made once from all training rows'
out-of-fold scores, never from validation or test labels. The same lexical expert without cross-fitting
(trained on all training rows, then used on them) scores 0.034 lower, well below Joint TAR.

Everything that uses labels is fit on training rows only, and the encoder is never fine-tuned: it is the public
`intfloat/e5-small-v2`, and no pretrained weights are needed beyond it (the attention probe's small heads are trained
inside each run, like the Ridge experts). Details, every option and runtimes:
[**method README**](multabench/ours/README.md).

## Quickstart

```bash
uv sync --frozen            # Python 3.11 environment, pinned by uv.lock
# only for MulTaBench datasets (--dataset): create a file .env with two lines, KAGGLE_USERNAME=... and KAGGLE_KEY=...
# (do not copy .env.example: its placeholder HF_TOKEN makes Hugging Face reject the public e5 download)

# train on a MulTaBench dataset and fold (the benchmark's own split), or on any CSV with a target column
.venv/bin/python scripts/train.py --dataset MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --out_dir models/ps
.venv/bin/python scripts/train.py --csv my_table.csv --target price --out_dir models/my_table --learner lgbm

# predict new rows with the saved model folder
.venv/bin/python scripts/predict.py --model_dir models/ps --csv new_rows.csv --out predictions.csv
```

Learner choice, API keys (TabPFN-2.5 needs one), measured runtimes and the benchmark runner: [method README](multabench/ours/README.md).

## Model weights and Hugging Face

The method pretrains and fine-tunes nothing. Its only pretrained models are public and load from Hugging Face on
first use: the frozen encoder `intfloat/e5-small-v2` (never updated) and the TabPFN learners' weights. What the method
learns (the Ridge experts, the attention probe's small heads, the PCA and the tabular learner) is fit from scratch on
each dataset's training rows inside every run and saved in that run's model folder (`attn.joblib`, `ridge.joblib`, …).
There is therefore no single trained model to publish: `scripts/train.py` is the end-to-end training mode and
`scripts/predict.py` the inference mode that loads a saved model folder.

## Reproduce the evaluation

```bash
# the 480 final runs (5 learners x 20 datasets x 5 folds; TabPFN: 18 datasets), about 47 GPU-hours one job at a time
.venv/bin/python scripts/run_experiment.py --jobs_file jobs/p18.txt --out my_results/
.venv/bin/python scripts/p18_report.py --out my_results/ --report_dir my_reports/   # the pre-registered test (claim C13)
```

[`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) runs the same jobs file on a Colab GPU. From the committed
results, `scripts/make_results.py`, `scripts/paper_assets.py` and `scripts/build_results_notebook.py` rebuild
`results/`, every number of the paper and notebook 02; `scripts/check_recompute.py` and `scripts/check_fold_scores.py`
check them against the raw result files ([`results/README.md`](results/README.md)).

## Is this the official benchmark?

- The upstream code is included unmodified, at tag [`paper_version`](https://github.com/alanarazi7/MulTaBench/tree/paper_version).
  For the 20 text datasets its splits, metrics, preprocessing and published baseline scores are identical to upstream
  `master` of 26 July 2026 (commits `599bd6a` and `c955466`); `git diff` over those paths is empty.
- Our code plugs in at the single text-encoder step of upstream's learners
  ([`learners.py`](multabench/ours/learners.py)); the learners, their settings, the splits and the metric functions
  are upstream's own.
- Our re-implementation of Joint TAR reproduces the published scores (mean difference −0.002).
- Every headline number is recomputed from the committed raw result files (`results/raw/`) by a script that uses no
  project code ([`check_recompute.py`](scripts/check_recompute.py)), and with shuffled training labels the method falls to chance
  ([`check_label_shuffle.py`](scripts/check_label_shuffle.py)).

## Notebooks

| # | Notebook | What it answers |
|---|---|---|
| 01 | [Benchmark study](notebooks/01_benchmark_study.ipynb) | What the published baselines show before building anything: how much fine-tuning adds per dataset and learner, and how large the fold noise is |
| 02 | [Results](notebooks/02_results.ipynb) | Every comparison with the baselines: per learner, per dataset, consistency, noise, task type, class imbalance, ablations, the 36 further datasets, trust checks |
| – | [Colab runner](notebooks/colab_run.ipynb) | Runs a jobs file of experiments on a Colab GPU, results to Google Drive |

## How we got here

The method was reached in steps, each tested on development splits of the training rows and then, once specified in
advance, on the full benchmark. Fine-tuning variants that put the structured columns into the encoder's input did not
beat Joint TAR. Cross-fitting the fine-tuned head fixed the over-trust but cost 22 GPU-hours and was no better, which
raised the question of whether the target-awareness needs fine-tuning at all. Cheap frozen experts matched Joint TAR;
a probe on the full embedding, then probes on the inner layers, took it clearly past, and an attention probe on the
frozen token states removed the last dataset Joint TAR won with every learner (Jigsaw Toxicity). Every stage, with its numbers:
[`archive/README.md`](archive/README.md).

## Limitations

- **Smaller gains on new data.** On 36 further datasets the gain is positive but small, and not significant after
  correcting for the number of tests; this was measured before the attention probe existed.
- **The 20 datasets were seen during development.** Early on, their test scores were looked at; later ideas were
  compared on splits of the training rows only and each final claim was specified before its run, but the method was
  chosen on these datasets, so they are not a pristine holdout (the attention probe, for example, was motivated by
  the known result on Jigsaw Toxicity). The 36 datasets never used show a smaller gain.
- **One learner seed** (42), as in the published baselines; the method's own randomness, the assignment of rows to
  cross-fitting folds, was checked with two more seeds (version without the attention probe).

## Repository layout

```
multabench/ours/          the method (cte.py, p7.py, depth.py, attn.py) and its plumbing; README with every detail
multabench/ours/archive/  code of earlier versions, kept runnable
scripts/                  train / predict, the benchmark runner, report and check scripts
results/                  final scores per fold, per dataset and per learner; figures
reports/                  the pre-registered analysis of each final claim (p14 to p18), checks, tables
notebooks/                01 benchmark study, 02 results, the Colab runner
jobs/                     the jobs files of the final runs
paper/                    the paper (LaTeX)
archive/                  development history: lab notebook, earlier raw results, reports, jobs, scripts
tests/                    unit and end-to-end tests
multabench/ (the rest), benchmark.py   upstream MulTaBench, unmodified
```

Built on [MulTaBench](https://github.com/alanarazi7/MulTaBench) by Arazi et al.; all benchmark, preprocessing and
evaluation code is theirs.
