# PROGRESS — lab notebook

Read this at the start of every session. Append-only: new entries go at the bottom; never rewrite old ones
(fix mistakes with a new entry that references the old one).

Entry template:
```
## E<n> — <short title>   (YYYY-MM-DD, commit <hash>)
- Question:
- Command / config:
- Result:
- Hostile review: (each risk → evidence)
- Conclusion:
- Next step:
```

---

## Pre-registration

### Claim C1 — DRAFT (not yet frozen; confirm the test with the course forum, then mark FROZEN + date)
- **Method:** our method (TBD — chosen from candidates A–D in CLAUDE.md; name it here when frozen).
- **Baseline:** Joint TAR = `multimodal_state=ft` rows in `multabench/leaderboard/results/text/*.csv`
  (upstream precomputed, tag `paper_version`).
- **Metric:** upstream `test_score` — AUROC (binary), macro-OvR AUROC (multiclass), R² (regression).
- **Datasets:** the 20 MulTaBench text datasets (files in `multabench/leaderboard/results/text/`).
  TabPFN-v2 and TabPFN-v2p5 have no baseline on MUL_TEXT_SPOTIFY_GENRES and MUL_TEXT_WINE_REVIEW
  (too many classes) → for those two learners the test uses n = 18 datasets.
- **Folds:** 0–4, upstream splits (`SEED + fold`), train cap 10,000, test 10% capped at 2,000.
- **Learners:** CatBoost, LightGBM, TabM, TabPFN-v2, TabPFN-v2p5 (upstream default hyperparameters).
- **Statistical test:** per learner, Wilcoxon signed-rank test (two-sided) across datasets on the
  fold-averaged difference d = mean₅folds(method) − mean₅folds(Joint TAR); p < 0.05.
- **Success threshold:** significant **and** mean d > 0 for at least 3 of the 5 learners.
- **Also reported:** mean d per learner with a 95% bootstrap CI (resampling datasets, 10,000 resamples).

Changes to a frozen claim: append below with date and reason; never edit the text above.

---

## E0 — Environment setup   (2026-10-09, commit 413813a)
- Question: can we reproduce upstream's environment on WSL Ubuntu?
- Command / config: `source init.sh` (rewritten to use uv: Python 3.11.17, 232 pinned packages from
  `requirements.txt`; original pyenv path needs sudo + system pip, blocked by PEP 668).
- Result: install OK; `uv pip check` → all compatible; torch 2.7.1+cu126 (no GPU on laptop), tabstar 1.1.15,
  transformers 4.57.3, peft 0.18.0; `benchmark.py --help` runs; `.env` loaded, `WANDB_MODE=disabled`.
- Conclusion: environment works. Note `wandb_run` refuses to start if `WANDB_API_KEY`/`WANDB_ENTITY` are
  empty — keep the placeholders in `.env`.
- Next step: CPU smoke test against the baseline CSV.

## E1 — Smoke test: CatBoost / REG_TEXT_BOOK_PRICE / fold 0 / Joint Frozen   (2026-10-09, commit 413813a)
- Question: does our environment reproduce a precomputed baseline number?
- Command / config: `python benchmark.py --model cat --dataset_name REG_TEXT_BOOK_PRICE --fold 0 --multimodal_state all`
  (laptop CPU, i7-13700H, 7.5 GB RAM; upstream defaults: frozen e5-small-v2 → PCA 30 per text column, CatBoost
  2000 iters / early stop 50, seed 42).
- Result: **R² = 0.5602** (RMSE 0.2236) vs CSV **0.5551** → Δ = +0.0051. CSV CatBoost/all over 5 folds:
  0.5551, 0.5796, 0.5350, 0.5169, 0.5301 (sd ≈ 0.024). n_train 4491, n_test 498, 3 tabular + 5 text columns
  → 153 features. Runtime 91 min (CSV run: 215 s) — almost all E5 on CPU.
- Hostile review:
  - Test leakage via PCA/encoder? No: `fit_text_encoders_vanilla` fits PCA on the `x` passed from
    `TabularModel.fit` (`baselines/abstract_model.py:110`), after the test split and the 90/10 val split — log
    shows encoders fit on 4042 texts = 90% of 4491 train rows. E5 is frozen in this state.
  - Split mismatch vs CSV? Splits are deterministic: `random_state = SEED + fold` (`preprocessing/splits.py:34`),
    test = 10% (498 of ~4989). Same code tag as the CSV. Can't compare row IDs (CSV stores only scores).
  - Seeds: SEED = 42 (tabstar constants), CatBoost `random_state=SEED`; CatBoost on CPU in both runs.
  - Metric: `r2_score` from sklearn (`baselines/training/metrics.py:32`), same as CSV.
  - Train size: dataset < 10,000 cap, so the cap is irrelevant here.
  - Hyperparameters chosen on test: none — upstream defaults, nothing tuned.
  - Remaining explanation for Δ: E5 embeddings computed on CPU (here) vs GPU (CSV) → float differences → PCA →
    different tree splits. Δ ≈ 0.2 sd of fold-to-fold variation.
- Conclusion: environment validated. Hardware alone can move one fold by ~0.005, so only compare methods on
  fold-averaged, multi-dataset results — never single folds.
- Next step: choose the method (A–D); build the embedding cache + resumable JSON runner in `multabench/ours/`;
  run real experiments on Colab GPU.

## E2 — Phase 0 diagnostics on precomputed baselines   (2026-10-09, base commit 8a5903f; code committed with this entry)
- Question: how big, how noisy and how significant is Joint TAR's gain over Joint Frozen — i.e. how hard is C1?
- Command / config: `python scripts/phase0_diagnostics.py` → `reports/phase0/*.csv`; `notebooks/phase0.ipynb`
  (executed, outputs saved). Input: upstream CSVs only (1,920 runs, 384 dataset×learner×state cells, all 5 folds,
  no duplicates). CPU only; no E5, no fine-tuning. Dev deps (ipykernel, nbclient, nbformat) via `pyproject.toml` +
  `uv.lock`, whose 231 runtime pins are verified identical to `requirements.txt`.
- Result (summary):
  1. Our C1 test on TAR − Frozen: significant for 4/5 learners. CatBoost +0.020, LightGBM +0.025, TabPFN-v2 +0.016
     win on every dataset (W = 0); TabM +0.012 (16/20, p = 0.002); TabPFN-v2p5 +0.009 (14/18, W = 41 vs crit 40,
     p = 0.054) misses by one rank.
  2. Gains are small: median over datasets +0.011; mean without Jigsaw +0.011; Jigsaw alone +0.117.
  3. 88/96 dataset×learner cells have gain > 0; only 41/96 exceed TAR's fold std.
  4. Noise: median TAR fold std ≈ 0.012; paired (same-split) diff std is smaller → median per-dataset detectable
     gain (5 folds, paired t) = 0.0098 (unpaired 0.018). TAR's own gain clears it on only 10/20 datasets.
  5. Headroom ≈ 0: TAR is the best of the 4 conditions in 88/96 cells (max headroom 0.005, Fake Jobs).
  6. Complementarity: Joint Frozen beats the best unimodal by median +0.026 (max +0.171, Vancouver Salaries);
     text > structured on 15/20 datasets.
  7. TAR gain is unrelated to dataset size (Spearman 0.02; train is capped at 10k).
  8. Text length known only for cached Book Price (Synopsis ≈ 1,000 chars mean; other columns 14–44).
- Hostile review:
  - Pairing: Frozen and TAR folds are matched by fold id; both use `random_state = SEED + fold` from the same
    code tag (`preprocessing/splits.py:34`), so fold k is the same split. Pairing done per (dataset, learner, fold).
  - Missing cells: TabPFN has no SPOTIFY_GENRES / WINE_REVIEW → n = 18, handled explicitly (not imputed).
  - Metric direction: AUROC and R² both higher-is-better; negative R² values exist (min −0.038) and are kept.
  - Wilcoxon: scipy exact two-sided; critical W checked against tables (52 for n = 20, 40 for n = 18).
  - "Best of 4 conditions" picks the max on test scores — descriptive only, never used to choose a method.
  - MDE uses TAR − Frozen diff std as a proxy for our-method − TAR noise; could differ for our method.
- Conclusion: C1 is demanding but passable; it rewards consistency across datasets more than big wins.
  Representation must improve (not modality selection); text/structure complementarity favors ideas A/B;
  D alone touches only the 10 regression datasets; C needs a direct pilot to size.
- Next step: Dima chooses the main idea; pilot it on 4–5 datasets before freezing C1.

---

## Pilot protocol P1 — written 2026-10-09, BEFORE any pilot run. Do not edit; append changes below with date + reason.
- **Pilot set:** 5 small datasets, 3 folds, CatBoost only. Datasets: Book Price, Product Sentiment, Baby Products,
  Book Readability, Montgomery Salaries (`REG_TEXT_BOOK_PRICE`, `MUL_TEXT_PRODUCT_SENTIMENT`,
  `REG_TEXT_BABIES_PRICES`, `REG_TEXT_BOOK_READABILITY`, `REG_TEXT_MONTGOMERY_SALARIES`). Folds 0, 1, 2.
- **Variants:** reproduction of TAR (`tar_repro`), then A, B, C.
- **Advance rule:** a variant moves to the full run only if its mean gain over TAR is at least +0.005 and it wins on
  at least 4 of the 5 datasets.
- **Combinations:** if two variants pass, we test them together once.
- **Operational definitions** (drafted by Claude; Dima to confirm before the first A/B/C run, then they are fixed):
  - "TAR" in the advance rule = our `tar_repro` on the same 5 datasets × 3 folds (same code, hardware, seed 42),
    not the upstream CSV.
  - Per-dataset gain = mean over folds 0–2 of (variant − tar_repro); mean gain = mean over the 5 datasets;
    a "win" = per-dataset gain > 0.
  - Upstream hyperparameters (LoRA r16 top-3, lr 1e-4, ≤ 50 epochs, patience 3, batch 256), train cap 10,000.
  - `tar_repro` sanity check (not an advance criterion): per-dataset fold-mean difference vs the CSV `joint_tar`,
    flagged if larger than 2 × the CSV fold std.

Changes to P1:
- **Amendment 1 (2026-10-09): pilot set amended on 2026-10-09 BEFORE any pilot run, to include a second
  classification dataset.**
  - Added `MUL_TEXT_DATA_SCIENTIST_SALARY` (Data Scientist Salary, multiclass, 6 classes). Rule used: among the
    classification datasets in `datasets_summary.csv`, the smallest N × text length (text length measured on the raw
    data with upstream's text-column detection): 15,841 rows × 227 chars/row ≈ 3.6 M chars. Next: Women's Clothing
    6.2 M (5 classes), Fake Job Postings 16.4 M (smallest binary; long job descriptions, 1,290 chars/row),
    Kickstarter 19.3 M (binary), Jigsaw 29.7 M (binary). Product Sentiment (0.5 M) was already in the set.
  - Pilot set is now 6 datasets × folds 0, 1, 2, CatBoost only = 18 jobs (`jobs/pilot_tar_repro.txt`).
  - **Advance rule becomes:** mean gain over TAR ≥ +0.005 **and** wins on at least **4 of 6** datasets.
- **Operational definitions confirmed by Dima (2026-10-09)** — now fixed:
  - TAR = our own `tar_repro` on the same datasets and folds (not the upstream CSV).
  - Per-dataset gain = mean over folds 0–2 of (variant − `tar_repro`); mean gain = mean over the 6 datasets.
  - Win = per-dataset gain > 0.

## E3 — TAR engine + experiment runner + Colab notebook (no real results yet)   (2026-10-09, base 7a7a286; code in this entry's commit)
- Question: can we re-run Joint TAR ourselves, faithfully, with caching, resumability and a leakage diagnostic?
- Built: `multabench/ours/tar_engine.py` (re-implementation of `fit_text_encoders_tuned` + `finetune_e5_with_lora`
  with only dynamic padding / fp16+GradScaler on GPU / fixed seed changed; cache per dataset/fold/variant/seed/
  config-hash/rows-hash), `multabench/ours/learners.py` (upstream learners subclassed; only the text-encoder step
  swapped), `scripts/run_experiment.py` (single job or `--jobs_file`; one JSON per method/learner/dataset/fold;
  skip if exists), `scripts/progress_table.py`, `jobs/pilot_tar_repro.txt` (15 jobs), `notebooks/colab_run.ipynb`,
  `tests/test_tar_engine.py`. Nothing under `multabench/` outside `ours/` changed.
- Local CPU checks (tiny settings, scores meaningless): `--train_examples 300 --epochs 1 --micro_batch_size 8`:
  CatBoost/Book Price ran end-to-end in 4 min; LightGBM on the same fold hit the TAR cache (10 s); Product
  Sentiment (multiclass) and TabPFN-v2 (no learner val split → own cache entry) ran; re-running a finished job
  skipped it. `pytest tests/test_tar_engine.py`: 6 passed, incl. two parity tests against upstream code.
- Findings while building:
  - The settings `benchmark.py` actually passes for `ft` are `E5TrainArgs`: batch 256, weight decay 0.001,
    50 epochs, patience 3 (not `finetune_e5_with_lora`'s own defaults of 32 / 0.01 / 40).
  - Upstream TAR's E5 sees only the learner's training rows (after the learner's own 90/10 split, val ≤ 1,000),
    then splits those 90/10 again for early stopping. TabPFN has no learner split → a different fine-tune.
  - Regression needs ≥ ~223 train rows: the stratified E5 val split must contain all 20 bins (upstream crashes too).
  - Batch 256 × up to 512 tokens does not fit in 7.5 GB RAM; micro-batching (gradient accumulation, same math) and
    automatic halving on CUDA OOM added.
  - Which columns count as "text" depends on the data: Book Price has 5 text columns at full size but 4 with 300 rows.
- Hostile review:
  - Same fine-tuning inputs as upstream? Test runs upstream's `fit_text_encoders_tuned` with the fine-tune stubbed
    out and checks texts, labels and E5 train/val rows are identical (cls + reg). Passed.
  - Dynamic padding = upstream padding? Fresh LoRA (B = 0) model encoded both ways: max |diff| < 1e-5. Passed.
  - Test labels used for training? No: `run_tar` uses extras' y only in `_head_quality` (diagnostic); E5 trains on
    `prepare_e5_data(x_fit, y_fit)` only; PCA is fit on training-row embeddings only (`learners.py`).
  - Cache could return stale/wrong embeddings? Key includes config hash + hash of all row indices; every lookup
    checks a hash of the text against the cached one (mismatch test passed).
  - Remaining non-identical parts: RNG streams (init/shuffle/dropout), fp16 numerics, own AdamW loop instead of HF
    Trainer (same optimizer, schedule, clipping, early stopping rule). Expect fold-level differences of the order
    seen in E1 (~0.005); judged by `tar_repro` vs CSV in the pilot.
- Next step: Dima runs `jobs/pilot_tar_repro.txt` on Colab; then E4 = tar_repro vs CSV + leakage diagnostic.
