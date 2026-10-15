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

## Ledger (one line per entry; details below)
| entry | date | what | key number |
|---|---|---|---|
| E0 | 2026-10-09 | environment (uv, Python 3.11, pinned deps) | 232 packages, all compatible |
| E1 | 2026-10-09 | CPU smoke test, CatBoost / Book Price / fold 0 / Frozen | R² 0.5602 vs CSV 0.5551 |
| E2 | 2026-10-09 | Phase 0 diagnostics on upstream CSVs | TAR > Frozen significant for 4/5 learners; median gain +0.011 |
| E3 | 2026-10-09 | TAR engine + runner + Colab notebook | parity tests vs upstream pass |
| E4 | 2026-10-09 | first GPU job, tar_repro / CatBoost / Book Price / fold 0 | R² 0.5736 vs CSV 0.5793; 40.8 min on T4; head train−test +0.048 |
| E5 | 2026-10-09 | fast engine (cached frozen layers + bucketing) + compute budget | CPU 10.7× on Book Price; old-engine pilot 12.4 GPU-h/variant |
| note | 2026-10-09 | tar_repro frozen at 96c1f5c; golden-fingerprint test | 14 tests pass |
| E6 | 2026-10-09 | T4 speed benchmark, Book Price fold 0 | fast 11.9×/epoch; pilot 1.1 GPU-h, full 9.6 GPU-h per variant |
| E7 | 2026-10-09 | Book Price fold 0, fast vs old engine | same fine-tune quality; CatBoost 0.5608 vs 0.5736 (noise ~0.01) |
| P1 am. 2 | 2026-10-09 | A/A control (seed 43) + engine-bias check; C1: leaderboard TAR primary, tar_repro secondary | awaiting the 18 tar_repro results |
| E8 | 2026-10-09 | tar_repro pilot seed 42 (18 jobs): engine bias + leakage | bias −0.0036 (p 0.15; Book Price/Readability −0.018/−0.011); leakage gap +0.031 (16/18, p 0.0002) |
| E9 | 2026-10-10 | A/A control, seed 43 vs 42 | does not pass (+0.0028, 3/6) → advance rule unchanged; seed sd per fold ≈ 0.008 |
| P1 am. 3 / P2 | 2026-10-10 | control = mean of seeds 42+43; variant spec + predictions (docs/method_spec.md) | written before any variant result |
| E10 | 2026-10-10 | variant (a) offset implemented + tested | jobs/pilot_offset.txt ready |
| E11 | 2026-10-10 | variants (b) prefix, prefix_offset implemented + tested | jobs/pilot_prefix*.txt ready |
| E12 | 2026-10-10 | variants (c) context, context_offset + sensitivity probe implemented + tested | jobs/pilot_context*.txt ready |
| E13 | 2026-10-10 | P2 pilot: 5 variants vs control | none passes; offset +0.0031 (4/6) best; context −0.022 |
| P3 | 2026-10-10 | stacking: cross-fitted head predictions as features (4 variants) + decision rule | written before any P3 run |
| E14 | 2026-10-10 | P3 implemented + tested | jobs/pilot_p3.txt ready (54 jobs, ≈ 5 GPU-h) |
| E15 | 2026-10-10 | P3 pilot, 3 of 4 variants | none passes; xfit_offset_pred +0.0035 (6/6) best; frozen_xfit_pred failed (GPU check), fixed |
| E16 | 2026-10-10 | P3 complete; decision rule | frozen_xfit_pred +0.0040 (4/6) highest → final method for the full run |
| P4 | 2026-10-10 | C1 FROZEN (CatBoost vs leaderboard Joint TAR) + full-run plan | jobs/full_part{1,2}.txt; ≈ 20 h A100 / 34 h L4 |
| E17 | 2026-10-11 | FULL RUN part 1 (400 jobs, A100) | **C1 fails**: CatBoost −0.0025 vs leaderboard TAR, p = 0.41; Video Games −0.056 |
| E18 | 2026-10-12 | exploration: frozen + cheap OOF text experts (CPU) | beats LB TAR on 4 of 6 explored datasets |
| P5 | 2026-10-12 | experts recipe + claims C2 (tar_experts) / C3 (frozen_experts) frozen | jobs/p5_{A,B}.txt; ≈ 10–14 h A100 each |
| E19 | 2026-10-12 | post-freeze CPU check of P5 v1; stopped | PS: tar_experts ≈ tar_repro; frozen_experts BP f0 +0.024 vs LB TAR |
| P5 v2 | 2026-10-12 | CTE (no fine-tuning) + claim C2 frozen; supersedes P5 v1 (never run) | jobs/cte_full.txt |
| E20 | 2026-10-12 | CTE CPU smoke + parallel lexical views | PS f0 0.8857, BP f0 0.6074; est. 20–40 h |
| P5 v2 am. 1 | 2026-10-12 | speed: TF-IDF once, fixed Ridge, concatenation-only lexical, parallel jobs (cte/2) | written before any full-run result |
| E21 | 2026-10-13 | profiling before/after am. 1 | experts ~40× faster; multiclass CatBoost dominates; Jigsaw drop |
| P5 v2 am. 2 | 2026-10-13 | C > 10 column budget + OOF alpha (cte/3); held-out 13 / strict 11 subsets | disclosure of fold-0 scores seen |
| E22 | 2026-10-13 | re-timing after am. 2 | Wine 18 min/job (CatBoost); Jigsaw 0.8695 vs LB 0.9137 |
| P6 | 2026-10-13 | exploratory CatBoost variants (native text, plus_native, nb, residual) | written before any P6 code/run |
| E23 | 2026-10-14 | P6 runtime check (laptop) | native text 6–8 min/job, Wine > 1.8 h (stopped) |
| E24 | 2026-10-14 | P5 v2 full run (600 jobs, A100) | **C2 fails**: +0.0106, p = 0.058; held-out 13 p = 0.0105; LightGBM p = 0.021 |
| P6 am. 1 / C3 | 2026-10-14 | BoW only for C > 10 (p6/2); claim C3 (3 CTE variants vs LB TAR, Holm) | written before any P6 result |
| P6 am. 2 | 2026-10-15 | native-text variants dropped (runtime); C3 = cte_nb + cte_residual, Holm over 2 | decided before any cte_plus_native result |
| E25 | 2026-10-15 | P6 results (200 jobs) | **C3 fails**: cte_nb Holm p = 0.097, cte_residual Holm p = 0.097; held-out 13 passes both |
| E26 | 2026-10-15 | packaging: train.py / predict.py, model folder, CPU smoke | PS f0: train 36 s, predict 15 s, round trip exact |
| P7 / E27 | 2026-10-16 | dev round (training rows only, 6 datasets) | tf_num_lsa +0.018 vs frozen (6/6); cte_probe +0.005 vs CTE (5/6) |
| P8 | 2026-10-16 | C4 (label-free tf_num_lsa vs Joint Frozen) + C5 (cte_probe vs Joint TAR), Holm over 2; approved | jobs/p8_B.txt then jobs/p8_A.txt |
| E28 | 2026-10-17 | P8 part B (195 jobs, A100) | **C4 passes**: +0.0129 vs Joint Frozen, 18/20, p = 1.9e-5; no environment effect (p = 0.47) |

---

## Pre-registration

### Claim C1 — DRAFT (not yet frozen; confirm the test with the course forum, then mark FROZEN + date)
- **Method:** our method (TBD — chosen from candidates A–D in CLAUDE.md; name it here when frozen).
- **Baselines (draft revised 2026-10-09):** the claim is tested against BOTH
  1. **primary — the official leaderboard Joint TAR**: `multimodal_state=ft` rows in
     `multabench/leaderboard/results/text/*.csv` (upstream precomputed, tag `paper_version`). This test decides
     success, so an engine that runs systematically below the leaderboard can never inflate our result;
  2. **secondary — our own `tar_repro`** (frozen engine `tar_engine/2`, same folds, same learners): the same test,
     reported next to the primary one. It isolates the effect of our method from engine differences.
  Both are reported in the paper; if they disagree, both are shown and the disagreement is discussed.
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
- **Amendment 2 (2026-10-09): A/A control, added BEFORE any variant result exists** (no variant is implemented yet;
  only tar_repro seed 42 results exist).
  - Why: E7 showed that two equally good fine-tunes can move a fold's CatBoost score by ~0.01, so a variant that
    changes nothing might pass the advance rule by luck.
  - What: run tar_repro with seed 43 on the same 18 pilot jobs (`jobs/pilot_tar_repro_seed43.txt`, results in
    `tar_repro__s43`) and apply the advance rule to "seed 43 vs seed 42" exactly as to a variant
    (`scripts/pilot_report.py`).
  - Rule: if the A/A comparison PASSES the advance rule (mean gain ≥ +0.005 and ≥ 4/6 wins), the rule is tightened
    before any variant runs, e.g. the mean gain must exceed the A/A mean |gain| by a clear margin. Claude proposes the
    exact new threshold from the A/A numbers (`pilot_report.py` prints candidates: 2 × A/A mean |gain|, A/A mean
    |gain| + 0.005, 3 × A/A noise sd of the mean, A/A max |gain|); Dima sees it before it is frozen, and it is
    then recorded here as amendment 3. If the A/A comparison does not pass, the rule stays as is and the A/A
    numbers are reported as the measured noise level.
  - Engine-bias check (reported, not an advance criterion): after the 18 tar_repro jobs, mean and std of
    (tar_repro − leaderboard ft) per dataset and overall, with paired two-sided Wilcoxon tests over the 18 jobs and
    over the 6 dataset means (`pilot_report.py`).
- **Amendment 3 (2026-10-10): control = mean of tar_repro seeds 42 and 43, added BEFORE any variant result exists.**
  - For every (dataset, fold) the control score is the mean of `tar_repro` (seed 42) and `tar_repro__s43`
    (seed 43); the per-dataset gain is mean over folds 0–2 of (variant − control). Both control runs must exist.
  - Why: averaging two fine-tunes halves the control's seed variance: Var(variant − control) = σ² + σ²/2 = 1.5σ²
    instead of 2σ² with seed 42 alone (E9: per-fold seed sd σ ≈ 0.008).
  - Caveat (Claude): variants also run with seed 42; if same-seed runs are positively correlated (same LoRA init,
    same batch order), seed 42 alone would cancel part of that noise and the mean keeps only half of it. Not
    measured; judged smaller than the variance gain.
  - The advance rule itself is unchanged (mean gain ≥ +0.005 and ≥ 4/6 wins).

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

## E4 — First GPU job: tar_repro / CatBoost / Book Price / fold 0 (old engine)   (2026-10-09, commit 7002687, Colab T4)
- Question: does our TAR reproduction match the leaderboard on a real GPU run, and what does the leakage diagnostic show?
- Command / config: Colab `notebooks/colab_run.ipynb`, cell 7 with `JOB_FILTER = ["REG_TEXT_BOOK_PRICE", "fold 0"]`,
  `MAX_JOBS = 1` → `run_experiment.py --method tar_repro --learner catboost --dataset REG_TEXT_BOOK_PRICE --fold 0`;
  engine `tar_engine/1` (full-model path), upstream hyperparameters, fp16 on a Tesla T4. Log read from the
  notebook outputs Dima saved to GitHub (commit ebfdcf5; checked: no secrets in it).
- Result: **R² 0.5736 vs leaderboard `ft` 0.5793 (diff −0.0056)**; CSV CatBoost TAR folds 0–4 = 0.579, 0.580, 0.568,
  0.551, 0.546 (sd 0.016), so the gap is ~0.35 sd. Fine-tune: 3,638 rows × 5 text cols = 18,190 cells, 20 epochs
  (best 17, val loss 2.996 → 2.861; ln 20 = 2.996 = uniform over 20 bins), ~115 s/epoch = **6.3 ms/cell/epoch**,
  38.6 min TAR training, **40.8 min per job**.
- Leakage diagnostic (R² of the fine-tuning head itself, cell predictions averaged per row):
  e5_train 0.4193 | e5_val 0.3311 | learner_val 0.3531 | test 0.3709.
- Our reading:
  - The head is better on the rows E5 was trained on than on unseen rows: 0.419 vs 0.371 on test (+0.048) and
    0.363 on learner_val + test pooled (+0.057). Final train/val CE: 2.73 vs 2.86 nats. So the in-sample advantage
    exists but is moderate on this dataset.
  - This matters for the learner: CatBoost is trained on embeddings of e5_train + e5_val rows (90% in-sample) but
    early-stops and is tested on rows E5 never saw. Its trees can learn to trust text features that are "too good"
    on training rows. That is exactly what idea C (cross-fitted TAR) targets; this fold says the effect is real but
    not huge.
  - e5_val (404 rows) below test is small-sample noise (it is the early-stopping set, so if anything it is biased
    upward), not evidence of anything.
  - One dataset, one fold: no conclusion about C until the pilot shows the gap across 6 datasets × 3 folds.
- Hostile review:
  - Same split as the CSV? Same upstream code paths (`subsample_dataset`, tabstar `split_to_test`, seed 42 + fold);
    n_train/n_test 4491/498 as in E1. Score within 0.35 sd of the CSV value.
  - Metric: upstream `score_all_metrics` (sklearn R²) on the 498 test rows.
  - Test labels used? Only in `_head_quality` (diagnostic); not for training, early stopping, PCA or CatBoost.
  - Speed is the blocker: 12.4 GPU-hours per variant for the pilot on this engine (see E5).
- Next step: fast engine (E5), then re-run this job with it (the engine-1 result is kept as
  `fold0.tar_engine-1.json` for comparison) and continue the pilot.

## E5 — Fast TAR engine (tar_engine/2) + compute budget   (2026-10-09, base ebfdcf5; code in this entry's commit)
- Question: can fine-tuning be made much faster without changing the method?
- Built (only `multabench/ours/`, `scripts/`, `tests/`, notebook): frozen embeddings + layers 0–8 run once per unique
  cell string (eval mode, no grad) → fp16 np.memmap on local disk (offsets + lengths per string), content-keyed so
  seeds/learners/variants with the same strings share it; training and inference run only layers 9–11 (LoRA) + head
  on it. Length-bucketed batches (shuffle → chunks of 50 batches → sort by length → batches of 256 → shuffle batch
  order): same batch size, same optimizer steps per epoch. Explicit `attn_implementation="sdpa"`, one tokenization per
  unique string, deduplicated cells. Old path kept: `--no_fast` (results go to `tar_repro__nofast`). Falls back to the
  old path if the cache would not fit on local disk. `scripts/speed_benchmark.py` + Colab cell 6b;
  `scripts/compute_budget.py` → `reports/compute_budget.csv`; Colab cell 9 exports results as a zip.
- **Deviations from upstream introduced by the fast path** (documented in `tar_engine.py`):
  1. the frozen lower 9 layers (and embeddings) run without dropout during training (upstream: dropout 0.1 there too);
  2. batches are length-bucketed instead of fully random (same size and number of steps).
  Everything that is trained (LoRA on layers 9–11, their dropout, the head), the loss, optimizer, schedule and early
  stopping are unchanged.
- Parity (CPU, `pytest tests/test_tar_engine.py`: 12 passed):
  - eval mode, random non-zero LoRA: fast vs old embeddings max |diff| 2.0e-5 (cosine ≥ 0.999999), head logits
    4.8e-5 — only fp16 storage rounding;
  - dropout off everywhere + same batch order: one epoch on each path gives the same LoRA weights (< 1e-3) and the
    same val loss (< 1e-3);
  - both paths match upstream `encode_texts_with_e5` with a fresh LoRA; same fine-tuning inputs as upstream.
- CPU speed (provisional, laptop, `speed_benchmark.py --train_examples 300`): Book Price 242 s → 23 s per epoch
  (**10.7×**, cache build 7 s, epoch-1 val loss 3.0177 vs 3.0174); Product Sentiment (1 short column) 2.5×. The
  GPU speedup must be measured on Colab (cell 6b); the GPU may gain less than the CPU from skipping layers.
- **Compute budget** (`reports/compute_budget.csv`; 6.3 ms/cell/epoch, 20 epochs per fine-tune, cost ∝ cells):
  - old engine: **pilot 12.4 GPU-hours per variant** (6 datasets × 3 folds, CatBoost); **full run 108 GPU-hours per
    variant** (20 datasets × 5 folds, one fine-tune per dataset-fold, shared by CatBoost/LightGBM/TabM) and
    **219 GPU-hours** if TabPFN's own fine-tune (no learner val split → different rows; 18 datasets × 5 folds) is
    added. TAR baseline + 3 variants on the old engine ≈ 400–900 GPU-hours: impossible before Oct 26.
  - fast engine, IF the GPU speedup is like the CPU's (~10×): pilot ≈ 1.2 h, full ≈ 10 h (≈ 21 h with TabPFN) per
    variant. To be replaced by the measured T4 speedup (`compute_budget.py --speedup X`).
  - The per-cell model overstates short-cell datasets (Data Scientist Salary: 43 chars/cell, 85 min predicted) and
    understates long ones (Fake Job Postings: 591 chars/cell); Book Price, where 6.3 ms was measured, has 221.
- Next step: Colab → cell 6b (speed benchmark), then cell 7 on all pilot jobs; update the budget with the T4 speedup.

## Note (2026-10-09): tar_repro frozen at 96c1f5c
- The tar_repro pilot runs on Colab at commit 96c1f5c (`tar_engine/2`). From now on the tar_repro method must stay
  bit-for-bit the same: variants are added next to it, with their own engine versions; `ENGINE_VERSION` stays
  `tar_engine/2` (bumping it would make the runner re-run every pilot result).
- Guard: `tests/test_tar_repro_frozen.py` recomputes a fingerprint (E5 train/val rows, training/val texts, labels,
  classes, token ids, epoch-1 batch order, a 30k-cell bucketing probe, config dict and config hash for CPU and CUDA,
  lower-cache key) and compares it with `tests/golden/tar_repro_96c1f5c.json`, which was generated by running the
  same fingerprint code against a `git worktree` checkout of 96c1f5c (engine module loaded from the checkout,
  verified). Mutation check: changing the text prefix, the bucketing chunk size or the weight decay each fails it.
- Result JSONs now also carry top-level `engine_version` and `tar_path` (fast / full_model); JSONs from the running
  pilot have the version under `tar.config.engine_version` (same value). Metadata only — no effect on results.

## E6 — T4 speed benchmark: old vs fast TAR path   (2026-10-09, commit 96c1f5c, Colab Tesla T4, Colab cell 6b)
- Command: `scripts/speed_benchmark.py --dataset REG_TEXT_BOOK_PRICE --fold 0` (full 50-epoch LR schedule, 1 epoch timed).
- Result: 18,190 train cells, 12,941 unique strings (929,484 tokens; lower-layer cache 0.71 GB, built in 12 s).
  Old path 107.2 s/epoch (5.9 ms/cell), fast path 9.0 s/epoch (0.50 ms/cell) → **11.9× per epoch, 11.2× for a
  20-epoch fine-tune** including the cache build. Epoch-1 val loss 2.9956 (old) vs 2.9961 (fast), diff +0.0005.
  Peak GPU memory 11.26 GB on both: it is set by the longest batch (256 × 512 tokens through the 3 trainable layers),
  which is the same on both paths; the frozen layers never stored activations even on the old path.
- **Compute budget with the measured speedup** (`compute_budget.py --speedup 11.2`, `reports/compute_budget.csv`):
  **pilot 1.1 GPU-hours per variant; full run 9.6 GPU-hours per variant** (20 × 5, one fine-tune per dataset-fold),
  **19.6 GPU-hours** with TabPFN's separate fine-tune. Old engine for comparison: 12.4 / 108 / 219.
- Conclusion: the full benchmark is now feasible on Colab (TAR baseline + 3 variants ≈ 40–80 GPU-hours).
- Next step: run the tar_repro pilot (cell 7) on the fast engine.

## E7 — Book Price fold 0 re-run on the fast engine vs the old engine   (2026-10-09, commit 96c1f5c, Colab T4)
- Question: does the fast engine give the same tar_repro result as the old one on a real job?
- Source: Colab export `results_export.zip` (2 result JSONs, log, benchmark), read from Windows Downloads.
- Result (CatBoost / REG_TEXT_BOOK_PRICE / fold 0):
  | | old engine (tar_engine/1, E4) | fast engine (tar_engine/2) |
  |---|---|---|
  | best val loss (epoch) | 2.8606 (17 of 20) | 2.8626 (18 of 21) |
  | head R² e5_train / e5_val / learner_val / test | 0.419 / 0.331 / 0.353 / 0.371 | 0.426 / 0.331 / 0.358 / 0.372 |
  | **CatBoost R² (test)** | **0.5736** | **0.5608** |
  | TAR train / whole job | 38.6 / 40.3 min | 3.2 / 5.3 min (cache 19 s) |
  CSV `ft` 0.5793; CSV CatBoost TAR fold sd 0.016 → fast is −0.0185 (1.2 sd), old was −0.0056.
- Reading: the fine-tune itself is equally good on both engines (val loss +0.002, head R² within ±0.007 on every
  split), but the learner's score moved by −0.013. That is fine-tune-to-fine-tune noise reaching the learner:
  a different but equally good LoRA solution → different embeddings → different PCA features → different trees.
  (E1: a much smaller embedding perturbation, CPU vs GPU frozen E5, moved the score by +0.005.)
- Hostile review:
  - Is the fast engine worse? Not supported: every fine-tune metric is equal or slightly better; one fold cannot
    separate engine effect from noise. The pilot (18 jobs) will show whether tar_repro sits systematically below CSV.
  - Same split / commit / clean tree: both JSONs report the same n_train/n_test; fast run at 96c1f5c, not dirty.
- **Consequence for the pilot rule (P1):** if a single fine-tune moves a fold's score by ~±0.01 at random, a variant
  that changes nothing could still reach "mean gain ≥ +0.005" by luck. Over 6 datasets × 3 folds the mean
  difference of two runs has a noise sd of roughly 0.014/√18 ≈ 0.003 if all folds were this noisy (classification
  AUROC is usually less noisy). Proposal for Dima (not adopted): measure it directly with an A/A control — run
  tar_repro with a second seed (43) on the 18 pilot jobs (~1.1 GPU-h) and apply the advance rule to seed 43 vs
  seed 42 before any variant result exists. If the A/A run "passes", the bar is too low.
- Next step: run the remaining 17 pilot jobs; Dima decides on the A/A control.

## E8 — tar_repro pilot, seed 42: engine-bias check + leakage diagnostic (18/18 jobs)   (2026-10-09, commit 96c1f5c, Colab T4)
- Source: Colab export `results_export (2).zip` (18 result JSONs + logs); analysis `scripts/pilot_report.py` →
  `reports/pilot/engine_bias.csv`, `engine_bias_overall.json`, `leakage.csv`. All 18 jobs on `tar_engine/2`
  (fast path), seed 42, CatBoost; 66.7 job-minutes in total on a T4.
- **Engine bias** (tar_repro − leaderboard `ft`, per fold):
  | dataset | fold 0 | fold 1 | fold 2 | mean | std |
  |---|---|---|---|---|---|
  | Data Scientist Salary | −0.0017 | +0.0020 | −0.0039 | −0.0012 | 0.0030 |
  | Product Sentiment | +0.0203 | +0.0067 | −0.0084 | +0.0062 | 0.0144 |
  | Baby Products | +0.0008 | −0.0107 | +0.0151 | +0.0017 | 0.0129 |
  | Book Price | −0.0185 | −0.0116 | −0.0226 | −0.0176 | 0.0055 |
  | Book Readability | −0.0232 | −0.0028 | −0.0069 | −0.0109 | 0.0108 |
  | Montgomery Salaries | −0.0012 | +0.0053 | −0.0037 | +0.0001 | 0.0047 |
  Overall: mean −0.0036, std 0.0116, 12/18 negative; paired two-sided Wilcoxon over 18 jobs W = 52, p = 0.154
  (critical W for n = 18: 40); over the 6 dataset means p = 0.69. → **no significant overall bias**, but Book Price
  (3/3 folds negative, −0.018) and Book Readability (3/3, −0.011) are consistently below the leaderboard. They are the
  two datasets with the longest cells (mean 53 and 50 tokens vs 12–38 elsewhere), i.e. where the fast path's
  length-bucketing changes batch composition most (long Synopsis-like cells batched apart from short ones).
  Hypothesis, not shown: bucketing (or no dropout in frozen layers) costs ~0.01–0.02 on long-text multi-column data.
  On Book Price fold 0 the old engine gave −0.0056 (E4) vs fast −0.0185 (E7) with equal fine-tune quality.
- **Leakage diagnostic** (head quality of the fine-tuning head; AUROC for cls, R² for reg; mean over 3 folds):
  | dataset | e5_train | learner_val | test | gap (train − test) | gap vs learner_val |
  |---|---|---|---|---|---|
  | Data Scientist Salary | 0.836 | 0.804 | 0.804 | +0.032 | +0.032 |
  | Product Sentiment | 0.845 | 0.813 | 0.811 | +0.034 | +0.032 |
  | Baby Products | 0.465 | 0.424 | 0.466 | −0.001 | +0.041 |
  | Book Price | 0.410 | 0.354 | 0.357 | +0.053 | +0.056 |
  | Book Readability | 0.578 | 0.540 | 0.549 | +0.028 | +0.038 |
  | Montgomery Salaries | 0.582 | 0.568 | 0.541 | +0.041 | +0.014 |
  Overall gap e5_train − test: mean +0.031, std 0.023, positive in 16/18 jobs, Wilcoxon p = 0.0002; vs learner_val
  positive on all 6 datasets. → the fine-tuned encoder is consistently better on its own training rows (relative
  +4% for cls AUROC, +5–15% for reg R²). This is the in-sample effect idea C targets; whether it *hurts the learner*
  is not measured here (that is what a cross-fitted variant would test).
- Other observations: Baby Products ran the full 50 epochs (best 49) on all folds — upstream's 50-epoch cap binds
  (faithful, but its TAR is under-trained). Montgomery Salaries has 4 text columns at full size (summary: 3).
- Hostile review:
  - Old-engine Book Price JSON (`fold0.tar_engine-1.json`) excluded by `load_runs`; all 18 used files are engine 2.
  - Leaderboard values are the same (dataset, CatBoost, `ft`, fold) rows; folds 0–2; same commit for all jobs.
  - 3 folds per dataset: per-dataset means have large uncertainty (e.g. Product Sentiment std 0.014).
  - The primary C1 baseline is the leaderboard, so a lower engine can only make our result look worse, never better.
- Next step: A/A control (seed 43, running). Open question for Dima: diagnose the Book Price / Book Readability gap
  (e.g. same jobs with `--no_fast`, ~3 GPU-h, or fast path without bucketing as a separate run name).

## E9 — A/A control: tar_repro seed 43 vs seed 42 (P1 amendment 2)   (2026-10-10, Colab T4, code at d66a711+)
- Source: the 18 seed-43 `DONE score` lines from Colab cell 7 (pasted in two parts; the first paste was cut by the
  chat's 50,000-character limit) + the 18 seed-42 JSONs from E8. Scores have 4 decimals. To be re-checked against the
  seed-43 JSONs when they are exported.
- Advance rule applied to seed 43 vs seed 42 (`pilot_report.advance_rule`, the code that will judge variants):
  | dataset | gain (mean of folds 0–2) | fold sd |
  |---|---|---|
  | Book Price | +0.0104 | 0.016 |
  | Book Readability | +0.0102 | 0.011 |
  | Product Sentiment | +0.0058 | 0.010 |
  | Data Scientist Salary | −0.0001 | 0.002 |
  | Montgomery Salaries | −0.0022 | 0.005 |
  | Baby Products | −0.0077 | 0.011 |
  Mean gain **+0.0028**, wins **3/6** → **the A/A control does NOT pass** (needs ≥ +0.005 and ≥ 4/6).
- **Amendment 2 outcome: the advance rule stays unchanged** (as pre-specified for a non-passing A/A). No amendment 3.
- Measured noise level: per-fold seed-to-seed difference sd 0.0112 (single fine-tune sd ≈ 0.008); A/A mean |gain|
  0.0061, max |gain| 0.0104; noise sd of the 18-pair mean 0.0026. Simulating pure Gaussian noise with that sd
  (independent folds), the current rule (≥ +0.005, ≥ 4/6) is passed by luck 2.9% of the time (+0.0075: 0.2%).
  Reading guide: a single dataset moving by ~0.01 is within seed noise; only the 6-dataset pattern counts.
- Engine bias revisited: seed 43 − leaderboard mean −0.0008 (Wilcoxon over 18 jobs p = 0.12); two-seed average
  −0.0022 (p = 0.18). Book Price two-seed mean −0.012 (sd 0.004), Book Readability −0.006. The −0.018 Book Price
  gap in E8 was mostly seed luck; the bucketing hypothesis from E8 is weakened (not tested).
- Hostile review: noise simulation assumes independent, zero-mean fold differences; real folds share data, so the
  2.9% is approximate. Variant-vs-tar_repro comparisons will both use seed 42, the same setting as this A/A.
- Next step: Dima defines the variants; they are judged by the unchanged P1 rule.
- **Known limitation (added 2026-10-10, Dima's decision: not diagnosed further):** tar_repro sits slightly below the
  leaderboard on the two datasets with the longest texts — Book Price (two-seed mean −0.012) and Book Readability
  (−0.006). Overall bias with both seeds is −0.0022 (n.s.), the leaderboard is the primary baseline, and GPU time
  goes to the variants. Length-bucketed batching (or no dropout in the frozen layers) is a plausible but untested
  cause.

## Pilot protocol P2 — TAR variants. Written 2026-10-10, BEFORE any variant is implemented or run. Do not edit.
- **Spec:** `docs/method_spec.md` (sections 1–6 = Dima's spec; section 7 = details Claude resolved, to confirm).
- **Variants** (built and pushed in this order): (a) `offset`; (b) `prefix`, `prefix_offset`;
  (c) `context`, `context_offset`. Each is run on the 18 P1 pilot jobs (6 datasets × folds 0–2, CatBoost, seed 42,
  T4) and judged by the P1 advance rule (unchanged) against the P1 control (amendment 3).
- **Predictions (Dima, 2026-10-10):**
  - `offset` ≥ tar_repro, with larger gains where the structured model is strong and text is complementary;
  - `context` ≈ tar_repro or slightly worse;
  - `context_offset` ≥ `offset`;
  - `prefix` ≈ tar_repro (risk: it learns the structure and ignores the text);
  - `prefix_offset` ≥ `offset`, largest gains where Phase 0 complementarity is large.
- **Diagnostics recorded per job:** TAR head qualities; for offset variants struct_only_quality, combined_quality
  and lift on e5_train / e5_val / learner_val / test (spec section 5).

## E10 — P2 variant (a) `offset` implemented (no results yet)   (2026-10-10; code in this entry's commit)
- Built: `multabench/ours/structured_offset.py` (5-fold OOF CatBoost, 500 it., depth 6, seed 42, on the structured
  columns of TAR's E5 train + val pool; full fit only for importances + diagnostics), `multabench/ours/variants.py`
  (`run_variant`, engine `variants/1`), dispatch in `learners.py` (tar_repro branch unchanged), `--method offset`,
  `jobs/pilot_offset.txt` (18 jobs, seed 42). Shared training loop got optional per-cell logit offsets
  (`train_offsets` / `val_offsets`, default None = tar_repro).
- Classification: loss on `head(h) + log(clip(p_oof, 1e-3, 1))`, offset also in the early-stopping val loss.
  Regression: target = y − y_oof, 20 quantile bins with edges from the e5_train residuals. Embeddings / PCA / learner
  exactly as TAR; offset never given to the learner. Same strings as tar_repro → same lower-layer cache.
- Diagnostics in the JSON (`tar.*`): head_quality (vs the variant's target), struct_only_quality, combined_quality,
  lift, offset_info (top features, importances).
- Tests (CPU): OOF never uses a row's own label (memorizing test model); zero head → logits = log(offset);
  cls + reg end to end + cache hit; offset reuses tar_repro's lower cache; no structured columns → identical to
  tar_repro (delegated, same cache entry, same embeddings). tar_repro guards: input fingerprint (fast) and a new
  bit-for-bit training fingerprint (tiny CPU runs, fast + full path, cls + reg) generated from a 96c1f5c checkout.
- Tiny real-data check (CPU, 300 rows, 1 epoch): Book Price top structured features BookCategory, Genre, Ratings;
  Product Sentiment: Product_Type. Runs end to end.

## E11 — P2 variant (b) `prefix` and `prefix_offset` implemented (no results yet)   (2026-10-10; code in this entry's commit)
- Built: `multabench/ours/feature_tokens.py` (FeatureSpec / RowFeatures / FeatureTokenizer / PrefixSource) and the
  prefix branch of `run_variant`; `--method prefix|prefix_offset`; `jobs/pilot_prefix.txt`, `jobs/pilot_prefix_offset.txt`.
- Mechanism: top-8 structured features by full-fit CatBoost importance → one token each,
  `W_name · E5_frozen("passage: {name}") + value_enc(x)` (PLE over ≤ 16 pool quantile bins → Linear(16→384);
  categories with ≥ 3 pool occurrences get an embedding, others "other"; missing → learned vector). Tokens are
  concatenated to the cached layer-8 states at the input of layer 9; attention over text + feature tokens; mean
  pool over text tokens only. New parameters train with LoRA + head (same optimizer, lr, early stopping). Text
  inputs identical to TAR → tar_repro's lower-layer cache is reused. Embeddings stored per (row, column).
  `prefix_offset` adds the offset loss of (a).
- Tests (CPU): PLE smooth and monotone; masked feature tokens → embeddings equal TAR's (< 1e-5), visible tokens
  change them; rare/unseen categories → "other", upstream's missing placeholder → missing vector; prefix and
  prefix_offset end to end for cls and reg, no new lower-layer cache, cache hit on re-run.
- Tiny real-data check (CPU, 300 rows, 1 epoch): Book Price and Product Sentiment run end to end for both methods,
  lower-layer cache hit each time.

## E12 — P2 variants (c) `context` and `context_offset` implemented; sensitivity probe (no results yet)   (2026-10-10; code in this entry's commit)
- Built: `multabench/ours/context_text.py` (Verbalizer, capped token construction, ContextTokenStore) and the
  context branch of `run_variant`; `--method context|context_offset`; `jobs/pilot_context.txt`,
  `jobs/pilot_context_offset.txt`; `scripts/sensitivity_probe.py` (to run later).
- Mechanism: each cell's input is `passage: {context} | {col}: {val}`, context = top-8 features by full-fit
  importance as `name=1.23 (decile 4/10)` / `name=<≤ 30 chars>` / `name=missing`, joined by "; ", deciles from the
  pool; context ≤ 80 tokens, text truncated so the total ≤ 512. Target, loss, hyperparameters as TAR (context_offset:
  + offset loss). New lower-layer cache (texts change); context_offset reuses context's. Embeddings per (row, column).
- Disk: context caches are much larger (unique per row, +60–80 tokens per cell; Book Price at 300 rows 0.25 GB vs
  0.02 GB for TAR). `run_experiment --jobs_file` now prunes the oldest lower-layer caches above
  `--max_lower_cache_gb` (default 40) before each job, so the engine never falls back to the full-model path for
  lack of disk (that would mix paths within a variant). Disk/rebuild time only — no effect on results.
- Tests (CPU): verbalizer deterministic (row order irrelevant), categoricals cut at 30 chars, context capped at 80
  tokens, total = 512 for very long inputs, piecewise tokenization == whole-string tokenization when no cap is hit;
  context and context_offset end to end (cls, reg), cache hit on re-run.
- Sensitivity probe (tiny CPU check, Book Price, 300 rows, 1 epoch, 20 rows, top-2 features): mean cosine change
  tar_repro 0.00000 (by construction), prefix ~0.00001 (barely trained), context ~0.014.

## E13 — P2 pilot results: all 5 variants vs the control (mean of tar_repro seeds 42 + 43)   (2026-10-10, commit 746c2a9, Colab T4)
- Source: `MyDrive/.../exports/results_export_20261010_085057.zip` (90 variant JSONs + both control runs, logs,
  reports); one unattended run of `jobs/pilot_all_variants.txt`, AUTO_DISCONNECT. All 90 jobs at 746c2a9, clean tree,
  0 failures (no Traceback in any log), no variant delegated to tar_repro. Analysis: `scripts/pilot_report.py`
  → `reports/pilot/advance_<variant>.{csv,json}`. GPU time: offset 59 min, prefix 74, prefix_offset 69,
  context 139, context_offset 96 (sum of job times).
- **Advance rule (P1 + amendments 1–3; gain = mean over folds 0–2 of variant − control):**
  | dataset | offset | prefix | prefix_offset | context | context_offset |
  |---|---|---|---|---|---|
  | Data Scientist Salary | +0.0007 | +0.0003 | −0.0003 | +0.0016 | +0.0005 |
  | Product Sentiment | −0.0070 | −0.0053 | −0.0088 | −0.0141 | −0.0053 |
  | Baby Products | **+0.0122** | +0.0063 | +0.0013 | −0.0154 | −0.0122 |
  | Book Price | +0.0140 | −0.0283 | −0.0089 | −0.0760 | −0.0805 |
  | Book Readability | −0.0019 | +0.0004 | −0.0018 | −0.0145 | −0.0254 |
  | Montgomery Salaries | +0.0004 | −0.0123 | −0.0045 | −0.0148 | −0.0115 |
  | **mean gain / wins** | **+0.0031 / 4** | −0.0065 / 3 | −0.0038 / 1 | −0.0222 / 1 | −0.0224 / 1 |
  | passes (≥ +0.005 and ≥ 4/6) | no | no | no | no | no |
  → **No variant passes; per P1 none advances to the full run and no combination is tested.**
- **Predictions (P2) vs outcome:** offset ≥ tar_repro — weakly (+0.0031, 4/6; the A/A noise level was +0.0028,
  3/6, so not distinguishable from noise overall); context ≈ or slightly worse — clearly worse (−0.022);
  context_offset ≥ offset — no (−0.022); prefix ≈ tar_repro — slightly worse (−0.0065); prefix_offset ≥ offset — no
  (−0.0038).
- **Diagnostics:**
  - offset, per dataset: Baby Products +0.012 with fold sd 0.002 (3/3 folds up) is the only consistent gain;
    Product Sentiment −0.007 (sd 0.004) is a consistent loss; Book Price +0.014 is noise (sd 0.030). The offset
    model works as intended at the head level — structured-only vs structured + text-head on test: Data Scientist
    Salary AUROC 0.550 → 0.796, Product Sentiment 0.819 → 0.898, Baby Products R² 0.406 → 0.618, Book Price
    0.214 → 0.325 — but that complementary signal reaches the CatBoost score only on Baby Products.
  - context / prefix: the fine-tuning head is much better than TAR's on test (context: Montgomery R² 0.917 vs
    0.541, Book Readability 0.749 vs 0.549, Baby Products 0.617 vs 0.466; prefix similar) because it now sees the
    structured features — yet the downstream score drops. PCA-30 explained variance rises (Montgomery 0.963 context /
    0.996 prefix vs 0.800 TAR; Book Readability 0.977 / 0.994 vs 0.827): the embeddings collapse onto the
    structured information, which CatBoost already has as raw columns, and the text-specific detail is squeezed out
    of the 30 PCA components. Book Price (context −0.08) is the extreme case.
  - Leakage gap (head e5_train − test) persists in every variant (~0.01–0.05), similar to tar_repro (E8).
- Hostile review:
  - Control complete (18/18 pairs, both seeds); variants judged with the same code as the A/A control.
  - E9 re-check: seed-43 JSON scores equal the pasted ones (max |diff| 4.5e-5, 4-decimal rounding).
  - Same splits/rows: variants use TAR's E5 split; n_train/n_test identical to tar_repro per (dataset, fold).
  - Test labels: only in diagnostics (head/struct/combined quality); OOF offsets never see their own row's label
    (test 1); the full structured fit is never applied to pool rows.
  - context token budget: mean training cell 23–131 tokens (TAR 12–53), so no truncation pressure explains it.
  - The context/prefix explanation (redundancy + PCA collapse) is an interpretation of the PCA-variance and head
    numbers, not a tested causal claim.
- Conclusion: P2 is a negative result for all five variants on the pilot. The clearest lesson: making the text
  encoder see the structure (context, prefix) makes its *own* predictions better but its *embeddings* worse for a
  learner that already has the structure. offset is closest (+0.0031, 4/6) but below the bar.
- Next step: Dima decides (see options in the chat); any new variant needs a P3 pre-registration first.
- **Summary:** P2 failed: none of 5 variants passed; inference: the gain lives in the head's predictions and is lost
  through PCA'd embeddings, and encoders that see the table collapse onto it -> P3 tests cross-fitted head
  predictions as features.

## Pilot protocol P3 — cross-fitted head predictions as features (stacking). Written 2026-10-10, BEFORE any P3 run. Do not edit.
- **Why:** E13 — the text head's predictions carry the gain (offset: structured → structured + text head on test,
  e.g. Data Scientist Salary AUROC 0.550 → 0.796), but it is lost through the PCA'd embeddings; and in-sample head
  predictions on training rows would leak (E8 leakage gap +0.031). So: give the learner the head's predictions as
  extra features, cross-fitted so no training row's prediction comes from a model that saw its label.
- **Setting:** CatBoost, the same 18 P1 pilot jobs (6 datasets × folds 0–2), seed 42, T4; control = mean of tar_repro
  seeds 42 + 43 (P1 amendment 3); the P1 advance rule unchanged (mean gain ≥ +0.005 and ≥ 4/6 wins).
- **Variants:**
  - `insample_pred` (ablation): tar_repro features + the full-fit TAR head's predictions on all rows (training rows
    in-sample).
  - `xfit_pred`: tar_repro features + cross-fitted TAR head predictions. K = 3 folds over the E5 pool (stratified for
    classification); per fold, fine-tune exactly like tar_repro on the other 2/3 (same hyperparameters, early
    stopping on a split taken from those 2/3 only), predict the held-out third. Test and learner-val rows get the
    MEAN of the 3 fold models' predictions.
  - `frozen_xfit_pred`: Joint Frozen E5 PCA embeddings (as upstream `all`) + the same cross-fitted predictions as
    `xfit_pred` (reuses its fine-tunes; written by the same job).
  - `xfit_offset_pred`: tar_repro features + cross-fitted predictions of the OFFSET head (fold models trained with
    the offset loss, using the existing pool-level OOF structured offsets). The feature is the text head's own output
    without the offset added.
- **Prediction features, per text column:** classification = class probabilities (binary: 1 column; multiclass:
  C − 1 columns); regression = expected value over the 20 bin midpoints + the distribution's std. Offset heads: the
  same on the residual scale.
- **Predictions (Dima, 2026-10-10):** insample_pred ≤ tar_repro; xfit_pred > tar_repro; xfit_offset_pred ≥ xfit_pred,
  largest where the structured model is strong; frozen_xfit_pred ≈ xfit_pred.
- **Decision rule (pre-registered):** if no P3 variant passes the advance rule, the variant with the highest pilot
  mean gain goes to the full run as the final method; the full run is the real test.
- **Diagnostics per job:** quality of the prediction features on training rows (OOF for xfit, in-sample for
  insample) vs test — the leakage gap should close for xfit and stay open for insample; CatBoost feature importance
  of the new columns.
- **Implementation choices (Claude; same spirit as docs/method_spec.md §7):**
  1. Pool = the learner's rows handed to the text-encoder step (TAR's E5 train ∪ val). Folds:
     `StratifiedKFold(3, shuffle=True, random_state=42)` on the class label (cls; plain `KFold` if a class has < 3
     rows) and `KFold(3, shuffle=True, random_state=42)` (reg).
  2. Fold fine-tunes call the frozen tar_repro function (`tar_engine.run_tar`) on the fold's 2/3 (it does its own
     90/10 early-stopping split and its own 20 y-bins inside those rows); the held-out third, learner-val and test
     rows are passed as extra row sets (predicted, never trained on). Offset folds call `run_variant("offset")` the
     same way with the pool-level OOF structured model passed in (not refitted per fold).
  3. Regression bin midpoints = midpoints of the quantile edges of the fold model's own target bins (y for TAR heads,
     residual for offset heads); "std" = std of the head's bin distribution around its expected value.
  4. Per text column, a row's features come from that column's cell; the feature columns are appended to that text
     column's 30 PCA components (upstream names them `<col>_txt_pca_30…`), so no upstream code changes.
  5. insample_pred reuses the cached seed-42 tar_repro fine-tune (its head = the "full-fit TAR head").
- **GPU estimate (T4, from E8/E13 timings):** insample_pred ≈ 0.2 h (cache hits, CatBoost only); xfit_pred +
  frozen_xfit_pred ≈ 3 fine-tunes on 2/3 of the pool ≈ 2 × tar_repro ≈ 2–2.5 h; xfit_offset_pred ≈ 2–2.5 h.
  Total ≈ 4.5–5.5 GPU-hours.

## E14 — P3 implemented (no P3 results yet)   (2026-10-10; code in this entry's commit)
- Built: `multabench/ours/stacking.py` (`run_stacking`: insample_pred / xfit_pred / frozen_xfit_pred /
  xfit_offset_pred; K = 3 deterministic folds; prediction features; frozen embeddings; diagnostics),
  `PredColumnEncoder` in `learners.py` (appends the features after each text column's PCA components, inside the
  object upstream's `transform_text_features` calls), `run_variant(..., structured=)` for the offset folds,
  `run_experiment --method …` (+ companion: an xfit_pred job also writes frozen_xfit_pred; engine `stacking/1`),
  `jobs/pilot_p3.txt` (54 jobs: insample_pred, xfit_pred, xfit_offset_pred). `pilot_report.py` without `--variant`
  now judges every run it finds (so the AUTO_DISCONNECT report covers P3). Colab cell 7 now also echoes the
  variants' `[TAR:…]` / `[STACK:…]` lines.
- Fold fine-tunes are the unchanged tar_repro function (or `run_variant("offset")`) on the fold's 2/3 with the
  held-out third, learner-val and test as predict-only row sets; cached like any job. insample_pred and the base of
  every P3 job reuse the cached full-pool tar_repro fine-tune (cache hit on Drive for the pilot jobs).
- Tests (CPU): folds deterministic, disjoint, covering, stratified; a spy on `run_tar` shows no fold model trains on
  rows it predicts and the 3 held-out folds cover the pool; a direct tar_repro run on a fold's rows hits the fold
  model's cache entry (same code path, same key); all 4 methods × binary / multiclass / regression end to end with
  the right feature shapes (1 / C − 1 / 2 per text column); frozen embeddings = upstream frozen encoder (< 1e-3);
  through the real learner, the appended column reaches CatBoost (first real-data check caught that upstream calls
  the PCA object directly — fixed and covered by this test). tar_repro checks unchanged.
- GPU estimate unchanged from P3: ≈ 4.5–5.5 T4-hours for the 54 jobs.
- **Test-run incident (2026-10-10):** commit 8096a22 was pushed while 8 tests were failing (my commit command checked
  `tail`'s exit code instead of pytest's). Cause: the laptop's `/tmp` (a 3.8 GB RAM disk) had 2.5 GB free, so the
  frozen-layer cache's disk guard refused to build in the tests' temp folders (fast-path tests fell back / prefix
  asserted) — an environment problem, not P3 code. Fix: pytest temp folders moved to the real disk
  (`pyproject.toml`: `--basetemp=.pytest_tmp`), `/tmp` cleaned; full suite 54 passed, 1 skipped (slow bit-for-bit
  check, unchanged code path). Commits now require pytest's own exit code 0.

## E15 — P3 pilot results for 3 of 4 variants (frozen_xfit_pred failed to run)   (2026-10-10, commit 60b19ad, Colab T4)
- Source: `results_export_20261010_171243.zip`; one unattended run of `jobs/pilot_p3.txt`. 54 jobs at 60b19ad,
  clean tree; base tar_repro fine-tunes all cache hits. GPU time: insample_pred 27 min, xfit_pred 119 min,
  xfit_offset_pred 89 min. Reports: `reports/pilot/advance_{insample_pred,xfit_pred,xfit_offset_pred}.*`.
- **frozen_xfit_pred did not run:** in every xfit_pred job the companion step raised tabstar's
  `RuntimeError("No free GPU found!")` — `get_device()` refuses a GPU that already holds memory, i.e. our own
  process after xfit_pred. The xfit_pred JSONs were written before that, so they are complete. Fix (this entry's
  commit): one device per job, reused by the companion (+ test). Re-running `jobs/pilot_p3.txt` now only computes
  the 18 missing frozen_xfit_pred results (fold fine-tunes are cached on Drive).
- **Advance rule vs control (mean of tar_repro seeds 42 + 43):**
  | dataset | insample_pred | xfit_pred | xfit_offset_pred |
  |---|---|---|---|
  | Data Scientist Salary | +0.0002 | +0.0018 | +0.0011 |
  | Product Sentiment | +0.0083 | +0.0024 | +0.0048 |
  | Baby Products | +0.0056 | +0.0096 | +0.0026 |
  | Book Price | +0.0013 | +0.0061 | +0.0061 |
  | Book Readability | −0.0013 | −0.0022 | +0.0045 |
  | Montgomery Salaries | +0.0026 | +0.0019 | +0.0018 |
  | **mean gain / wins** | **+0.0028 / 5** | **+0.0033 / 5** | **+0.0035 / 6** |
  | passes (≥ +0.005, ≥ 4/6) | no | no | no |
- **Diagnostics:**
  - Leakage gap of the prediction features (quality on training rows − test): insample_pred +0.03…+0.05 (5/6
    datasets; Baby Products −0.005), xfit_pred and xfit_offset_pred ≈ 0 or negative (−0.07…+0.004). → cross-fitting
    closes the gap, as predicted. (Negative gaps: training rows get one fold model trained on 2/3 of the pool,
    test rows the mean of 3 models.)
  - CatBoost importance share of the prediction columns: insample_pred 0.12–0.44, xfit_pred 0.07–0.25,
    xfit_offset_pred 0.01–0.25 → CatBoost leans much harder on the leaky in-sample predictions.
  - Test quality of the prediction features (cls AUROC / reg R², mean over text columns): insample ≈ xfit_pred
    (e.g. Montgomery 0.618 / 0.599); xfit_offset_pred lower by construction (residual-scale head alone).
- Predictions (P3) so far: insample_pred ≤ tar_repro — no (+0.0028, 5/6); xfit_pred > tar_repro — yes but small
  (+0.0033, 5/6, below the bar); xfit_offset_pred ≥ xfit_pred — yes, marginally (+0.0035 vs +0.0033) and the only
  variant so far that wins on all 6 datasets; frozen_xfit_pred ≈ xfit_pred — pending.
- Noise reference: with the per-fold seed sd σ ≈ 0.008 (E9) and the two-seed control, a no-effect variant's
  per-dataset gain has sd ≈ 0.0057, so all 6 datasets positive by chance ≈ 1/64 (1.6%); mean ≥ +0.005 by chance
  rare too. The gains are consistent but small (+0.001…+0.006 per dataset).
- Hostile review: control complete; same rows/splits as tar_repro (base fine-tune cache hits = same rows hash);
  test labels only in diagnostics; fold models never trained on rows they predict (test suite, spy).
- **P3 decision rule not applied yet** — it needs all four P3 variants; waiting for frozen_xfit_pred.
- Next step: re-run `jobs/pilot_p3.txt` (≈ 30 min, only frozen_xfit_pred left), then apply the P3 decision rule.

## E16 — P3 complete: frozen_xfit_pred; P3 decision rule applied   (2026-10-10, commit 8daf8d5, Colab T4)
- Source: `results_export_20261010_191406.zip`; re-run of `jobs/pilot_p3.txt`, which only computed the 18 missing
  frozen_xfit_pred results (0 errors). All 18 reused xfit_pred's 3 fold fine-tunes (identical cache keys, all hits).
- frozen_xfit_pred vs control: Data Scientist Salary +0.0017, Product Sentiment −0.0011, Baby Products −0.0045,
  Book Price +0.0155, Book Readability +0.0101, Montgomery Salaries +0.0022 → **mean +0.0040, 4/6 wins, does not
  pass**. Prediction columns take 0.15–0.51 of CatBoost's importance (more than with tar_repro embeddings, 0.07–0.25).
- **All P3 variants:** insample_pred +0.0028 (5/6), xfit_pred +0.0033 (5/6), xfit_offset_pred +0.0035 (6/6),
  frozen_xfit_pred **+0.0040** (4/6); none passes (≥ +0.005 and ≥ 4/6).
- **P3 decision rule (pre-registered): none passed → the variant with the highest pilot mean gain goes to the full run
  as the final method → `frozen_xfit_pred`.** (xfit_offset_pred is the most consistent, 6/6, but the rule is mean gain.)
- P3 predictions: insample_pred ≤ tar_repro — no (+0.0028); xfit_pred > tar_repro — yes, small; xfit_offset_pred ≥
  xfit_pred — yes, marginally; frozen_xfit_pred ≈ xfit_pred — yes (+0.0040 vs +0.0033, slightly better).
- Reading: the final method uses NO fine-tuned embeddings: frozen E5 PCA embeddings (Joint Frozen) + K = 3
  cross-fitted TAR head predictions. Every training row's text-derived features come from models that never saw its
  label, so TAR's in-sample leakage is removed entirely; the y-information enters only through cross-fitted head
  outputs. Frozen ≥ TAR embeddings with the same predictions is consistent with E8/E15: in-sample fine-tuned
  embeddings carry a train/test gap the learner can over-trust.
- Pilot caveat: the pilot gains are small (+0.004 mean) and below the pilot bar; the full run (20 datasets × 5 folds
  × 5 learners, C1 primary baseline = leaderboard Joint TAR) is the real test, as pre-registered.
- Next step: freeze C1 for frozen_xfit_pred and plan the full run (P4).

## P4 — Full run + claim C1 FROZEN. Written 2026-10-10, BEFORE any full-run result exists. Do not edit.
### Claim C1 — FROZEN (2026-10-10; supersedes the C1 draft above, which stays for the record)
- **Method:** `frozen_xfit_pred` (P3 decision rule, E16): Joint Frozen E5 PCA embeddings (upstream `all`) + K = 3
  cross-fitted tar_repro head predictions per text column (`multabench/ours/stacking.py`, engine `stacking/1`).
- **Primary test (decides C1):** learner **CatBoost**; frozen_xfit_pred vs the **official leaderboard Joint TAR**
  (`ft` rows of `multabench/leaderboard/results/text/*.csv`); per dataset d = mean over folds 0–4 of (ours − `ft`);
  **two-sided Wilcoxon signed-rank test over the 20 datasets; pass = p < 0.05 AND mean d > 0.**
- **Secondary (all reported regardless of outcome):** LightGBM, TabM, TabPFN-v2, TabPFN-v2.5 (TabPFN: n = 18, no
  leaderboard baseline on Spotify Genres / Wine Review) vs leaderboard `ft`, same test, **Holm-corrected over these 4**.
  Plus CatBoost frozen_xfit_pred vs **our tar_repro** (CatBoost, same GPU type, same folds), same test.
- **Reported for every comparison:** n, mean / median difference, 95% bootstrap CI of the mean (10,000 resamples of
  datasets, seed 42), Wilcoxon W and p, matched-pairs rank-biserial r, Cohen's d_z, wins. A dataset enters a
  comparison only if all 5 folds exist on both sides (excluded datasets are listed). Implemented and tested before any
  full-run result: `scripts/full_report.py` (`tests/test_full_report.py`).
- **Forum:** no answer yet; if the course forum prescribes a different test, it is added here as an amendment
  BEFORE full-run results exist.
### Run plan
- Datasets: the 20 MulTaBench text datasets; folds 0–4; seed 42; upstream splits / caps; upstream learner defaults.
- Jobs, in this order (cheapest datasets first within each block, by `reports/compute_budget.csv`):
  `jobs/full_part1.txt` — tar_repro CatBoost (100), frozen_xfit_pred CatBoost (100), LightGBM (100), TabM (100);
  `jobs/full_part2.txt` — frozen_xfit_pred TabPFN-v2 (90), then TabPFN-v2.5 (90, reuses v2's fold fine-tunes).
  The split keeps every learner that shares fold fine-tunes in the same file (CatBoost / LightGBM / TabM use the same
  training rows; TabPFN has its own), so the fine-tune cache can live on the session's local disk.
- frozen_xfit_pred's base tar_repro fine-tune is only for JSON diagnostics: loaded if cached (CatBoost / LightGBM /
  TabM after tar_repro CatBoost), otherwise skipped (TabPFN). It never changes the method's features (test).
- **GPU:** A100 if available, else L4 — the **same type for every full-run job** (tar_repro and frozen_xfit_pred);
  recorded per JSON (`gpu`, `hardware.gpu_type`); `full_report.py` checks `same_gpu_type`. The pilot stays on T4.
- **Results root:** `MyDrive/text-tabular-nlp/results_full` (separate from the pilot's `results/`, so no T4 pilot
  result is reused). Fine-tune cache: local disk (`--cache_dir /content/tar_cache`) — a full run makes ~670 fine-tunes
  (~20–50 GB of cached embeddings), too much for Drive; a session that ends early only costs re-running its
  unfinished jobs (finished JSONs are on Drive and skipped).
- **Estimate** (T4 cost model calibrated on the pilot: predicted 1.1 h for tar_repro's 18 jobs, measured 67 min;
  xfit ≈ 1.8 × tar_repro measured; A100 ≈ 3 × and L4 ≈ 1.8 × a T4 assumed, to be checked on the first jobs; learner
  times for TabM / TabPFN guessed): part 1 ≈ 37 T4-h → **≈ 12 h A100 / ≈ 21 h L4**; part 2 ≈ 24 T4-h →
  **≈ 8 h A100 / ≈ 13 h L4**. Each part fits a 24-hour session at the estimate; if one does not finish, run the same
  file again (resumes).

## E17 — FULL RUN part 1: claim C1 does NOT pass   (2026-10-11, commit c2078f2, Colab A100-SXM4-40GB)
- Source: `results_export_20261011_200658.zip` (AUTO_DISCONNECT export of `jobs/full_part1.txt`): 400/400 jobs —
  tar_repro CatBoost, frozen_xfit_pred CatBoost / LightGBM / TabM, 100 each (20 datasets × folds 0–4). All at
  c2078f2, clean tree, 0 Tracebacks, one GPU type (A100-SXM4-40GB). GPU time: tar_repro 5.6 h, frozen_xfit_pred
  CatBoost 15.1 h, LightGBM 0.7 h, TabM 0.8 h (22.2 h total vs the ≈ 12 h estimate). Part 2 (TabPFN-v2 / v2.5) has
  NOT been run (the second Colab tab never started). Analysis: `scripts/full_report.py` → `reports/full/c1_report.json`
  (the pre-registered code, written and tested before any full result).
- **PRIMARY (pre-registered C1): CatBoost frozen_xfit_pred vs leaderboard Joint TAR, n = 20: mean −0.0025,
  95% bootstrap CI [−0.0089, +0.0017], Wilcoxon p = 0.41, rank-biserial r = −0.22, 8/20 datasets positive →
  NO PASS.**
- Secondary vs leaderboard `ft` (Holm over the 4): LightGBM mean −0.0068 (p 0.15, Holm 0.31); TabM +0.0021
  (p 0.29, Holm 0.31); TabPFN-v2 / v2.5 not run (n = 0).
- Secondary vs our tar_repro (CatBoost, A100, same folds): mean −0.0008, CI [−0.0068, +0.0033], p = 0.29.
- tar_repro (A100) vs leaderboard `ft`: mean −0.0017, 6/20 positive (cf. pilot engine bias −0.0022, E9).
- Per dataset (frozen_xfit_pred − leaderboard, CatBoost): largest loss **Video Games Sales −0.0555** (tar_repro
  −0.0049); largest gains Mercari +0.0118, Fake Job Postings +0.0079, Rotten Tomatoes +0.0040, Product Sentiment
  +0.0036; the other 15 within ±0.005.
- Hostile review / diagnosis (exploratory, does not change C1):
  - Video Games is not a bug: Phase 0 showed the largest TAR − Frozen gain there for CatBoost (+0.094) — the
    fine-tuned embeddings matter most on this dataset and frozen_xfit_pred drops them. In addition the prediction
    features fail there: their test R² is −0.36 … −3.1 per fold. Video Games sales are extremely heavy-tailed, so the
    top quantile bin's midpoint is huge and the pre-registered "expected value over bin midpoints" feature is
    dominated by it (bin means would be robust).
  - Without Video Games the mean vs leaderboard is +0.0003 (7/20 positive, median −0.0009) — i.e. frozen_xfit_pred is
    roughly on par with Joint TAR elsewhere, not better. vs tar_repro, 12/20 datasets favour frozen_xfit_pred and the
    mean without Video Games is ≈ +0.0018 — post hoc, not a claim.
  - Splits/metric: upstream code paths; per-dataset values use all 5 folds on both sides (no exclusions).
- **Conclusion: C1 fails. The method (cross-fitted head predictions + frozen embeddings) removes TAR's in-sample
  leakage and matches Joint TAR on most datasets, but it does not significantly beat it; on the dataset where
  fine-tuned embeddings matter most (Video Games) it loses clearly.**
- Next step: Dima decides whether to run part 2 (TabPFN, pre-registered secondary; ≈ 15–20 A100-h) and how to frame
  the paper (negative result + mechanism analysis).

- **Summary (2026-10-12):** C1 failed; pivot to P5 = Cross-fitted Text Experts, no fine-tuning (reasons: PCA bottleneck, TAR's advantage = one honest number per text column, all our variants used the same reader E5).

## E18 — EXPLORATION (not pre-registered): cheap label-aware text experts on top of Joint Frozen   (2026-10-11/12, laptop CPU)
- Harness: `scripts/explore_cheap.py` (upstream loading / splits / learner preprocessing; plain CatBoost with upstream
  hyperparameters and its own 90/10 val split). Its frozen base reproduces the leaderboard Joint Frozen within ≈ 0.003
  (e.g. Product Sentiment 0.9017 vs 0.8991). Experts here: word 1–2-gram TF-IDF → ridge / logistic (C = 4), and
  k = 20 cosine-kNN target mean in frozen E5 space, 5-fold OOF, per text column.
- Results (CatBoost, mean over folds; LB = leaderboard):
  | dataset (folds) | frozen base | + lex | + knn | + both | LB Joint Frozen | LB Joint TAR |
  |---|---|---|---|---|---|---|
  | Video Games (0–2) | 0.378 | 0.402 | 0.385 | 0.424 | 0.405 | 0.460 |
  | Product Sentiment (0–2) | 0.902 | 0.911 | 0.901 | 0.909 | 0.899 | 0.906 |
  | Book Price (0–2) | 0.546 | 0.575 | 0.579 | 0.585 | 0.557 | 0.576 |
  | Baby Products (0–2) | 0.880 | 0.899 | 0.889 | 0.899 | 0.883 | 0.907 |
  | Fake Job Postings (0) | 0.901 | 0.937 | 0.937 | 0.941 | 0.893 | 0.927 |
  | Data Scientist Salary (0) | 0.818 | 0.836 | 0.821 | 0.833 | 0.817 | 0.824 |
- Frozen + both experts beat LB Joint TAR on Book Price, Product Sentiment, Fake Job (fold 0) and Data Scientist Salary
  (fold 0); not on Video Games and Baby Products (where TAR's fine-tuning matters most). Gains over the frozen base
  +0.007 … +0.046.
- The "TAR embeddings + experts" test did not finish a fold on CPU (stopped; fine-tuning on CPU > 1 h per fold).
  Fake Job stopped after fold 0 (timeout), Data Scientist Salary after fold 0; Book Readability and Montgomery Salaries
  not run (stopped before any result, to freeze P5 without further peeking).
- Caveats: hand-picked small datasets, ≤ 3 folds, a simplified harness, no pre-registration — direction only. The
  explored datasets are 6 of the 20 P5 datasets (disclosed in P5).

## P5 — Cheap cross-fitted text experts stacked on TAR / Frozen. Written 2026-10-12, BEFORE any P5 run. Do not edit.
### Recipe (frozen; `multabench/ours/experts.py`, `stacking.run_experts`, engine `experts/1`)
- **Experts, per text column AND on all text columns concatenated** (`"__all__"`: "col: val | col: val …"):
  - **Lexical:** TF-IDF word 1–2-grams + char_wb 3–5-grams (sublinear_tf, min_df = 2, vocab capped at 100k each) →
    RidgeCV (regression, alphas 0.1…100) / LogisticRegression (classification) with C ∈ {0.25, 1, 4, 16} chosen by an
    inner 3-fold CV (mean log-likelihood) on the learner's training rows, then reused for every out-of-fold fit.
  - **Neighbour:** k = 20 cosine neighbours in frozen E5 space, similarity-weighted (max(1 − d, 0)) mean label (reg) /
    class distribution (cls). For `"__all__"` the row vector is the L2-normalised mean of its per-column frozen
    embeddings (no extra encoding).
- **Cross-fitting:** 5-fold out-of-fold on the learner's training rows (StratifiedKFold for cls when possible, KFold
  otherwise, seed 42); learner-val and test rows get the predictions of experts fit on all training rows.
- **Features:** cls → class probabilities for classes 1.. (binary: 1 column), reg → 1 column; appended after the
  owning text column's 30 PCA components (`"__all__"` features go with the first text column). Expert features use
  the target, so the bar is Joint TAR for both variants.
- **Methods:** `tar_experts` = tar_repro's full-pool fine-tune embeddings (one fine-tune per dataset-fold, cached,
  shared) + lexical + neighbour; `frozen_experts` = Joint Frozen embeddings + lexical + neighbour.
  Ablations (CatBoost): `tar_experts_lex` (lexical only), `tar_experts_nbr` (neighbour only),
  `tar_experts_lex_insample` (lexical fit and predicted on the same training rows — not cross-fitted — to show leakage).
### Claims (frozen)
- **C2 (primary):** CatBoost `tar_experts` vs the leaderboard Joint TAR (`ft`); per dataset the mean over folds 0–4
  of (ours − `ft`); **two-sided Wilcoxon over the 20 datasets; pass = p < 0.05 AND mean difference > 0.**
- **C3 (secondary):** CatBoost `frozen_experts` vs leaderboard Joint TAR (same test); also reported vs leaderboard
  Joint Frozen for context.
- **Secondary learners:** LightGBM, TabM for both methods vs `ft`, Holm-corrected over these 4. TabPFN not run.
- **Also reported:** C2 vs our tar_repro (CatBoost, A100, `results_full`); ablations vs `ft` and vs tar_experts;
  bootstrap CI, rank-biserial r, Cohen's d_z, wins; runtime (hours) per method — the cost story.
- Analysis code written and tested before any P5 result: `scripts/p5_report.py` (`tests/test_full_report.py`).
- Disclosure: E18 explored frozen + experts (a simpler version: word n-grams only, fixed C) on 6 of the 20 datasets
  (Video Games, Product Sentiment, Book Price, Baby Products, Fake Job Postings, Data Scientist Salary) before this
  freeze; "TAR + experts" was not observed on any dataset.
### Run plan
- 20 datasets × folds 0–4, seed 42, upstream splits / caps / learner defaults, **A100** for every job.
- `jobs/p5_A.txt` (200 jobs): tar_experts CatBoost, then frozen_experts CatBoost — cheapest datasets first.
  `jobs/p5_B.txt` (700 jobs): the 3 ablations (CatBoost), then LightGBM and TabM for both methods.
- Results root `MyDrive/text-tabular-nlp/results_p5`; cache on Drive (`results_p5/_cache`, default) so the 100 TAR
  fine-tunes (~5 GB) and the expert features are reused across blocks and sessions.
- Estimate (A100; tar_repro CatBoost took 5.6 h in E17; experts are CPU work, ≈ 1–5 min per dataset-fold; CatBoost
  ≈ 1–2 min per job): file A ≈ 10–14 h, file B ≈ 10–14 h. Each fits one 24 h session; re-run a file to resume.

## E19 — Post-freeze CPU check of P5 v1 (tar_experts / frozen_experts), stopped   (2026-10-12, laptop CPU, code aaaacec)
- Real P5 v1 code via `run_experiment.py` on CPU (`.explore/p5_cpu`; exploration only, not official results):
  Product Sentiment folds 0–2 — tar_experts 0.8932 / 0.9335 / 0.9169 (mean 0.9145), frozen_experts 0.8955 / 0.9135 /
  0.9259 (mean 0.9116); leaderboard Joint TAR mean 0.9055; our tar_repro (mean of seeds 42 + 43) 0.9146.
  Book Price fold 0 — frozen_experts 0.6032 (leaderboard TAR 0.5793; our tar_repro 0.561 / 0.581); tar_experts ran out
  of laptop RAM (7.2 GB, exit −9) and was stopped. All fine-tuning work stopped at Dima's request.
- Reading: on Product Sentiment the experts add ≈ 0 on top of our own TAR; frozen + experts nearly matches TAR without
  fine-tuning (25 s vs ≈ 20 min per job on CPU); on Book Price fold 0 frozen + experts is clearly above TAR.
- Cross-fitted TAR head predictions from the C1 full run were NOT saved (fold fine-tunes lived on the Colab session's
  local disk; JSONs keep scores/diagnostics only), so no "+ cross-fitted TAR predictions" variant is possible
  without new fine-tuning.
- P5 v1 (commit aaaacec: tar_experts / frozen_experts) was never run on Colab; it is superseded by P5 v2 below.

## P5 v2 — "Cross-fitted Text Experts" (CTE), no fine-tuning. Written 2026-10-12, BEFORE any P5 v2 run. Do not edit.
- Supersedes P5 v1 (aaaacec), of which no official job was ever run.
### Method (engine `cte/1`, `multabench/ours/cte.py`)
Features for the learner = structured columns (upstream preprocessing)
 + frozen e5-small-v2 embeddings, PCA 30 per text column, PCA fit on the learner's training rows only
 + per text column AND on all text columns concatenated, cross-fitted predictions of
   1. **Lexical expert:** TF-IDF word 1–2-grams + char_wb 3–5-grams (sublinear_tf, min_df = 2, ≤ 200k features each)
      → RidgeCV (alphas 0.1, 1, 10, 100) for regression; LogisticRegression with C ∈ {0.1, 1, 10} chosen by inner
      3-fold CV (mean log-likelihood) on the training rows, for classification.
   2. **Neighbour expert:** k = 20 cosine neighbours in frozen E5 space among the training rows of the other folds;
      distance-weighted (1 / (distance + 1e-6)) mean label (regression) / class distribution (classification). For the
      concatenated view, the row vector is the L2-normalised mean of the per-column frozen embeddings.
 + **target-free numeracy and form features per text column:** count of numbers, first / max / min number
   (sign-preserving log1p), count of 4-digit years 1900–2030 and max year, currency symbol present and max money
   amount, count of percentages, chars, words, sentences, mean word length, uppercase ratio, digit ratio, punctuation
   ratio, count of "!" and "?", unique-word ratio, is_empty.
- Regression experts are fit on a rank-Gaussian transform of y (QuantileTransformer, normal output) fit on the rows
  the expert is trained on only (each fold's training rows; all training rows for the full fit); the feature is the
  prediction in that space.
- Classification outputs: binary 1 column, multiclass C − 1 columns.
- Cross-fitting: 5-fold OOF on the learner's training rows (stratified for classification); learner-val and test rows
  use experts fit on all training rows. No test or val row is used to fit anything.
- Frozen embeddings are computed once per dataset (target-free) and cached; everything else runs on CPU.
### Claims (frozen)
- **C2 (primary):** CatBoost CTE vs leaderboard Joint TAR (`ft`); two-sided Wilcoxon over the 20 datasets on
  fold-averaged differences (folds 0–4); **pass = p < 0.05 AND positive mean.** Also reported on the 16 datasets not
  in Dima's exclusion list (all except Book Price, Product Sentiment, Baby Products, Video Games) and on the strict 14
  that also exclude Fake Job Postings and Data Scientist Salary (explored at fold 0 in E18).
- **Secondary:** CTE vs leaderboard Joint Frozen (`all`); LightGBM CTE vs `ft` (Holm over the secondary learner tests);
  ablations (CatBoost) lexical-only, neighbour-only, numeracy-only, lexical in-sample (NOT cross-fitted). No
  "+ cross-fitted TAR predictions" variant (they do not exist, E19).
- **Diagnostics per job:** each expert's quality on training rows (OOF) vs test; CatBoost importance share of each
  feature group (structured, PCA, lexical, neighbour, numeracy); runtime.
- Bootstrap CIs, rank-biserial r, Cohen's d_z, wins for every comparison (same code as C1).
### Run plan
- `jobs/cte_full.txt`: CTE CatBoost (100), CTE LightGBM (100), then the 4 ablations (CatBoost, 400); cheapest datasets
  first. Results root `MyDrive/text-tabular-nlp/results_cte`; cache (frozen embeddings + expert features) on Drive.

## E20 — CTE (P5 v2) real-data CPU smoke + parallel lexical views   (2026-10-12, laptop CPU i7-13700H 8 cores)
- **Question:** does the frozen CTE code run end-to-end on real data, and how long will `jobs/cte_full.txt` take?
- **Commands:** `scripts/run_experiment.py --method cte --learner catboost --dataset <D> --fold 0 --seed 42
  --out .explore/cte_cpu --cache_dir .explore/cte_cpu/_cache --train_examples 10000` (D = Product Sentiment, Book Price).
  These are fold-0 smoke checks of datasets already in Dima's exclusion list, NOT part of claim C2 (they are re-run on Colab).
- **Results (fold 0, CatBoost):** Product Sentiment 0.8857 (LB `ft` 0.8740, +0.012), job 5.2 min (emb 29 s, experts 253 s);
  Book Price R² 0.6074 (LB `ft` 0.5793, +0.028), job 17.8 min (emb 147 s, experts 885 s). Experts dominate the runtime.
- **Change (execution only, method unchanged):** the lexical expert now fits its views (each text column + `__all__`)
  in parallel processes (`compute_expert_features(..., n_jobs)`; CTE uses `OURS_EXPERT_JOBS`, default = CPU count).
  Book Price re-run with 6 workers and a fresh expert cache: experts 885 s → 459 s, score identical (0.607430588359001),
  max |Δ feature| 2.5e-12 (BLAS thread count inside workers); peak RSS 1.3 GB. Test: serial ≈ parallel (atol 1e-10),
  parallel bit-reproducible with itself. Engine stays `cte/1` (results are the same up to float noise).
- **Runtime estimate (crude):** about 0.03 s per (training row × text view) serially, so ≈ 38 h serial for the 100 CTE
  dataset-folds. With parallel views it's ≈ 15–25 h on 8 laptop-class cores. Colab cores are slower per core, so ≈ 20–40 h for
  the whole jobs file. The LightGBM block and the cte_lex / cte_nbr ablations hit the expert cache (minutes each); cte_num
  needs no expert; cte_lex_insample refits one lexical model per view (≈ +3–6 h). Unmeasured risks: long texts
  (Jigsaw, Wine), many-class LogisticRegression (Spotify genres).
- **Hostile review:** the smoke rows are fold 0 of the standard split (n_train 4582 / 4491 = upstream caps), and the
  PCA / rank-Gauss / experts fit on train rows only (tests in `tests/test_cte.py`). These are two datasets that were already
  explored, so they are not evidence for C2.
- **Next:** commit, run `jobs/cte_full.txt` on Colab (CPU runtime allowed for CTE jobs), then `scripts/cte_report.py`.

## P5 v2 — Amendment 1 (2026-10-12): speed changes to CTE, engine `cte/2`. Written BEFORE any CTE full-run result.
- **Why:** E20 estimated 20–40 h for `jobs/cte_full.txt`, too slow for a TF-IDF + linear-model method. Target: ≤ 3 h
  for the whole CTE CatBoost block. No full-run job had been started; no CTE result other than the E20 fold-0 smoke
  (Book Price, Product Sentiment, both on the exclusion list) has been seen.
- **Method changes (replace the P5 v2 lexical/neighbour details; everything else unchanged):**
  1. TF-IDF vectorizers are fit ONCE per job on the learner's training rows (target-free) and that matrix is reused for
     every OOF fold and for val/test. word 1–2 max 50k + char_wb 3–5 max 50k (sublinear, min_df 2); the char
     vectorizer sees each text truncated to its first 2,000 characters.
  2. The lexical model is ONE fixed linear least-squares fit: `Ridge(alpha=1.0, solver="sparse_cg")`. Regression: on
     the rank-Gaussian y (QuantileTransformer fit on that fit's training rows, as before). Classification: the same
     Ridge on one-hot class indicators (one multi-output fit); output = class scores, binary 1 column (class 1),
     multiclass C−1 columns (scores sum to 1, so C−1 loses nothing). No inner CV, no RidgeCV, no LogisticRegression.
  3. The lexical expert runs on the concatenation of all text columns ("col: val | …") only, not per column. The
     neighbour expert stays per column + concatenation; k = 20, inverse-distance weights, computed as one matrix
     product of L2-normalised frozen embeddings + top-k (GPU if available, else CPU).
  4. Execution only: independent jobs run in parallel processes (n = CPU cores // 2), each learner with
     thread_count = cores // n, so cores are not oversubscribed.
- **Unchanged:** structured columns, frozen E5 PCA-30 per text column (fit on train rows), numeracy features, 5-fold
  (stratified) OOF with val/test from the fit on all training rows, the variants (cte, cte_lex, cte_nbr, cte_num,
  cte_lex_insample, with the same meaning under the new lexical expert), claims C2 / C2_16 / C2_14, the secondaries,
  the statistical tests, the jobs file and `scripts/cte_report.py`. Claim C2 now refers to engine `cte/2`.

## E21 — Profiling CTE before/after amendment 1   (2026-10-12/13, laptop CPU i7-13700H, uncommitted code on top of ae145f4)
- **Question:** where does a CTE job spend its time, and what does amendment 1 change?
- **Tool:** `scripts/profile_cte.py` (wraps each stage with a timer, runs one job in-process); reports in
  `.explore/prof_old/`, `.explore/prof_new/` (gitignored).
- **Before (cte/1, ae145f4 snapshot, Book Price fold 0, serial, all cores):** RidgeCV 276.7 s (42 calls), TF-IDF
  fit_transform 55.4 s (84), TF-IDF transform 9.7 s, frozen embeddings 65.7 s, CatBoost 11.4 s, sklearn kNN 1.2 s,
  numeracy 1.5 s; job 427 s; score 0.6074 (same as E20). E20's 17.8 min was inflated by another process running
  at the same time.
- **After amendment 1 (cte/2, fold 0, 2 threads per job = one Colab slot):**

  | dataset | emb (CPU) | experts | CatBoost | job | score | LB `ft` |
  |---|---|---|---|---|---|---|
  | Product Sentiment | 17 s | 2 s | small | 28 s | 0.8755 | 0.8740 |
  | Book Price | 100 s | 8 s | 30 s | 143 s | 0.6008 | 0.5793 |
  | Jigsaw | 316 s | 8 s | 6 s | 334 s | 0.8652 | 0.9137 |
  | Wine | 202 s | 46 s | 738 s | 1002 s | 0.9823 | 0.9800 |
  | Spotify Genres | 97 s | 46 s | > 45 min (stopped, no score) | — | not seen | — |

- **Conclusions:** the experts are ~40× faster and no longer the bottleneck. Multiclass CatBoost with C−1 columns per
  expert view (Wine: 216 text features) dominates. Jigsaw dropped well below TAR with the fixed alpha = 1.
- **Next:** amendment 2 (generic rules A and B below), then re-time Wine and Jigsaw.

## P5 v2 — Amendment 2 (2026-10-13): column budget + OOF alpha selection, engine `cte/3`. BEFORE any full-run result.
- **Disclosure:** during the profiling (E20, E21) the fold-0 TEST scores of Product Sentiment, Book Price, Jigsaw
  and Wine were seen. Spotify Genres' run was stopped before scoring (its fold-0 score was NOT seen; its runtime was).
  The rules A and B below are generic rules, the same for every dataset, decided before the full run, prompted by a
  runtime problem (A) and the Jigsaw drop (B). B selects only on training-row OOF predictions.
- **A. Multiclass column budget.** For classification with C > 10, every expert view outputs the scores of the 10
  most frequent classes among the learner's training rows (fixed order, ties by class index) + max probability +
  entropy (12 columns) instead of C−1. Max probability / entropy are taken over all C classes (the neighbour expert
  already gives a distribution; the Ridge scores are clipped at 0 and renormalised; uniform if all are 0). For every
  multiclass dataset (C > 2) the neighbour expert runs on the concatenated text only (no per-column views). Binary
  (1 column) and regression (1 column) are unchanged; 3 ≤ C ≤ 10 keeps C−1 columns.
- **B. Lexical alpha by OOF.** For alpha in {0.3, 1, 3, 10}: the 5-fold OOF predictions on the learner's training
  rows (the TF-IDF fitted once, reused). The alpha with the best OOF metric is chosen: AUROC for binary, macro
  one-vs-rest AUROC of the class scores for multiclass, Spearman with y for regression; ties go to the smaller alpha.
  The chosen alpha's OOF predictions are the feature; val/test come from the fit on all training rows with that
  alpha. The in-sample ablation uses the same chosen alpha. Val/test rows are never used. The chosen alpha and the
  OOF metric per alpha are logged in every result JSON.
- **Reporting change (secondary only; C2 over 20 datasets unchanged):** the held-out report excludes every dataset
  used in exploration/profiling: `C2_heldout13` = 20 minus Book Price, Product Sentiment, Baby Products, Video
  Games, Jigsaw, Wine, Spotify Genres (13 datasets); `C2_strict11` also excludes Fake Job Postings and Data Scientist
  Salary (E18). The originally pre-registered C2_16 / C2_14 are still printed for transparency.

## E22 — Re-timing after amendment 2 (engine cte/3)   (2026-10-13, laptop CPU, 2 threads per job, frozen embeddings cached)
- **Command:** `OURS_LEARNER_THREADS=2 OMP_NUM_THREADS=2 python scripts/profile_cte.py -- --method cte --learner
  catboost --dataset <D> --fold 0 --out .explore/prof_v3 --cache_dir .explore/prof_v3/_cache` (D = Wine, Jigsaw).
- **Wine (C = 30):** text features 216 → 124. Experts 159 s (multi-output Ridge with 30 targets × 4 alphas × 5 folds;
  cte/2: 46 s). CatBoost 929 s (cte/2: 738 s, even with fewer columns: the multiclass CatBoost cost is driven by C and
  upstream's 2000 iterations, not by our columns). Job 18 min. Score 0.9779 (cte/2 0.9823, LB `ft` 0.9800). Alpha 10
  (OOF macro AUROC 0.937 / 0.947 / 0.955 / 0.962). Expert quality train-OOF vs test: lex 0.952 / 0.946, nbr 0.931 / 0.929.
- **Jigsaw (binary):** job 30 s. Score 0.8695 (cte/2 0.8652, LB `ft` 0.9137). Alpha 3 (OOF AUROC 0.852 / 0.857 /
  0.859 / 0.853). Lexical OOF 0.859 vs test 0.855. TF-IDF + Ridge reaches only ≈ 0.86 on toxicity, where fine-tuning
  helps most (leaderboard TAR − Frozen = +0.117). Amendment 2B does not close this gap, and nothing was tuned on test.
- **Hostile review:** OOF selection over 4 alphas makes the chosen alpha's OOF features slightly optimistic for
  CatBoost (selection bias over 4 values; the train-vs-test gaps above are 0.004–0.006). Both datasets are now out of
  the held-out report. Wall times on a laptop vary run to run (thermal), so they are ±20%.
- **Revised estimate (A100 runtime, 12 vCPU, 6 jobs × 2 threads):** binary/regression jobs ≈ 0.5–2 min. Small
  multiclass ≈ a few min. Wine ≈ 18 min/job. Spotify Genres unknown (CatBoost > 45 min/job at cte/2; C is large).
  CatBoost block ≈ 10 slot-hours / 6 ≈ 2–3 h wall (Colab cores ~1.3× slower), with the Spotify tail as the risk.
  Whole jobs file (LightGBM + 4 CatBoost ablations, which reuse the expert cache) ≈ 8–12 h.

## P6 — Secondary / EXPLORATORY CatBoost variants. Written 2026-10-13, BEFORE any P6 code or run. Do not edit.
- **Status:** exploratory and secondary. P6 results never replace or modify claim C2 (P5 v2, engine `cte/3`), and
  are reported separately, labelled exploratory. Engine `p6/1`. Learner CatBoost only; 20 datasets × folds 0–4;
  same splits, seeds (42), learner val split, leaderboard comparison and leakage rules as P5. The frozen-embedding
  and expert caches of the CTE run are reused (same cache root), so P6 runs after `jobs/cte_full.txt`.
- **Variants** (all four: features computed from the learner's training rows only; OOF where a label is used):
  1. `catboost_native_text`: upstream's structured features + the raw text columns passed as CatBoost
     `text_features` (CatBoost's default tokenizer and dictionaries: Space, Word + BiGram). Calcers: BoW + NaiveBayes
     + BM25 for classification; BoW only for regression (CatBoost supports NaiveBayes/BM25 for classification only).
     No embeddings, no experts. The baseline a reviewer will ask for.
  2. `cte_plus_native`: the full CTE features (cte/3, same caches) + the same CatBoost text_features.
  3. `cte_nb`: CTE where, for classification, the lexical expert uses NB log-count-ratio weighted TF-IDF (NBSVM,
     Wang & Manning 2012): for each class c (binary: class 1 only), r_c = log((p/‖p‖₁)/(q/‖q‖₁)), with p = 1 + the sum
     of the TF-IDF rows of class c and q = 1 + the sum over the other rows. Then a one-vs-rest Ridge on X·diag(r_c) with
     target 1[y = c]. r_c is computed inside every fit from that fit's training rows only. The same alpha grid and OOF
     selection, column layout and neighbour expert as cte/3. Regression is unchanged (= cte).
  4. `cte_residual`: the lexical and neighbour experts are trained on the residuals of a structured-only CatBoost S
     (structured columns only; fixed: 300 iterations, depth 6, learning rate 0.1, seed 42, categorical columns as
     cat_features; a constant model if there are no structured columns). Regression: residual = rank-Gaussian y − S's
     prediction (S fitted on the rank-Gaussian y). Classification: residual = one-hot(y) − softmax(S's logits), i.e.
     the negative log-loss gradient at the offset logits (the least-squares form of an offset). **Still cross-fitted,
     nested:** for expert fold k, S's predictions for the expert's training rows T_k are 3-fold OOF within T_k only,
     so no label of the held-out rows H_k reaches the expert that predicts H_k. For val/test: residuals of all
     training rows from S's 5-fold OOF (the expert folds), then the expert is fitted on all training rows. Output =
     predicted residual: regression 1 column; binary class 1; 3–10 classes C−1; C > 10 the top-10 class residuals
     + the max and min residual over all classes (12 columns). Alpha is chosen by the OOF metric of
     S_OOF + predicted residual (AUROC / macro AUROC / Spearman). Numeracy features as in cte.
- **Analysis (exploratory, `scripts/p6_report.py`):** each variant vs leaderboard Joint TAR (`ft`, CatBoost) and
  vs our `cte` CatBoost (paired over datasets on fold-averaged scores). Two-sided Wilcoxon, Holm over the 8 tests,
  bootstrap CI, wins. Printed on all 20 and on the held-out 13 (`C2_heldout13` list). No success threshold: this is
  for the paper's discussion/ablation section only.
- **Tests:** the same leakage tests as P5 (each held-out row's OOF prediction is bit-identical when that fold's labels
  change, including through S for cte_residual; NB ratios from training rows only), determinism, an end-to-end run
  of every variant, and a golden check that cte/3 (P5) features are unchanged (`tests/golden/cte3_features.npz`,
  generated at f1514bc).

## E23 — P6 runtime check (fold 0, laptop, 2 threads per job, 3 jobs at once)   (2026-10-14, uncommitted P6 code on f1514bc)
- **Command:** `OURS_LEARNER_THREADS=2 python scripts/profile_cte.py -- --method <M> --learner catboost --dataset <D>
  --fold 0 --out .explore/p6 --cache_dir .explore/p6/_cache` for the 4 P6 variants on Book Price, Product Sentiment
  and Wine (all three already explored; excluded from the held-out report). Frozen embeddings cached from E21.
- **Job times:** Book Price: native_text 8.1 min, plus_native 7.5 min, nb 37 s, residual 53 s. Product Sentiment:
  6.2 min, 6.0 min, 19 s, 22 s. Wine (C = 30): `catboost_native_text` was still running after 1 h 48 min and was
  stopped by Dima (no score). CatBoost's native text processing is ≈ 98% of a native job's time. NaiveBayes and BM25
  are computed per class, so the cost grows with C. cte_nb / cte_residual add ≈ 10–15 s of expert time per job
  (including the nested structured-CatBoost fits).
- **Scores were not inspected for any decision.** The method and the calcers stay as pre-registered in P6.
- **Execution change only:** `jobs/p6_secondary.txt` now runs cte_nb and cte_residual first, then the two native
  blocks.
- **Estimate (A100, 6 slots):** cte_nb + cte_residual ≈ 1–2 h together (CatBoost multiclass dominates, as in E22).
  The native blocks are ≈ 6–8 min per job on small datasets, and multiple hours per job on Wine / Spotify. Total for the
  native blocks is likely ≥ 1 day of runtime, unbounded until Wine / Spotify are measured.
- **Open decision (Dima):** keep the pre-registered calcers and run the native blocks last, or amend P6 (BoW only for
  C > 10) before any P6 result.

## E24 — P5 v2 full run (CTE, engine cte/3): claim C2 does NOT pass; held-out 13 passes   (2026-10-14, Colab A100)
- **Run:** `jobs/cte_full.txt` (600 jobs) at f1514bc (clean), A100-SXM4-40GB, 6 jobs in parallel × 2 threads,
  RESULTS = MyDrive/text-tabular-nlp/results_cte. Exports `results_export_20261014_061322.zip` (599 jobs) +
  `results_export_20261014_095322.zip` (all 600). One job (cte catboost Product Sentiment fold 3) crashed in the first
  pass: two parallel jobs of the same dataset rebuilt the same lower-layer cache dir and one `rmtree`'d it under the
  other (FileNotFoundError, `tar_engine.LowerCache.get_or_build`). It was re-run alone at the same commit/GPU/threads
  (score 0.9086). Analysis: `python scripts/cte_report.py --out <export>/results` (identical to Colab's report);
  `reports/cte/cte_report.json`.
- **C2 (primary, CatBoost cte vs leaderboard `ft`, 20 datasets, fold-averaged, two-sided Wilcoxon):**
  mean +0.0106, median +0.0044, bootstrap 95% CI [+0.0028, +0.0193], **p = 0.058**, wins 11/20, rank-biserial r 0.49,
  d_z 0.54 → **does not pass** (pre-registered threshold p < 0.05).
- **Other pre-registered lines:** C2_heldout13 (amendment 2) mean +0.0173, CI [+0.0073, +0.0282], **p = 0.0105**,
  wins 9/13 → passes. C2_strict11 +0.0139, p = 0.042 → passes. C2_16 +0.0125, p = 0.065 → no. C2_14 +0.0092, p = 0.22 → no.
  vs Joint Frozen +0.0310, p = 1.9e-6, 20/20 wins. LightGBM cte vs `ft` (secondary, Holm) +0.0152, p = 0.021 → passes.
- **Per dataset (cte − ft):** Rotten Tomatoes +0.056, Mercari +0.045, Fake Job +0.042, DS Salary +0.030, Book
  Readability +0.027, Zomato +0.014, Book Price +0.013, Baby +0.012, Kickstarter +0.011, Michelin +0.006, Montgomery
  +0.003; Spotify −0.001, Wine −0.001, Vancouver −0.002, US Accidents −0.002, SCImago −0.002, Women Clothing −0.003,
  Product Sentiment −0.004, Video Games −0.008, **Jigsaw −0.022** (0/5 folds). Wins are large; losses are mostly
  tiny, except Jigsaw (toxicity, where fine-tuning helps most).
- **Ablations (CatBoost, mean vs cte):** lexical-only −0.011 (≈ `ft` level), neighbour-only −0.017, numeracy-only
  −0.021, lexical in-sample −0.045 (−0.034 vs `ft`). Cross-fitting matters: in-sample lexical predictions hurt a lot.
  CatBoost importance share (cte): PCA 0.42, lexical 0.21, structured 0.15, neighbour 0.13, numeracy 0.09.
- **Runtime:** CTE CatBoost block 3.4 h wall (19.7 slot-hours; the longest job 2.5 h, Spotify-sized multiclass CatBoost).
  Whole file 13.2 h wall. Target was ≤ 3 h for the CatBoost block: slightly over.
- **Hostile review:**
  - *Leakage:* experts, TF-IDF, rank-Gauss and PCA are fit on the learner's training rows only. Tests
    (`tests/test_cte.py`) show held-out rows' OOF predictions are bit-identical when their labels change. Frozen
    embeddings are target-free. The in-sample ablation behaves as leakage would (−0.045), so the OOF version is not
    leaking in the same way.
  - *Split / size:* every dataset has 5 folds; n_train / n_test equal upstream's caps (10,000 / 10% ≤ 2,000), from
    upstream's own `split_to_test`. The same splits were used for tar_repro, which reproduced the leaderboard in the pilot.
  - *Metric:* upstream's `score_all_metrics` (AUROC / macro-OvR AUROC / R²).
  - *Seeds:* seed 42, as the leaderboard. Single seed, so seed noise is included in the per-dataset differences.
  - *Choices made after seeing test data:* amendments 1–2 came after fold-0 test scores of 4 datasets (disclosed).
    Those 4 + the profiled ones are excluded from heldout13, which is where the result is strongest.
  - *Comparator environment:* the leaderboard numbers come from the authors' environment, so part of +0.01 could be
    environment. Against our own run of Joint Frozen the gap is large (+0.031, 20/20), but that is a weaker baseline.
  - *Crash:* the race could only crash a job; it could not silently corrupt one. A reader opens a cache only once
    `meta.json` exists, builds are written to a private temp dir and moved into place atomically, and the content is
    deterministic. The affected job was re-run.
- **Conclusion:** the primary claim C2 fails narrowly (p = 0.058 over 20; the CI excludes 0). The pre-registered
  held-out subsets and the LightGBM secondary pass. The paper must report C2 as not significant at 0.05, and report
  the held-out and secondary results as such, not as the primary claim.
- **Next:** fix the parallel cache race in our code (one job per dataset builds the frozen embeddings, the others wait),
  then P6 (exploratory). Dima to decide the P6 native-text calcer question (E23).

## C2 reporting decision (Dima, 2026-10-14)
- C2 is reported exactly as pre-registered: it does NOT pass on all 20 datasets (p = 0.058). The pre-registered
  held-out 13 (amendment 2) and the LightGBM secondary pass. No new tests or subsets are added for P5.

## P6 — Amendment 1 (2026-10-14, Dima): native-text calcers. Written BEFORE any P6 result.
- **State of P6 at this point:** no P6 job has run on Colab. Only E23's laptop timing runs exist (fold 0 of Book
  Price and Product Sentiment, all 4 variants; Wine `catboost_native_text` stopped). Their scores were not looked
  at; the logs stay in `.explore/p6/` (gitignored).
- **Change:** `catboost_native_text` and `cte_plus_native` use CatBoost feature_calcers = BoW only for
  classification datasets with C > 10 classes (E23: NaiveBayes/BM25 are per class, multiple hours per job on
  Wine). Binary and 3..10 classes: BoW + NaiveBayes + BM25. Regression: BoW (unchanged). The same rule for every
  dataset. Everything else in P6 is as pre-registered. Engine `p6/2` (p6/1 = before this amendment; never run on
  Colab).

## C3 — P6 claim (pre-registered 2026-10-14, BEFORE any P6 result). C2 stays as reported; C3 never replaces it.
- **Claim C3:** each of the 3 CTE variants (`cte_plus_native`, `cte_nb`, `cte_residual`), CatBoost, vs leaderboard
  Joint TAR (`ft`, CatBoost). Two-sided Wilcoxon over the 20 datasets on fold-averaged differences (folds 0–4),
  Holm-corrected over these 3 tests. **Pass (per variant) = Holm-adjusted p < 0.05 AND positive mean difference.**
- **Also reported:** the held-out 13 (the C2_heldout13 list) for each of the 3 variants (Holm over the same 3, labelled
  secondary). `catboost_native_text` vs `ft` is reported as a baseline, not a claim (no correction, no pass/fail). The
  exploratory P6 comparisons (each variant vs our `cte`) stay as in the P6 pre-registration.
- Bootstrap CIs, wins, rank-biserial r and d_z as for C1/C2. Script: `scripts/p6_report.py`.

### Note on the name C3 (2026-10-14)
- P5 v1 (superseded, never run) also defined a "C3" (frozen_experts). That claim is void. "C3" from here on means
  the P6 claim pre-registered above; the paper should call it "C3 (P6)" or rename both consistently.

## P6 — Amendment 2 (2026-10-15, Dima): native-text variants dropped for runtime; C3 over cte_nb + cte_residual
- **Why (runtime only):** on the Colab P6 run (A100 VM, 6 parallel jobs × 2 threads), CatBoost native
  `text_features` on **Jigsaw ran 5 h 20 min with 0/5 folds done**. **Women Clothing took ≈ 58 min per fold** (BoW +
  NaiveBayes + BM25 on long texts). Decided before any `cte_plus_native` result existed.
- **Change:** `catboost_native_text` and `cte_plus_native` are **dropped from the experiments**. No native-text job is
  re-run. The code keeps two harmless changes, which affect no reported number:
  - `native_feature_calcers` = BoW only for every task (engine p6/3 for the two native methods);
  - a 30-min per-job timeout in the parallel runner (`--job_timeout_min`, logged in
    `<out>/_logs/process_errors.jsonl`, not retried, reported as missing).
- **C3 (amended):** `cte_nb` and `cte_residual` (CatBoost) each vs leaderboard Joint TAR, two-sided Wilcoxon over the
  20 datasets on fold-averaged differences, **Holm over these 2 tests**. Pass = Holm-adjusted p < 0.05 and positive
  mean. Both are complete (100/100 jobs each). The held-out 13 is reported for each (secondary). C2 stays as reported;
  C3 never replaces it.
- **Disclosure:** reducing the Holm family from 3 to 2 tests makes the correction less conservative. The reason is
  the dropped variants' runtime, not results. The `cte_nb` / `cte_residual` results already existed on Drive when
  this was decided. Claude had not seen them; the Colab progress output prints each job's DONE line.
- **The 29 finished `catboost_native_text` jobs** are reported only as a partial, descriptive side table (per-dataset
  scores vs Joint TAR, no test, no claim). They mix calcer configurations: BoW only for regression and C > 10;
  BoW + NaiveBayes + BM25 for the classification ones that finished.
- **Engine bookkeeping:** `cte_nb` / `cte_residual` stay p6/2 (unchanged code), so finished results stay valid. The
  `cte_residual` cache key is unchanged (9280a862b6). A runner bug was fixed: results without a `tar` block
  (catboost_native_text) were treated as stale on every restart because the engine was read only from
  `tar.config`; it is now read from the top-level `engine_version`.

## Working rule (Dima, 2026-10-15)
- Every new job must come with a **measured** runtime estimate. Nothing over **1 hour total** runs without Dima's
  approval.

## E25 — P6 results: claim C3 does NOT pass (cte_nb, cte_residual); held-out 13 passes for both   (2026-10-15)
- **Data:** export `results_export (3).zip` (2026-10-15) of MyDrive/text-tabular-nlp/results_cte, made with no new
  runs. `cte_nb` 100/100 and `cte_residual` 100/100, all at d99be74 (clean), A100-SXM4-40GB, 2 threads per job.
  `catboost_native_text` 29/100 (dropped, amendment 2), `cte_plus_native` 0/100. No `process_errors` and no stray
  renamed results. Analysis: `python scripts/p6_report.py --out <export>/results` → `reports/p6/p6_report.json`.
- **C3 (CatBoost vs leaderboard Joint TAR, 20 datasets, two-sided Wilcoxon, Holm over 2):**
  - `cte_nb`: mean **+0.0112**, CI [+0.0038, +0.0196], p = 0.048, **Holm p = 0.097**, wins 11/20 → **does not pass**.
  - `cte_residual`: mean **+0.0102**, CI [+0.0023, +0.0189], p = 0.064, **Holm p = 0.097**, wins 12/20 → **does not pass**.
- **Held-out 13 (secondary, Holm over 2):** `cte_nb` +0.0173, Holm p = 0.021 → passes; `cte_residual` +0.0153,
  Holm p = 0.048 → passes.
- **Exploratory (vs our CTE, paired):**
  - `cte_nb` +0.0005, p = 0.53. It differs only on classification: Jigsaw +0.009, Product Sentiment +0.002, US
    Accidents +0.001, the rest ±0.001.
  - `cte_residual` −0.0004, p = 0.024 (wins 5/20): many small losses (Montgomery −0.007, Zomato −0.007, Vancouver
    −0.006, Product Sentiment −0.005), one large gain (Video Games +0.032, where TAR − Frozen is +0.094).
  - Neither variant improves on CTE overall.
- **Native-text side table (partial, descriptive only; `reports/tables/table_native_text_partial.csv` via the
  notebook):** 29 jobs on 6 datasets. Mean vs Joint TAR: Product Sentiment −0.025, Women Clothing −0.032 (4 folds),
  Wine −0.009, Baby +0.005, Vancouver −0.053, Video Games −0.410 (R²). Mean job time: Wine 237 min (BoW only), Women
  Clothing 55 min, the rest 7–15 min.
- **Hostile review:**
  - *Provenance / splits:* one commit, clean, one GPU type; n_train / n_test per dataset identical across cte,
    cte_nb and cte_residual.
  - *Bookkeeping:* for regression, `cte_nb` is defined as `cte`, and the scores are identical on 47/50 folds. The 3
    exceptions (Zomato folds 0, 2, 3: −0.0007, +0.0015, +0.0013) have identical expert features (cache hits, the same
    lexical test quality to 6 digits) but **different frozen embeddings**. In the CTE run those folds ran in parallel
    on the pre-lock code and each encoded the texts itself; the shared store kept the last writer's copy, which
    `cte_nb` then read. The same string encoded in a different GPU fp16 batch gives slightly different vectors.
    → Reproducibility caveat: CTE scores are reproducible to about ±0.0015 per fold across encodings. This is not
    leakage and has no direction. The lock (d99be74) now makes every job of a dataset use one stored encoding.
  - *Ties:* the `cte_nb` vs `cte` comparison has 10 regression datasets with (near) zero difference; Wilcoxon drops
    zeros, so that p-value rests on the 10 classification datasets.
  - *Leakage:* NB ratios inside every fit (tests); `cte_residual` nested cross-fitting (bit-identity tests on
    fold-0 label changes, including through S). The structured model's OOF quality is plausible per dataset
    (Spearman / AUROC 0.32–0.98; for example Montgomery 0.98, where structure dominates).
  - *Selection:* the C3 family was reduced from 3 to 2 for runtime (amendment 2, disclosed). With the original 3
    tests, Holm p would be ≥ 0.097 for both, so the conclusion does not change.
- **Runtime:** `cte_nb` 20.4 slot-hours, `cte_residual` 26.5 slot-hours (A100 VM, 6 parallel slots).
- **Conclusion:** neither variant beats Joint TAR on all 20 at the pre-registered level. Both reproduce the held-out
  13 result. They do not improve on CTE: NB weighting helps a little only on Jigsaw, and residual targets help only
  where structure is weak relative to text (Video Games). C2 and C3 are reported as pre-registered.

## E26 — Packaging: train.py / predict.py with a saved model folder; CPU smoke run   (2026-10-15, laptop CPU)
- **What:** `scripts/train.py` (MulTaBench dataset + fold, or `--csv` + `--target`) → model folder
  (`multabench/ours/cte_model.py`: TF-IDF vectorizers, Ridge with the chosen alpha, kNN index, PCA, CatBoost, upstream
  preprocessing state, meta.json) + metrics.json + test_rows.csv / test_predictions.csv. `scripts/predict.py`: model
  folder + CSV → predictions, nothing re-fitted. The full-pool experts are refitted once after training and are exactly
  the fit run_cte uses for test rows. The test reproduces the fit-time test-row features to 1e-9.
- **Command:** `python scripts/train.py --dataset MUL_TEXT_PRODUCT_SENTIMENT --fold 0 --out_dir .explore/smoke_ps`, then
  `python scripts/predict.py --model_dir .explore/smoke_ps --csv .explore/smoke_ps/test_rows.csv --out ...`.
- **Result:** train 36 s wall (fresh cache, CPU embeddings included), max RSS 1.25 GB; score 0.8859 (A100 run of the
  same fold: 0.8873; LB Joint TAR 0.8740). Predict 15 s wall for 509 rows. Round-trip predictions equal the fit-time
  predictions exactly (max |Δ| 0.0). Folder ≈ 8 MB.
- **Hostile review:** the CPU vs A100 difference (−0.0014) is the fp16 re-encoding effect from E25. The packaged model
  is one fold's model; it is not used for any claim. `uv lock --check` passes and no dependency changed.
- **Tests:** `tests/test_train_predict.py` (folder contents, fresh-process round trip, exact feature reproduction, CSV
  mode).

## P7 — Development round to improve CTE (exploration protocol; written 2026-10-16, BEFORE any P7 code or run)
- **Goal (Dima):** improve the results with cheap methods (verbalization, extra experts, no long fine-tuning), tested
  on a few datasets first. C2 and C3 stay as reported; anything from P7 is a new claim that must be pre-registered
  before its full run.
- **Post-hoc observation that motivates one direction (not a claim; computed 2026-10-16 from E24 results):** the
  label-free ablation `cte_num` (frozen PCA + numeracy) vs leaderboard Joint Frozen: +0.0096, 19/20 wins, Wilcoxon
  p < 1e-4. Two caveats: it was not pre-registered, and part of the gain may be our environment, not the method
  (E1: our frozen Book Price fold 0 was 0.5602 vs the leaderboard's 0.5551). A label-free claim therefore needs our
  own Joint Frozen run as a control.
- **Dev protocol (test rows never used):** per dataset, fold 0's training rows only (upstream split) are split 80/20
  (stratified for classification) into dev-train / dev-test, seeds 0 and 1. The learner (CatBoost, upstream defaults)
  is fit on dev-train and scored on dev-test with the benchmark metric. Label-free variants are compared with our own
  frozen baseline (`frozen0`); label-aware variants with `cte`. Expert cache keys use fold = 100 + seed, so nothing
  mixes with real folds.
- **Dev datasets:** Jigsaw, Kickstarter (binary), Product Sentiment (multiclass), Book Price, Baby Products, Video Games
  (regression). These cover CTE's biggest loss (Jigsaw), a regression loss (Video Games) and wins.
- **Candidates (engine p7/1, `multabench/ours/p7.py`):**
  - Label-free: `frozen0` (control = our Joint Frozen); `tf_num` (+ numeracy); `tf_lsa` (+ 30 LSA components:
    TF-IDF word + char of all text columns → TruncatedSVD, fit on training rows); `tf_row` (+ PCA-30 of a frozen e5
    embedding of the verbalized full row "col: val | ...", text + structured); combinations.
  - Label-aware (on top of `cte`): `cte_probe` (+ a cross-fitted linear probe: Ridge on the full 384-d frozen
    embeddings of every text column, alpha by training-row OOF metric, the same machinery as the lexical expert);
    `cte_rowlex` (+ a cross-fitted lexical expert on the verbalized full row: text tokens + structured tokens, i.e.
    numeric deciles and categorical values); combinations.
- **Budget:** each job's runtime is measured; the dev sweep stays under 1 hour of laptop CPU (rule 9). Any full run
  needs Dima's approval with a measured estimate.

## E27 — P7 dev sweep: LSA + numeracy (label-free) and the linear probe (label-aware) help; verbalized-row embedding does not   (2026-10-16, laptop CPU)
- **Command:** `python scripts/dev_eval.py --datasets <6 dev datasets> --methods frozen0 tf_num tf_lsa tf_row
  tf_num_lsa tf_all cte cte_probe cte_rowlex cte_lsa_row cte_probe_rowlex cte_all --seeds 0 1
  --out .explore/dev/results.csv` (protocol in P7: dev splits of fold 0's training rows; test rows never used).
- **Runtime: 144 jobs, 99 min of CPU.** That was **over the 1-hour rule**: the estimate (45–55 min, from Product
  Sentiment) missed the slower CatBoost fits (Kickstarter, Video Games) and the one-time CPU encoding. Claude should
  have stopped at 1 h. From now on, long sweeps are checked at the 1-hour mark.
- **Results (mean over 2 seeds, difference vs baseline, per dataset: Babies / Book Price / Jigsaw / Kickstarter /
  Product Sentiment / Video Games):**
  - label-free vs our frozen (`frozen0`):
    - `tf_num` +0.0088 (6/6);
    - `tf_lsa` +0.0163 (6/6);
    - `tf_row` +0.0007 (3/6);
    - **`tf_num_lsa` +0.0184 (6/6: +0.026 / +0.026 / +0.003 / +0.017 / +0.016 / +0.022)**;
    - `tf_all` +0.0121 (4/6; adding the row embedding hurts);
    - for reference, `cte` − `frozen0` = +0.0305 (6/6).
  - label-aware vs `cte`:
    - **`cte_probe` +0.0049 (5/6: −0.005 / +0.006 / +0.018 / +0.004 / +0.005 / +0.001)**;
    - `cte_rowlex` +0.0022 (4/6);
    - `cte_lsa_row` +0.0019 (4/6);
    - `cte_probe_rowlex` +0.0047 (5/6, Video Games −0.007);
    - `cte_all` +0.0030 (5/6, Video Games −0.017).
- **Conclusions:**
  - Label-free LSA + numeracy recovers about 60% of CTE's gain over frozen without using labels.
  - The cross-fitted linear probe on the full 384-d frozen embeddings is the one label-aware addition that helps
    consistently, most on Jigsaw (CTE's worst dataset vs TAR), as the "PCA keeps variance, not target signal"
    argument predicts.
  - The verbalized-row e5 embedding does not help (frozen e5 is not trained on "col: val" rows).
  - Dev noise is about ±0.005 per dataset, so the +0.005 probe gain is suggestive, not established.
- **Hostile review:**
  - Dev splits come from training rows only (the test split is discarded in `load_train_rows`), and expert caches use
    dev fold ids (100 + seed).
  - The 6 dev datasets are among the 20 of any later claim.
  - All 20 datasets' test results were already seen (C2, C3), and the numeracy-vs-frozen observation was post hoc
    (P7). Both are disclosed in the P8 proposal.
- **Next:** P8 proposal (below), awaiting Dima's approval; measured Colab estimate from E24 job times.

## P8 — PROPOSED pre-registration (2026-10-16). Becomes binding when Dima approves the run, before any P8 result.
- **C4 (label-free; the course's criterion for methods that do not use y):** `tf_num_lsa` (CatBoost: frozen e5
  PCA-30 per text column + numeracy features + 30 LSA components of TF-IDF word + char of all text columns; no label
  used) vs leaderboard **Joint Frozen** (`all`, CatBoost), 20 datasets, folds 0–4, fold-averaged differences,
  two-sided Wilcoxon. Pass = p < 0.05 and positive mean.
  - *Environment control (secondary):* `tf_num_lsa` vs our own `frozen0` (our Joint Frozen) on the 19 datasets
    without Spotify Genres (its CatBoost alone would add ≈ 10 slot-hours).
  - *Descriptive:* `tf_num_lsa` vs Joint TAR.
- **C5 (label-aware):** `cte_probe` (CTE + cross-fitted linear probe on the full frozen embeddings; alpha from
  (0.3, 1, 3, 10) by the training-row OOF metric) vs leaderboard **Joint TAR** (`ft`, CatBoost), 20 datasets, the same
  test. Pass = p < 0.05 and positive mean.
  - *Secondary:* the held-out 13 (C2_heldout13 list); `cte_probe` vs our `cte` (paired).
- **Multiplicity:** Holm over {C4, C5}. C2 and C3 stay as reported; C4 and C5 never replace them.
- **Disclosure:** the variants were chosen on dev splits of the training rows of 6 of the 20 datasets (E27). All 20
  datasets' test results of earlier methods were seen before this proposal; `cte_num` vs Joint Frozen was computed
  post hoc (P7).
- **Run plan (estimated from E24's measured job times, 6 slots × 2 threads, longest jobs first):**
  - `jobs/p8_A.txt`: `cte_probe` × 100, ≈ 21 slot-hours, **≈ 3.6 h wall**.
  - `jobs/p8_B.txt`: `tf_num_lsa` × 100 + `frozen0` × 95, ≈ 23 slot-hours, **≈ 3.8 h wall**.
  - Each part needs Dima's approval (rule 9). Results root `results_cte` (the CTE caches are reused).
- **Runner setting for P8 (execution only):** `--job_timeout_min 240` (EXTRA_ARGS on Colab). The 30-min default
  (P6 amendment 2) would kill the Spotify jobs, which took up to 152 min in E24.
- **APPROVED by Dima (2026-10-16), before any P8 run or result: P8 (C4, C5, Holm over the two, the secondaries and
  the run plan above) is now binding.** Order: part B (`jobs/p8_B.txt`) first, then part A (`jobs/p8_A.txt`), in two
  separate Colab sessions.

## E28 — P8 part B: claim C4 PASSES (label-free tf_num_lsa vs Joint Frozen)   (2026-10-16/17, Colab A100)
- **Data:** `results_export_20261015_170344.zip`: `tf_num_lsa` 100/100 and `frozen0` 95/95, all at c82596a (clean),
  A100-SXM4-40GB, 2 threads per job, no process_errors and no timeouts. Analysis script committed before the export was
  opened (872f659): `python scripts/p8_report.py --out <export>/results` → `reports/p8/p8_report.json`.
- **C4 (CatBoost tf_num_lsa vs leaderboard Joint Frozen, 20 datasets, two-sided Wilcoxon):** mean **+0.0129**,
  CI [+0.0060, +0.0218], **p = 1.9e-5**, wins **18/20** → **PASSES**. The Holm-adjusted p is ≤ 3.8e-5 whatever C5
  gives, so C4's verdict is final.
- **Environment control (19 datasets, no Spotify):**
  - our `frozen0` vs leaderboard Joint Frozen: **+0.0015, p = 0.47**, 9/19, so no systematic environment effect;
  - `tf_num_lsa` vs our `frozen0`: **+0.0119, p = 3.8e-5**, 17/19, so the gain is the method's.
- **Descriptive:** `tf_num_lsa` vs Joint TAR −0.0076 (5/20). Label-free features do not reach TAR, which is expected
  (Jigsaw −0.108).
- **Per dataset (vs leaderboard Joint Frozen):** only Women Clothing −0.0009 and Product Sentiment −0.0006 are below.
  The largest gains are Video Games +0.076, Rotten Tomatoes +0.052, Babies +0.021, Book Price +0.015 and Fake Job
  +0.015.
- **Hostile review:**
  - *Label-free:* numeracy, TF-IDF/LSA and PCA use no labels and are fitted on training rows only (p7 tests).
  - *Splits:* n_train / n_test are identical to the CTE run.
  - *Environment:* the control shows no overall effect, but **Video Games** is an outlier. Our frozen0 is +0.039 above
    the leaderboard's Joint Frozen there, so half of that dataset's C4 gain is environment. Post-hoc robustness
    without Video Games: +0.0096, 17/19, p = 3.8e-5 (vs our frozen: +0.0105, 16/18, p = 4.2e-4). The conclusion
    does not depend on it.
  - *Disclosure:* the variant was chosen on dev splits (E27), and the numeracy-vs-frozen observation was post hoc (P7).
- **Runtime:** tf_num_lsa 19.6 slot-hours, frozen0 4.7 slot-hours (about 3.8 h wall, as estimated).
- **Next:** part A (`jobs/p8_A.txt`, cte_probe, claim C5), approved.
