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
