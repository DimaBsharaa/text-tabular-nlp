# Project brief — NLP Final Project (Technion 097215, Spring 2026), Track 1: Algorithm

Owner: Dima (GitHub: DimaBsharaa, Kaggle: dimabishara, HF: DimaBsharaa). Individual work.
**Deadline: Monday Oct 26, 2026** (−20 pts/day late). Deliverables: code + eval results + 4-page NeurIPS paper.

## The task
Design a method for prediction on **text-tabular** data (tables with free-text columns), evaluated on the
**20 text-tabular datasets of MulTaBench** (AUROC for classification, R² for regression).
- If the method does NOT use y to build text representations → must significantly beat **Joint Frozen**.
- If it DOES use y (fine-tuning etc.) → must significantly beat **Joint TAR**. ← our case (most likely)
- Creativity is graded: swapping to a bigger encoder is NOT a valid contribution.
- No test leakage. Code must run on any text-tabular dataset, pinned deps (pyproject + uv.lock),
  inference mode + end-to-end training mode; upload weights to HF only if we pretrain anything.

## Upstream code: MulTaBench @ tag `paper_version`
Repo: https://github.com/alanarazi7/MulTaBench (paper: arXiv 2605.10616). ALWAYS use `paper_version`, not master
(master changed splits and metrics). `tabstar==1.1.15` is a pip dependency — do not clone TabSTAR.

- Entry: `python benchmark.py --model <catboost|lgbm|tabm|tabpfnv2|tabpfnv2p5|...> --dataset_name <ID> --fold <0-4> --multimodal_state <state>`
- States: `no_text`=Unimodal Structured, `text_only`=Unimodal Unstructured, `all`=**Joint Frozen**, `ft`=**Joint TAR**
- Protocol: 5 folds; test = 10% capped at 2,000; train capped at 10,000. Metrics in `baselines/training/metrics.py`
  (binary AUROC, multiclass macro-OvR AUROC, regression R²).
- Pipeline: `baselines/abstract_model.py::TabularModel.fit` → preprocessing → **`fit_text_encoders()`** in
  `baselines/preprocessing/text_embeddings.py` (E5-small-v2 → PCA 30 per text column) → learner.
  **This is the single plug-in point for our method.**
- TAR baseline (`fit_text_encoders_tuned`): LoRA r=16 on top-3 layers of e5-small-v2, linear head, lr 1e-4,
  ≤50 epochs, patience 3, 90/10 train/val; each text cell encoded ALONE as "col: val" (no structured context);
  regression targets → 20 quantile bins + cross-entropy.
- Known weakness: embeddings for training rows come from the encoder fine-tuned on those same rows (in-sample).
- Each learner re-fine-tunes E5 → we should cache embeddings per (dataset, fold) and reuse across learners.
- **Precomputed baselines**: `multabench/leaderboard/results/text/*.csv` — all 20 datasets × 5 learners ×
  4 states × 5 folds. Compare against these; reproduce a few to validate our environment.
  TAR−Frozen gain is mostly +0.004..+0.017 (Jigsaw +0.117, Video Games +0.040, Fake Jobs 0).

## Candidate ideas (to be decided with Dima — do not implement before we choose)
A. Context-aware TAR: fine-tune on text + verbalized structured row context (TabSTAR-style quantile verbalization).
B. Residual TAR: fine-tune E5 on residuals/errors of a structured-only model → complementary text signal.
C. Cross-fitted TAR: out-of-fold embeddings for training rows.
D. Ordinal/soft-label loss for regression instead of 20-bin CE.

## How we work
- Laptop (Windows + WSL Ubuntu): Claude Code edits code, CPU smoke tests on tiny subsets.
- Google Colab Pro (GPU): `git pull` + run experiments; results written to Google Drive.
- All experiment scripts must be **resumable**: one JSON per (method, model, dataset, fold); skip if exists.
- Keep upstream eval code untouched; our code lives in `multabench/ours/` + a new CLI option.
- Secrets live only in `.env` (gitignored) / Colab Secrets. Never print or commit tokens. W&B disabled (`WANDB_MODE=disabled`).
- Explain decisions to Dima as we go — he wants to understand everything, not just get code.
