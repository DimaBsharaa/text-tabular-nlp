# Paper notes — points for the 4-page write-up

Working notes for writing the NeurIPS-format paper. Every number below comes from `PROGRESS.md` (entry in brackets)
or from the MulTaBench paper (arXiv 2605.10616, tag `paper_version`). Items marked **[check]** come from the
MulTaBench paper text and still need a page or figure number before we cite them.

## 1. Relation to the MulTaBench paper

### 1.1 Our TAR reproduction matches their Appendix A (footnote on weight decay)
- `tar_repro` follows Appendix A **[check]**: LoRA r = 16 on the top 3 layers of e5-small-v2, a linear head,
  lr 1e-4, ≤ 50 epochs, patience 3, a 90/10 train/val split, each text cell encoded alone as "col: val", and
  regression targets binned into 20 quantile bins with cross-entropy.
- **Footnote:** the paper text gives weight decay 0.01 **[check]**. The code that produced the leaderboard uses
  **0.001**. `benchmark.py` builds `E5TrainArgs()` from `multabench/finetune/train_args.py` (`weight_decay = 0.001`)
  and passes it as `--e5_weight_decay`. That overrides the `0.01` default of `multabench/e5/e5_finetune.py`. We
  followed the code (`multabench/ours/tar_engine.py`, `TarConfig.weight_decay = 0.001`).
- Our engine has no significant bias against the leaderboard's Joint TAR: −0.0036, p = 0.15, over 18 pilot jobs
  (E8; largest on Book Price −0.018). It is frozen at 96c1f5c and guarded by golden tests. A/A seed noise is
  ≈ 0.008 per fold (E9).

### 1.2 Two of their open problems, which CTE addresses cheaply
- They state that one E5 fine-tuned jointly on all text columns may degrade with many columns **[check]**. CTE
  uses no shared fine-tuned encoder. It fits one cheap expert per column (neighbour expert) and one on all columns
  concatenated (lexical expert), each from the training rows of that job only.
- They list steering the embeddings toward the target as future work **[check]**. CTE adds the target signal
  directly, as cross-fitted out-of-fold predictions next to the target-free PCA components, with no gradient
  training of the encoder.

### 1.3 PCA is NOT the whole problem (do not over-claim)
- Their Fig. 6 **[check]**: TAR beats Frozen at 15, 30 and 60 PCA components. So we must **not** claim that PCA
  alone explains the TAR–Frozen gap.
- **Our claim:** PCA keeps the directions of largest *variance* in the embeddings, not the directions that predict
  the *target*. An OOF prediction column gives the learner an estimate of E[y | text] directly, in one feature,
  whatever the number of PCA components.

### 1.4 "TAR Small beats Frozen Large" **[check]**
- Their finding: a fine-tuned small encoder beats a frozen large one. So capacity alone does not add task
  awareness. CTE adds task awareness cheaply, with a small frozen encoder plus cross-fitted experts.

## 2. Our results (exact pre-registered labels)

### 2.1 Claim C2 (P5 v2, CTE, engine cte/3) — **not significant on all 20 datasets** [E24]
- CatBoost CTE vs leaderboard Joint TAR, 20 datasets, fold-averaged, two-sided Wilcoxon:
  mean **+0.0106**, median +0.0044, bootstrap 95% CI [+0.0028, +0.0193], **p = 0.058**, wins 11/20,
  rank-biserial r 0.49. The pre-registered threshold is p < 0.05, so **C2 does not pass**. Report it exactly so.
- **Held-out 13** (pre-registered in amendment 2: the 20 minus every dataset used in exploration or profiling):
  mean **+0.0173**, CI [+0.0073, +0.0282], **p = 0.0105**, wins 9/13 → passes. Label it "pre-registered held-out
  subset (secondary)", not the primary claim. The strict 11 also passes (p = 0.042). The original 16 and 14 subsets
  do not (p = 0.065 and 0.22).
- **LightGBM** CTE vs Joint TAR (pre-registered secondary, Holm): +0.0152, **p = 0.021** → passes.
- vs Joint Frozen: +0.031, p = 1.9e-6, 20/20 wins. This is a weaker baseline; use it as context, not as the claim.
- Disclose: amendments 1–2 (speed changes, the multiclass column budget, OOF alpha selection) were made after
  seeing fold-0 test scores of 4 datasets (Product Sentiment, Book Price, Jigsaw, Wine). Those, plus Spotify,
  are excluded from the held-out 13.

### 2.2 What drives it [E24]
- **Cross-fitting is essential:** the lexical expert used in-sample (not cross-fitted) scores **−0.045** vs CTE
  (−0.034 vs Joint TAR). This mirrors the known in-sample weakness of TAR, whose training-row embeddings come
  from an encoder fine-tuned on those same rows.
- Ablations (CatBoost, mean vs CTE): lexical only −0.011 (≈ Joint TAR level), neighbour only −0.017,
  numeracy only −0.021. The experts are complementary.
- CatBoost importance share (CTE): PCA 0.42, lexical 0.21, structured 0.15, neighbour 0.13, numeracy 0.09.
- Largest gains: Rotten Tomatoes +0.056, Mercari +0.045, Fake Job +0.042, DS Salary +0.030, Book Readability
  +0.027.
- **Jigsaw is the fine-tuning exception:** −0.022, losing all 5 folds. On toxicity the meaning of the words matters
  more than which words appear, and the leaderboard's TAR − Frozen gap is largest there (+0.108 with CatBoost; +0.117 averaged over the 5
  learners). TF-IDF + Ridge
  tops out at an OOF AUROC of ≈ 0.86 [E22]. Present this honestly as a limitation.

### 2.3 Compute — careful wording [E6, E17, E24]
- **Do NOT write "~10× cheaper" as a measured wall-clock result against our own TAR engine.** Measured
  text-representation time over the same 100 (dataset, fold) jobs: TAR fine-tuning with our optimised engine 3.07 h
  (exclusive A100) vs CTE frozen embeddings + experts 2.76 h (CPU, 6 jobs sharing the machine, 2 threads each).
  That is ≈ 1.1×, on different hardware.
- **What we can say:**
  - CTE needs **no gradient training and no GPU** beyond one frozen forward pass per unique text, cached once per
    dataset and shared by all folds and learners.
  - The experts are a few seconds to minutes of CPU per job.
  - Against upstream's TAR implementation, our engine is 11.9× faster per epoch (E6, one dataset, T4), so CTE is
    roughly an order of magnitude cheaper than TAR as implemented upstream. Label it as an extrapolation from E6.
- Wall time of the full CTE run (600 jobs, A100 VM, 6 parallel jobs): 13.2 h; the CTE CatBoost block took 3.4 h.
  Multiclass CatBoost itself dominates (Spotify ~2.5 h per job), the same for every method.

### 2.4 Claim C3 (P6) — pre-registered 2026-10-14, results pending
- Naming: P5 v1 also had a "C3" (frozen_experts), never run and superseded by P5 v2. In the paper, call
  this one "C3 (P6)" or rename it consistently.
- cte_plus_native, cte_nb and cte_residual (CatBoost) each vs Joint TAR; Holm over the 3; pass = adjusted
  p < 0.05 and positive mean. Also the held-out 13. `catboost_native_text` is reported as a baseline, not a claim.
- C3 never replaces C2. Report both.

## 3. Things not to claim
- That CTE "significantly beats Joint TAR" on all 20 datasets (C2: p = 0.058).
- That PCA is the whole problem (their Fig. 6).
- A measured 10× wall-clock saving against our own TAR engine (§2.3).
- Anything from the held-out subset as if it were the primary test.
