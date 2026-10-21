# Paper outline (4 pages NeurIPS + appendix tables). Working file; placeholders are marked [PENDING: ...]

Claim to make: replacing TAR's text features with ours (a learner-agnostic, label-aware feature construction, no
fine-tuning) improves over Joint TAR **with the same learner**. State the reading explicitly. Never write a bare "we beat TAR".

## Title / abstract (write last)
- One sentence on the idea, one on the result with the exact scope (learners, datasets), one on the main finding
  (no cross-fitting: −0.045; depth profile), one on the limitation (Jigsaw).

## 1. Introduction (~0.5 page)
- Text-tabular learning; MulTaBench; TAR = fine-tune the encoder on the target, then PCA-30 per text column.
- Our question: must the representation be made target-aware by fine-tuning, or is it enough to make the features
  target-aware honestly (cross-fitted)?
- Contributions: (1) cross-fitted text experts + probe + depth probes on a frozen e5; (2) results vs Joint TAR on 20
  datasets and a held-out confirmation on 36; (3) analysis of what matters.

## 2. Related work (~0.3 page)
- MulTaBench (arXiv 2605.10616 / course PDF [1]), TabSTAR [2], TAR; e5 [12]; LoRA [13]; stacking / cross-fitting
  (Wolpert 1992, Breiman 1996); probing intermediate layers (cite carefully, no invented references).
- The course PDF's reference list is a safe source for citations. Anything else: check before citing.

## 3. Method (~1 page)
- Figure 1 (draw): text columns → frozen e5 → {lexical TF-IDF+ridge, kNN, linear probe on final layer, depth probes on
  3 best layers, numeracy} → cross-fitted predictions → CatBoost / any learner.
- Cross-fitting definition (5 folds on training rows; test rows get experts fit on all training rows).
- What is label-free vs label-aware (course clarification 2026-10-06: target values, not names).
- Leakage controls: everything fit on training rows only; TF-IDF label-free and fit on training rows; tests.

## 4. Experiments (~1.7 pages)
- Setup: 20 MulTaBench text datasets, 5 folds, upstream splits and metrics, upstream learner defaults, seed 42.
- Statistics: fold-averaged per-dataset differences, two-sided Wilcoxon over datasets, p < 0.05, bootstrap CI;
  Holm over all pre-registered tests. State pre-registration and the development path honestly (C1–C3 failed).
- Main result table: final method vs same-learner Joint TAR for all 5 learners (C9, C10, C11: reports/p14, p15, p16),
  with per-learner Holm and Holm over all 17 tests (C10-LightGBM 0.061 does not survive the global one). C7 (36) separately.
- Per-dataset figure: reports/figures/fig6_c4_c5_per_dataset.png (replace by C9 per-dataset when exported).
- Cross-learner table: our CatBoost vs each learner's Joint TAR (computed 2026-10-20: LightGBM, TabPFN-v2 significant;
  TabM p = 0.058; TabPFN-2.5 tie) and same-learner results [PENDING: P15].
- Ablations (reports/tables/table_ablations.csv, fig3): cross-fitting essential (in-sample −0.045); experts complementary.
- Depth profile figure (reports/figures/fig7_depth_profile_dev.png): toxicity peaks in low layers, sentiment in the
  middle, prices at the top. Dev-split evidence, say so.
- Negative results in one compact table: C1, C2, C3, the query prefix (C6), P12 refinements.

## 5. Conclusion and limitations (~0.3 page)
- Jigsaw (toxicity) is where fine-tuning still wins (all learners); TabPFN results cover 18 datasets (no baseline for
  Spotify / Wine); compute (no fine-tuning, no GPU
  training); single seed; environment offset on Video Games; reuse of the 20 datasets during development, mitigated
  by pre-registration and the 36-dataset confirmation.

- Future work (Dima, 2026-10-20): test the method on a newly curated text-tabular dataset, our VAERS adverse-event
  reports (Track 2 work: free-text symptom excerpts + structured report fields, binary "serious" target). Describe it
  in words. The VAERS repository is private, so do not link it unless it is made public.

## Appendix (tables only, per the PDF)
- Per-dataset scores for every learner run; the full claim table with p-values and Holm; the 36-dataset table.

## Code and reproducibility paragraph
- Pinned deps (uv.lock), one command in the README, train.py / predict.py, no weights uploaded (nothing is fine-tuned;
  the encoder is the public intfloat/e5-small-v2).
