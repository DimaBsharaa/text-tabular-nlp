# Paper outline (4 pages NeurIPS + appendix tables). Working file; placeholders are marked [PENDING: ...]

Principle (Dima, 2026-10-21): the paper presents OUR method. Story = algorithm → pain points of Joint TAR we measured →
how each part of the method answers one → a short analysis-driven path to the design (it did not fall from the sky).
No catalogue of failed attempts in the main text; at most one line each in the appendix.

Claim to make: replacing TAR's text features with ours (label-aware, cross-fitted, no fine-tuning) improves over Joint
TAR **with the same learner**, for all 5 MulTaBench learners. State the reading explicitly. Never write a bare "we beat TAR".

## Title / abstract (write last)
- Idea (cross-fitted text experts on a frozen encoder), result (5 of 5 learners vs same-learner Joint TAR; 4 of 5 after
  Holm over all 17 pre-registered tests), main finding (cross-fitting is essential: the same lexical expert is 0.035 worse without it), limitation (Jigsaw).

## 1. Introduction (~0.5 page)
- Text-tabular learning; MulTaBench; the pipeline (e5 → PCA-30 → learner); Joint Frozen vs Joint TAR.
- Our idea in one line: make text features label-aware without fine-tuning, through cross-fitted "expert" predictions.
- Contributions: (1) the method; (2) 5-learner results vs Joint TAR on the 20 datasets; (3) analysis of why it works.

## 2. Pain points of Joint TAR (~0.4 page; each one measured by us)
- P1 over-trust: TAR's head is better on the rows e5 was fine-tuned on than on test: +0.031 (16/18 jobs, p = 0.0002; E8).
- P2 PCA is label-blind: 384 → 30 by variance; a probe on the full 384 adds on top (cte_probe vs cte: C5 vs C2).
- P3 last layer only: task signal peaks at different depths (depth profile, dev splits; reports/figures/fig7).
- P4 embeddings blur exact tokens / numbers: lexical expert and numeracy features add (ablation table).
- P5 cost: GPU fine-tuning per dataset and learner; ours is CPU minutes (compute_budget).

## 3. Method (~1 page)
- Figure 1: text → frozen e5 → {lexical TF-IDF+Ridge, kNN, probe on final layer, depth probes (top-3 layers),
  numeracy} → cross-fitted predictions → appended to structured + PCA-30 → any learner.
- Each component mapped to the pain point it answers (P1 → cross-fitting; P2 → probe; P3 → depth; P4 → lexical,
  numeracy; P5 → no fine-tuning).
- Cross-fitting definition (5 folds on training rows; test rows get experts fit on all training rows).
- Leakage controls: everything label-aware fit on training rows only; TF-IDF vocabulary from training rows; shuffle test.

## 4. From diagnosis to design (~0.25 page, one paragraph)
- Cross-fitting TAR's own fine-tuned heads fixes P1 → showed that honest label-awareness, not fine-tuning, is what
  matters → replaced fine-tuning by cheap frozen experts → then read the full embedding (P2) and inner layers (P3).
  Decided on dev splits of training rows, pre-registered before each full run.

## 5. Experiments (~1 page)
- Setup: 20 MulTaBench text datasets, 5 folds, upstream splits and metrics (`paper_version`; identical to the 27.7
  release for these 20, E47), upstream learner defaults, seed 42.
- Statistics: fold-averaged per-dataset differences, two-sided Wilcoxon, p < 0.05; Holm per family and over all 17.
- Main table: final method vs same-learner Joint TAR, 5 learners (C9, C10, C11; reports/p14, p15, p16).
- Per-dataset figure.

## 6. Analysis (~0.6 page)
- Ablations (reports/tables/table_ablations.csv): the same lexical expert without cross-fitting −0.035 (−0.045 vs the full experts); experts complementary.
- Depth profile figure: toxicity low, sentiment middle, prices top.
- Where it loses: Jigsaw (toxicity), where fine-tuning wins with every learner.
- Robustness: 36 extra datasets (C7), medians ≈ +0.003, not significant after Holm; say so plainly.
- Trust checks: independent recomputation; label-shuffle → chance.

## 7. Conclusion and limitations (~0.3 page)
- Jigsaw; TabPFN on 18 datasets (no baseline for Spotify / Wine); single seed; reuse of the 20 datasets during
  development (mitigated by dev splits, pre-registration, 36-dataset check).
- Future work (Dima, 2026-10-20): our VAERS adverse-event dataset (Track 2 work: free-text symptom excerpts + structured
  report fields, binary "serious" target). Describe in words; the repository is private, so do not link it.

## Appendix (tables only, per the PDF)
- Per-dataset scores for all 5 learners; the full claim table with p-values and Holm; the 36-dataset table (note: 2 of
  the 36 loaders differ from the 27.7 release, E47); one line per earlier attempt (C1–C3, C6).

## Code and reproducibility paragraph
- Pinned deps (uv.lock), one command in the README, train.py / predict.py, no weights uploaded (nothing is fine-tuned;
  the encoder is the public intfloat/e5-small-v2).
