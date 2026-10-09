# Method spec — TAR variants (pilot P2)

Written 2026-10-10, **before any variant is implemented or run**. The baseline is `tar_repro` (Joint TAR,
re-implemented in `multabench/ours/tar_engine.py`, frozen at commit 96c1f5c / engine `tar_engine/2`). Every variant
keeps TAR's encoder (e5-small-v2), LoRA (r 16, alpha 32, dropout 0.1, top 3 layers), head, optimizer, schedule,
batch size, epochs, early stopping, PCA (30 per text column) and downstream learner unchanged, and changes only
what is listed below. Section 7 lists the details the spec left open and how they were resolved (Claude's choices,
to be confirmed or corrected by Dima before the variant runs start).

## 1. Shared module `multabench/ours/structured_offset.py`

- **Input:** the rows the encoder is trained on — the same E5 train + val pool TAR uses for this learner (the
  learner's training rows handed to the text-encoder step) — and their **structured** features after upstream's own
  preprocessing (`TabularModel.do_model_agnostic_preprocessing`: dates expanded, numerics converted, categoricals as
  strings). No new feature engineering.
- **OOF model:** 5-fold out-of-fold CatBoost on structured features only (500 iterations, depth 6, fixed seed 42).
  Classification → OOF class probabilities; regression → OOF predictions.
- **Full fit:** one more CatBoost on the whole pool, used only for (a) feature importances and (b) predictions for
  test and learner-val rows, which are used **only for diagnostics, never as encoder inputs**.
- **No structured columns:** the offset is the class prior / the mean, so the variant equals TAR (see 7.6).
- **Never** use test rows or learner-val rows to fit anything.

## 2. Variant `offset` (idea B — "learn what the table can't explain")

Same input texts and hyperparameters as TAR (so it reuses TAR's lower-layer cache).
- **Classification:** `logits = head(h) + log(clip(p_oof, 1e-3, 1))`; cross-entropy on those logits.
- **Regression:** target `y − y_oof`, discretized into 20 quantile bins on the E5-train rows (same discretize path
  as upstream); cross-entropy.
- The offset only enters the loss. Embeddings are extracted and PCA'd exactly as in TAR; the offset is **not**
  given to the downstream learner.

## 3. Variant `prefix` ("structured feature tokens") and `prefix_offset`

Input text exactly as in TAR (cache reused). For the **top-8 structured features by full-fit importance**, one token
each: `token_j = W_name · E5_frozen(column_name_j) + value_enc_j(x_j)`.
- Numerics: piecewise-linear encoding over 16 training-set quantile bins → linear layer to 384 dims.
- Categoricals: a learned embedding per category (rare/unseen → "other").
- Missing: a learned "missing" vector per feature.
- The 8 tokens are inserted at the input of layer 9 (the first trainable layer); the attention mask is extended so
  text tokens can attend to them; LoRA as in TAR; mean-pool over **text tokens only**. The new parameters train
  with the head (same lr, same early stopping). Target and loss as in TAR.
- `prefix_offset` = `prefix` + the offset loss of section 2.

## 4. Variant `context` (idea A) and `context_offset`

- Input = `f"{context} | {col}: {val}"`. `context` joins the top-8 structured features by importance as
  `name=value (decile d/10)` for numerics (3 significant digits), the raw string (max 30 chars) for categoricals,
  `missing` for NaN. Context first, capped at 80 tokens; the text is truncated so the total stays ≤ 512 tokens.
- Target and loss as in TAR. New lower-layer cache (the texts change).
- `context_offset` = `context` + the offset loss of section 2.

## 5. Diagnostics (every result JSON)

- TAR head qualities as now (e5_train, e5_val, learner_val, test).
- Offset variants additionally: `struct_only_quality` (OOF on e5_train / e5_val, full fit on learner_val / test),
  `combined_quality` (offset + head) on e5_train, e5_val, learner_val and test, and `lift = combined − struct_only`.
- `engine_version` and the method in every JSON; cache keys include the method and a hash of the input texts.

## 6. CLI, jobs, tests

- `--method {tar_repro, offset, prefix, prefix_offset, context, context_offset}`; one jobs file per variant (the
  same 18 pilot jobs, seed 42).
- CPU tests: (1) OOF: every row's offset comes from a model that never saw its label; (2) with a zero head, logits
  equal log(offset); (3) the verbalizer is deterministic and respects token caps; (4) prefix reuses tar_repro's
  cache (same text hash); (5) with all feature tokens masked out, prefix embeddings equal TAR's; (6) the numeric
  encoding is smooth; (7) every variant runs end to end on a tiny setting; (8) with no structured columns, offset ==
  tar_repro. Plus: tar_repro stays bit-for-bit (golden fingerprint test).
- `scripts/sensitivity_probe.py` (to run later): for N test rows, perturb one structured feature (numeric ± 1
  quantile bin, categorical → another frequent category) and record the cosine change of the text embedding.

## 7. Details the spec left open — Claude's choices (to confirm)

1. **Pool and split.** Pool = the learner's rows handed to the text-encoder step (`x_fit`), i.e. e5_train ∪ e5_val.
   Every variant uses **TAR's own E5 train/val split** (stratified on TAR's labels / TAR's 20 y-bins), so the rows
   are identical to tar_repro's.
2. **Structured features** = the learner's `numerical_features ∪ categorical_features` (upstream detection),
   excluding text columns. CatBoost gets the categoricals as native `cat_features`; numeric NaN left as NaN.
3. **OOF folds.** `StratifiedKFold(5, shuffle=True, random_state=42)` on the class label (cls) or on TAR's y-bin
   (reg); plain `KFold` if a stratum has < 5 rows. CatBoost: `iterations=500, depth=6, random_seed=42`,
   default learning rate, no early stopping (no extra validation split).
4. **Class alignment.** OOF probabilities are mapped onto TAR's E5 classes; a class absent from a fold's training
   part gets probability 0 → clipped to 1e-3. No renormalization (the spec's formula, exactly).
5. **Regression bins.** Residual bin edges = upstream `discretize_numerical`'s quantile edges on the **e5_train**
   residuals (20 bins, duplicate edges dropped); outer edges extended to ±∞ and applied to the e5_val residuals.
   (TAR bins y over the whole pool; the spec asks for e5_train-only edges here.)
6. **No structured columns** → the offset variants run exactly the tar_repro computation (same cache entry) and the
   JSON records `offset_delegated: true`. Reason: a constant log-prior offset would be absorbed by the head bias
   only approximately, so "equals TAR" is guaranteed only by delegating.
7. **Regression diagnostics.** `combined = y_struct + Σ_b p_b · mean_residual_b` (bin means on e5_train);
   `head_quality` of offset variants is measured against the variant's own target (residual R² / class labels).
8. **Feature-token details.** `E5_frozen(name)` = masked mean of the frozen embeddings + layers 0–8 output for the
   string `"passage: {name}"` (the space the tokens are inserted into). `W_name` = one shared 384×384 linear map,
   initialised to the identity. Numeric PLE edges = unique quantiles of the pool values; numeric encoder =
   Linear(16 → 384) initialised N(0, 0.02). Categories seen ≥ 3 times in the pool get their own embedding, the rest
   and unseen ones share "other". Missing = NaN or upstream's missing-value placeholder → the feature's learned
   missing vector replaces the value encoding. Attention between all tokens is bidirectional (feature tokens also
   attend to text). Prefix requires the fast (cached lower layers) path.
9. **Per-row embeddings.** For `prefix` and `context` a text cell's embedding depends on its row's structured
   features, so embeddings are stored per (row, column) instead of per unique text.
10. **Verbalizer format.** Features in importance order, joined with `"; "`: numerics `name=1.23 (decile 4/10)`
    (`.3g`, deciles from the pool), categoricals `name=<first 30 chars>`, missing `name=missing`. Tokens:
    `[CLS] passage: <context ≤ 80 tokens> | <col>: <val> [SEP]`, text truncated so the total is ≤ 512.
11. **Engine versions.** Variants record `variants/1`; tar_repro keeps `tar_engine/2`.
