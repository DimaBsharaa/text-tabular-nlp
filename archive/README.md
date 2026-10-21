# Archive: how the method was developed

Everything here led to the final method but is not needed to use it. The code is kept importable
(`multabench/ours/archive/`) and every script still runs, so each logged result can be re-run.
The complete day-by-day record (every pre-registration, run, number and check) is the lab notebook
[`PROGRESS.md`](PROGRESS.md); entry numbers (E…) and claims (C…) below refer to it.

**Working method.** Ideas were first compared on development splits of the *training* rows only. Before every
multi-dataset run, the claim, baseline, test and pass threshold were written down (pre-registration) and never edited.
Every result was followed by a "hostile review" that looked for leakage, split mismatches and test-chosen settings.

## The path, stage by stage

| Stage | What we tried | Result | What we learned |
|---|---|---|---|
| 0. Benchmark study (E2) | Read the published baselines before building anything: noise per dataset, how much fine-tuning (Joint TAR) adds over Joint Frozen | TAR beats Frozen significantly for 4 of 5 learners, but the median gain is only +0.011; fold noise is of the same size | The margin to beat is small and noisy: compare per dataset, paired, with a test over all 20 datasets |
| 1. Reproduce Joint TAR (E3–E9) | Our own re-implementation of TAR (`tar_engine.py`), frozen at commit 96c1f5c | Matches the leaderboard (bias −0.004, not significant) | **Pain point found:** TAR's head is +0.031 better on the rows it was fine-tuned on than on test rows (16 of 18 runs, p = 0.0002), so the learner over-trusts the text features |
| 2. Change the encoder's input (P2, E10–E13) | Fine-tune with the row's other columns in the text ("context", "prefix") or with a structured-model offset | None of 5 variants beats the control; context −0.022 | Putting the structured columns into the encoder does not help; the learner already has them |
| 3. Cross-fit the fine-tuned head (P3–P4, claim C1, E14–E17) | Give the learner out-of-fold predictions of TAR's head, which fixes the over-trust | −0.0025 vs Joint TAR (p = 0.41), 22 GPU-hours | Fixing the leak by fine-tuning several times is expensive and no better. Question: does label-awareness need fine-tuning at all? |
| 4. Cheap frozen experts (E18, P5, claim C2, E20–E24) | Frozen e5 + cross-fitted TF-IDF/Ridge, neighbour and numeracy features ("CTE") | +0.0106 vs Joint TAR, p = 0.058 (just misses); runtime cut from an estimated 20–40 h to 3.4 h | No fine-tuning is needed to match TAR. Ablation: the same lexical expert without cross-fitting is 0.035 worse, so honesty of the features is what matters |
| 5. Other experts (P6, claim C3, E25) | CatBoost native text, Naive-Bayes lexical model, residual targets | Not significant after correction (0.097) | The choice of lexical model is not the bottleneck |
| 6. Read the full embedding (P7–P8, claims C4, C5, E27–E29) | A cross-fitted linear probe on all 384 dimensions, instead of only the 30 PCA components | **+0.0141 vs Joint TAR, 15/20, p = 0.0023: first pass.** Label-free variant vs Joint Frozen: +0.0129, 18/20 (C4) | **Pain point:** PCA keeps the directions that vary most, not the ones that predict the target |
| 7. Refinements (P9–P12, claim C6, E30–E33) | Averaged embeddings, prompts, random features, the e5 "query:" prefix, … | The query prefix passes vs TAR but is not better than stage 6 | Diminishing returns from changing how the text is fed to the encoder |
| 8. Other learners (claim C8, E35–E36) | Stage 6 with LightGBM | +0.0167, p = 0.011 | The gain is not specific to CatBoost |
| 9. Read the inner layers (P13–P14, claim C9, E37–E38) | Depth probes: a cross-fitted probe on each of the 12 layers, the best 3 averaged | +0.0033 over stage 6 on development splits; **+0.0161 vs Joint TAR, 14/20, p = 0.0023 → final method** | **Pain point:** the target signal sits at different depths (toxicity layer 3, sentiment layer 9, prices layers 10–12 of 12) |
| 10. Untouched datasets (claim C7, E39) | The final method on 36 further text datasets never used in development | Median +0.003, 24/36, p = 0.017; not significant after correction | The gain carries over, but smaller |
| 11. All five learners (P15–P16, claims C10–C11, E41–E46) | The final method with LightGBM, TabM, TabPFN-v2, TabPFN-2.5 | All 5 pass; 4 of 5 also after correction over all 17 tests | The method is learner-agnostic |
| 12. Trust checks (E40, E47) | Independent recomputation; shuffled-label test; benchmark version check | Numbers reproduced exactly; shuffled labels → chance; benchmark identical to upstream `master` of 26 July 2026 | No leakage path found; the comparison is with the official benchmark |

## What is in this folder

| Path | Contents |
|---|---|
| [`PROGRESS.md`](PROGRESS.md) | The lab notebook: all pre-registrations (P…), entries (E…) and claims (C…), in order |
| [`CLAUDE_project_brief.md`](CLAUDE_project_brief.md) | The original project brief and working rules |
| `../multabench/ours/archive/` | Code of stages 2, 5 and 7 (`p6.py`, `p9.py`, `context_text.py`, `feature_tokens.py`, `structured_offset.py`); our TAR re-implementation stays in `multabench/ours/tar_engine.py` (the final method uses its encoding helpers) |
| [`scripts/`](scripts/) | Reports and tools of earlier claims (`pilot_report.py`, `cte_report.py`, `p6_report.py`, `p8_report.py`, `p10_report.py`, …) |
| [`jobs/`](jobs/) | The jobs files of earlier runs |
| [`reports/`](reports/) | The reports, tables and figures of earlier claims |
| [`notebooks/`](notebooks/) | The analysis notebook of stages 1–5 |
| [`docs/method_spec.md`](docs/method_spec.md) | The specification of the stage-2 variants |
