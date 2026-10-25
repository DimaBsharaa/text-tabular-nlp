# jobs/

Each file lists the benchmark jobs of one final run, one per line: `method learner dataset fold`. Run with
`python scripts/run_experiment.py --jobs_file jobs/<file> --out <results root>` (or the Colab runner,
[`notebooks/colab_run.ipynb`](../notebooks/colab_run.ipynb)). Finished jobs are skipped.

| File | Run | Analysis |
|---|---|---|
| [`p14_A.txt`](p14_A.txt) | Depth-probe version (`cte_depth_top3`, the final method without the attention probe), CatBoost, 20 datasets × 5 folds (100 jobs) | `reports/p14/` (C9) |
| [`p14_B.txt`](p14_B.txt) | Depth-probe version and its earlier version without depth probes (`cte_probe`), CatBoost, 36 further datasets × folds 0–2 (216 jobs) | `reports/p14/` (C7) |
| [`p15.txt`](p15.txt) | Depth-probe version, LightGBM and TabM, 20 × 5 (200 jobs) | `reports/p15/` (C10) |
| [`p16.txt`](p16.txt) | Depth-probe version, TabPFN-v2 and TabPFN-2.5, 18 × 5 (180 jobs) | `reports/p16/` (C11) |
| [`p16_timing.txt`](p16_timing.txt) | 6 jobs used to measure TabPFN runtimes before the full run | – |
| [`p17.txt`](p17.txt) | Depth-probe version, LightGBM, cross-fitting fold seeds 43 and 44, 20 × 5 each (200 jobs) | `reports/p17/` |
| [`p18.txt`](p18.txt) | Final method with the attention probe (`cte_depth_top3_attn`), all 5 learners, 20 × 5 (TabPFN: 18 × 5; 480 jobs) | `reports/p18/` (C13) |

Development runs (earlier methods) are in [`archive/jobs/`](../archive/jobs/).
