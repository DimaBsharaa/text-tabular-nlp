# Working rules for Claude Code in this repository

Full original project brief: `archive/CLAUDE_project_brief.md` (local only, git-ignored; read it at the start of a session).
Owner: Dima (GitHub DimaBsharaa). Upstream: MulTaBench @ tag `paper_version` (identical to upstream `master` of
2026-07-26 for the 20 text datasets, see archive/PROGRESS.md E47). Final method: `cte_depth_top3`.

## Layout
- Final method: `multabench/ours/` (cte.py, p7.py, depth.py + shared machinery); earlier versions in
  `multabench/ours/archive/`. Scripts in `scripts/`; earlier versions' scripts, jobs, reports, notebooks in `archive/`.
- Upstream MulTaBench code (everything else under `multabench/`, `benchmark.py`): never modified.

## Rules
1. **Lab notebook.** Read `archive/PROGRESS.md` at the start of every session. After every experiment append an entry:
   date, git commit, question, exact command/config, result numbers, conclusion, next step.
2. **Pre-registration.** Before any multi-dataset run, write the claim in `archive/PROGRESS.md` (method, baseline, metric,
   datasets, folds, learners, statistical test, success threshold). Never edit it afterwards; append changes and why.
3. **Hostile review.** After every result, list what could make the number wrong (leakage, split mismatch, seeds,
   metric, train size, test-chosen hyperparameters) and check each with evidence.
4. **Output hygiene.** Redirect long logs to files; read only the tail or a summary.
5. **Upstream protection.** Never modify upstream files except adding our CLI option in `benchmark.py`. Before every
   commit run `git diff --stat paper_version` and show Dima the result.
6. **Resumable experiments.** One JSON per (method, learner, dataset, fold); skip if it exists.
7. **tar_repro is frozen at commit 96c1f5c** (engine `tar_engine/2`); `tests/test_tar_repro_frozen.py` guards it.
   Every result JSON records `engine_version`.
8. **Secrets & teaching.** Secrets live only in `.env` (gitignored) / Colab Secrets; never print or commit them, never ask
   Dima to paste them. W&B disabled. Explain things simply.
9. **Runtime approval.** Every new job comes with a measured runtime estimate; nothing over 1 hour total runs without
   Dima's approval.
10. Commits use pytest's own exit code. Every tracked file outside the lab notebook is written for an outside reader:
    the method, the evidence and how to run it, nothing about the project's administration.
