"""Run one (method, learner, dataset, fold) job, or a list of them, with resumable JSON results.

Single job:
    python scripts/run_experiment.py --method tar_repro --learner catboost --dataset REG_TEXT_BOOK_PRICE --fold 0 --out results
Job list (one "method learner dataset fold" per line, # comments allowed; each job runs in its own subprocess):
    python scripts/run_experiment.py --jobs_file jobs/pilot_tar_repro.txt --out results

Result: <out>/<run_name>/<learner>/<dataset>/fold<k>.json — skipped if it already exists (use --force to redo).
run_name = method, plus a suffix when a result-changing setting differs from the default (e.g. tar_repro__ep1_n200),
so debug runs can never be mistaken for (or block) real ones.
Fine-tuned TAR outputs are cached in <cache_dir> (default <out>/_cache) and shared across learners.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from os.path import abspath, basename, dirname, exists, join
from typing import Optional

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # works with plain `uv sync` (no .pth file needed)
os.environ.setdefault("WANDB_MODE", "disabled")

METHODS = ("tar_repro", "offset", "prefix", "prefix_offset", "context", "context_offset",
           "insample_pred", "xfit_pred", "frozen_xfit_pred", "xfit_offset_pred",
           "tar_experts", "frozen_experts", "tar_experts_lex", "tar_experts_nbr", "tar_experts_lex_insample",
           "cte", "cte_lex", "cte_nbr", "cte_num", "cte_lex_insample",
           "catboost_native_text", "cte_plus_native", "cte_nb", "cte_residual",  # P6 (exploratory, CatBoost)
           "frozen0", "tf_num", "tf_lsa", "tf_row", "tf_num_lsa", "tf_all",  # P7 label-free
           "cte_probe", "cte_rowlex", "cte_lsa_row", "cte_probe_rowlex", "cte_all",  # P7 label-aware
           "p9_base", "p9_avg", "p9_rep3", "p9_rep3_avg", "p9_prompt", "p9_rff", "p9_nb", "p9_query", "p9_percol", "p9_k10", "p9_maxpool", "p9_tok",  # P9 / P12
           "frozen0_query", "tf_num_lsa_query",
           "cte_depth", "cte_depth_mm", "cte_depth_top3", "cte_depth_sl",  # P13 depth probes (Cowork)
           "cte_depth_top3_xs43", "cte_depth_top3_xs44")  # P17: final method, cross-fitting fold seed 43 / 44
DEFAULTS = {"train_examples": 10_000, "epochs": 50, "seed": 42}


def split_xfit(method: str) -> tuple[str, Optional[int]]:
    """P17: 'cte_depth_top3_xs43' -> ('cte_depth_top3', 43): the same method with cross-fitting fold seed 43. The seed
    is passed to multabench/ours/cte.py through OURS_XFIT_SEED, which must be set before that module is imported."""
    import re
    m = re.fullmatch(r"(.+)_xs(\d+)", method or "")
    return (m.group(1), int(m.group(2))) if m else (method, None)


def set_xfit_seed(method: str) -> Optional[int]:
    _, xs = split_xfit(method)
    if xs is not None:
        os.environ["OURS_XFIT_SEED"] = str(xs)
    return xs
COMPANIONS = {"xfit_pred": ["frozen_xfit_pred"]}  # P3: written by the same job (shares the fold fine-tunes)
SUFFIX = {"epochs": "ep", "train_examples": "n", "seed": "s"}


def run_name(method: str, args: argparse.Namespace) -> str:
    parts = [f"{SUFFIX[k]}{getattr(args, k)}" for k in DEFAULTS if getattr(args, k) != DEFAULTS[k]]
    if getattr(args, "no_fast", False):
        parts.append("nofast")  # old full-model path: a different run, never mixed with fast-path results
    return method if not parts else f"{method}__{'_'.join(parts)}"


def existing_result_ok(path: str) -> bool:
    """True if a finished result exists AND was produced by the current engine version. A result from an older
    engine is kept (renamed to fold<k>.<engine>.json) and the job counts as not done, so it is re-run."""
    if not exists(path):
        return False
    from multabench.ours.variants import engine_version  # lazy: keeps this module importable without torch
    r = json.load(open(path))
    # top-level engine_version is in every result; tar.config (older results) as a fallback. catboost_native_text has
    # no tar block, so reading only tar.config made its finished results look stale (re-run on every restart).
    engine = r.get("engine_version") or ((r.get("tar") or {}).get("config") or {}).get("engine_version")
    ENGINE_VERSION = engine_version(r.get("method", "tar_repro"))
    if engine == ENGINE_VERSION:
        return True
    if same_config_as_current(r, engine, ENGINE_VERSION):
        return True
    old = path[:-len(".json")] + f".{str(engine).replace('/', '-')}.json"
    os.replace(path, old)
    print(f"  result made by engine {engine} (current {ENGINE_VERSION}) kept as {old}; job will re-run", flush=True)
    return False


def same_config_as_current(r: dict, engine: Optional[str], current: str) -> bool:
    """P6 amendment 2 (p6/2 -> p6/3) changed only the CatBoost text calcers of the native-text methods, to BoW only.
    A p6/2 native-text result that already used BoW only (regression; C > 10 classes under amendment 1) has exactly
    the p6/3 configuration, so it stays valid. NaiveBayes / BM25 results re-run."""
    return (r.get("method") in ("catboost_native_text", "cte_plus_native") and engine == "p6/2" and current == "p6/3"
            and ((r.get("native_text") or {}).get("feature_calcers")) == ["BoW"])


def timeout_marker(path: str) -> str:
    return path[:-len(".json")] + ".timeout.json"


def result_path(out: str, name: str, learner: str, dataset: str, fold: int) -> str:
    return join(out, name, learner, dataset, f"fold{fold}.json")


def git_info() -> dict:
    def _git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit, "dirty": dirty}


def reference_scores(dataset: str, model_name: str, fold: int) -> dict:
    """Upstream precomputed Joint Frozen / Joint TAR scores for the same (dataset, learner, fold), if any."""
    import pandas as pd
    path = join(REPO_ROOT, "multabench", "leaderboard", "results", "text", f"{dataset}.csv")
    if not exists(path):
        return {}
    df = pd.read_csv(path)
    df = df[(df["model"] == model_name) & (df["fold"] == fold)]
    names = {"all": "joint_frozen", "ft": "joint_tar", "no_text": "unimodal_structured", "text_only": "unimodal_text"}
    return {names[s]: float(g["test_score"].iloc[0]) for s, g in df.groupby("multimodal_state") if s in names}


@contextlib.contextmanager
def dataset_lock(name: str):
    """Exclusive local lock around a dataset's download + load. Parallel jobs of the same dataset otherwise download
    into the same cache at once, and a job can read a half-written file (E35: NaN labels in upstream's split)."""
    import fcntl
    root = join(os.path.expanduser("~"), ".cache", "ours_locks")
    os.makedirs(root, exist_ok=True)
    with open(join(root, f"dataset_{name}.lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def fit_learner(args: argparse.Namespace):
    """Load the dataset, make upstream's split for args.fold, configure our learner and fit it (shared by run_one and
    scripts/train.py). Returns (model, learner_cls, x_train, y_train, x_test, y_test, t0)."""
    from tabstar.preprocessing.splits import split_to_test
    from tabstar.training.devices import get_device

    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.constants import DEVICE
    from multabench.datasets.downloading import download_dataset
    from multabench.datasets.multimodal import MultimodalState
    from multabench.datasets.utils import dataset_from_name
    from multabench.ours.learners import make_learner
    from multabench.ours.tar_engine import TarConfig, set_seed

    t0 = time.time()
    set_seed(args.seed)
    # One device per process: tabstar's get_device() refuses a GPU that already holds memory, which is our own after
    # the first method of a job (P3 companions: xfit_pred -> frozen_xfit_pred).
    device = getattr(args, "_device", None) or get_device(device=DEVICE)
    args._device = device
    dataset_id = dataset_from_name(name=args.dataset)
    # Same loading / subsampling / test split as upstream evaluate_on_loaded_dataset with multimodal_state=ft.
    with dataset_lock(args.dataset):  # parallel jobs of one dataset: one downloads, the others wait (C8: NaN labels)
        dataset = download_dataset(dataset_id=dataset_id, multimodal_state=MultimodalState.ALL, target_override=None)
    x, y = subsample_dataset(x=dataset.x, y=dataset.y, is_cls=dataset.is_cls, train_examples=args.train_examples, fold=args.fold)
    x_train, x_test, y_train, y_test = split_to_test(x=x, y=y, is_cls=dataset.is_cls, fold=args.fold,
                                                     train_examples=args.train_examples)
    learner_cls = make_learner(args.learner)
    model = learner_cls(problem_type=dataset.task_type, device=device, verbose=False, dataset=dataset_id,
                        image_folder=dataset.image_folder, tune_e5=False, pca_components=30, no_pca=False)
    cfg = TarConfig(epochs=args.epochs, seed=args.seed, fast=not args.no_fast, micro_batch_size=args.micro_batch_size,
                    eval_batch_size=256 if device.type == "cuda" else 16,  # eval batch size: speed only
                    lower_cache_dir=args.lower_cache_dir)
    model.configure_ours(variant=split_xfit(args.method)[0], cfg=cfg, cache_dir=args.cache_dir, dataset=args.dataset,
                         fold=args.fold,
                         test=(x_test, y_test))
    model.fit(x_train, y_train)
    return model, learner_cls, x_train, y_train, x_test, y_test, t0


def run_one(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch

    from multabench.ours.variants import engine_version
    from multabench.utils.hardware import get_hardware_dict

    model, learner_cls, x_train, y_train, x_test, y_test, t0 = fit_learner(args)
    from multabench.ours import cte
    xs = split_xfit(args.method)[1]
    if xs is not None and cte.CTE_CONFIG.seed != xs:  # the seed must have reached every fold assignment
        raise RuntimeError(f"{args.method}: cross-fitting seed is {cte.CTE_CONFIG.seed}, expected {xs}")
    device = args._device
    t_fit = time.time() - t0
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    runtime = time.time() - t0

    stack = getattr(model, "stack_result", None)
    tar = stack.base_result if stack is not None else model.tar_result  # P3: TAR diagnostics of the base fine-tune
    hq = tar.diagnostics["head_quality"] if tar else {}
    ref = reference_scores(args.dataset, learner_cls.MODEL_NAME, args.fold)
    score = float(metrics.score)
    out = {
        "method": args.method, "run_name": run_name(args.method, args), "engine_version": engine_version(args.method),
        "xfit_seed": cte.CTE_CONFIG.seed,
        "tar_path": None if not tar else ("fast" if tar.diagnostics["train"].get("fast_path") else "full_model"),
        "learner": args.learner,
        "learner_model_name": learner_cls.MODEL_NAME, "dataset": args.dataset, "fold": args.fold,
        "task": args.dataset[:3], "score": score, "metrics": json.loads(json.dumps(asdict(metrics), default=float)),
        "reference": ref, "diff_vs_csv_joint_tar": (score - ref["joint_tar"]) if "joint_tar" in ref else None,
        "runtime_s": {"total": round(runtime, 1), "fit": round(t_fit, 1), "predict": round(runtime - t_fit, 1),
                      "tar_lower_cache": None if (not tar or tar.cache_hit) else tar.diagnostics.get("seconds_lower_cache"),
                      "tar_train": None if (not tar or tar.cache_hit) else tar.diagnostics["seconds_train"],
                      "tar_encode": None if (not tar or tar.cache_hit) else tar.diagnostics["seconds_encode"]},
        "tar": None if not tar else {
            "cache_key": tar.cache_key, "cache_hit": tar.cache_hit, "config": tar.config, "text_cols": tar.text_cols,
            "head_quality": hq, "n_unique_texts": tar.diagnostics["n_unique_texts"],
            "train": {k: tar.diagnostics["train"][k] for k in ("epochs_run", "best_epoch", "best_val_loss", "history",
                                                                "n_train_cells", "n_val_cells", "mean_train_tokens", "micro_batch_size_used",
                                                                "fast_path", "bucketed_batches", "peak_gpu_gb",
                                                                "truncated_frac", "fp16")},
            "split_sizes": {k: int(len(v)) for k, v in tar.splits.items()},
            **{k: tar.diagnostics[k] for k in ("struct_only_quality", "combined_quality", "lift", "offset_info",
                                               "feature_info", "offset_delegated", "seconds_struct")
               if k in tar.diagnostics},
        },
        # Leakage diagnostic: head quality on rows E5 was trained on minus on unseen test rows.
        "leakage_gap_head_train_minus_test": (hq["e5_train"]["score"] - hq["test"]["score"])
        if hq and hq.get("e5_train", {}).get("score") is not None and hq.get("test", {}).get("score") is not None else None,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)), "train_examples": args.train_examples,
        "config": {"epochs": args.epochs, "seed": args.seed, "micro_batch_size": args.micro_batch_size,
                   "fast": not args.no_fast,
                   "pca_components": 30, "cache_dir": args.cache_dir},
        "git": git_info(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:],
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",  # P4: same type per run
        "hardware": json.loads(json.dumps(get_hardware_dict(device), default=str)),
        "learner_threads": int(os.environ.get("OURS_LEARNER_THREADS", "0") or 0) or None,  # None = learner default
        "torch": torch.__version__, "numpy": np.__version__,
    }
    if stack is not None:  # P3: engine version of this method + stacking diagnostics
        if out["tar"] is None:  # frozen_xfit_pred without a cached base fine-tune (diagnostics skipped, P4)
            out["tar"] = {"cache_hit": None, "head_quality": {}, "config": {}, "base_skipped": True}
        out["tar"]["config"] = {**out["tar"]["config"], "engine_version": engine_version(args.method),
                                "method": args.method, "base": "tar_repro" if stack.base_result else None}
        out["stacking"] = {**stack.diagnostics, "catboost_importance": model.stack_feature_importance()}
    if hasattr(model, "native_text_info") and model.native_text_info() is not None:  # P6 native text variants
        out["native_text"] = model.native_text_info()
    return out


def prune_lower_cache(root: str, max_gb: float) -> None:
    """Keep the local frozen-layer cache under max_gb by deleting the oldest entries (context variants make big ones).
    Only disk space / rebuild time is affected, never results; avoids the engine's fallback to the full-model path."""
    import shutil
    if not os.path.isdir(root):
        return
    entries = []
    for name in os.listdir(root):
        d = join(root, name)
        if os.path.isdir(d) and not name.startswith(".tmp_"):
            size = sum(os.path.getsize(join(d, f)) for f in os.listdir(d))
            entries.append((os.path.getmtime(d), size, d))
    total = sum(e[1] for e in entries)
    for _, size, d in sorted(entries):
        if total <= max_gb * 1e9:
            break
        shutil.rmtree(d, ignore_errors=True)
        total -= size
        print(f"  pruned lower-layer cache {basename(d)} ({size / 1e9:.1f} GB)", flush=True)


def write_json(path: str, obj: dict) -> None:
    os.makedirs(dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written result that would be "skipped"


def read_jobs(path: str, default_seed: int = DEFAULTS["seed"]) -> list[tuple[str, str, str, int, int]]:
    """Lines "method learner dataset fold [seed]"; a missing seed means default_seed (the --seed flag, default 42)."""
    jobs = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in (4, 5):
            raise ValueError(f"{path}: expected 'method learner dataset fold [seed]', got {line!r}")
        method, learner, dataset, fold = parts[:4]
        jobs.append((method, learner, dataset, int(fold), int(parts[4]) if len(parts) == 5 else default_seed))
    return jobs


def job_label(method: str, learner: str, dataset: str, fold: int, seed: int = DEFAULTS["seed"]) -> str:
    return f"{method} {learner} {dataset} fold {fold}" + (f" seed {seed}" if seed != DEFAULTS["seed"] else "")


def job_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    """args with this job's seed (so run_name / result_path pick the seed-specific run, e.g. tar_repro__s43)."""
    return argparse.Namespace(**{**vars(args), "seed": seed})


def _fmt_secs(s: float) -> str:
    return f"{s:.0f}s" if s < 120 else f"{s / 60:.1f} min"


def parallel_plan(args: argparse.Namespace, jobs: list) -> tuple[int, dict]:
    """(number of parallel jobs, extra env for each job). "auto" = CPU cores // 2 when every job is a CPU-bound CTE
    / P6 job, else 1 (TAR fine-tuning jobs share one GPU and must run one at a time). Each job's learner and BLAS get
    cores // n threads so the cores are not oversubscribed."""
    cores = os.cpu_count() or 1
    if args.parallel_jobs == "auto":
        cpu_bound = lambda m: m.startswith(("cte", "tf_", "p9_", "frozen0")) or m == "catboost_native_text"
        n = max(1, cores // 2) if jobs and all(cpu_bound(j[0]) for j in jobs) else 1
    else:
        n = max(1, int(args.parallel_jobs))
    if n == 1:
        return 1, {}
    t = str(max(1, cores // n))
    env = {"OURS_LEARNER_THREADS": t, "OMP_NUM_THREADS": t, "MKL_NUM_THREADS": t, "OPENBLAS_NUM_THREADS": t}
    if "CUDA_VISIBLE_DEVICES" not in os.environ and shutil.which("nvidia-smi") and \
            subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0:
        env["CUDA_VISIBLE_DEVICES"] = "0"  # tabstar's get_device() refuses a GPU another job already uses
    return n, env


def run_jobs(args: argparse.Namespace) -> int:
    jobs = read_jobs(args.jobs_file, default_seed=args.seed)
    if args.filter:  # every substring must appear in "method learner dataset fold k"
        jobs = [j for j in jobs if all(f in job_label(*j) for f in args.filter)]
        print(f"filter {args.filter}: {len(jobs)} matching jobs", flush=True)
    n_done = n_failed = n_timeout = 0
    common = ["--out", args.out, "--cache_dir", args.cache_dir, "--train_examples", str(args.train_examples),
              "--epochs", str(args.epochs), "--micro_batch_size", str(args.micro_batch_size)]
    common += ["--no_fast"] if args.no_fast else []
    common += ["--lower_cache_dir", args.lower_cache_dir] if args.lower_cache_dir else []
    todo = []
    for i, (method, learner, dataset, fold, seed) in enumerate(jobs, 1):
        name = run_name(method, job_args(args, seed))
        path = result_path(args.out, name, learner, dataset, fold)
        tag = f"[{i}/{len(jobs)}] {name} {learner} {dataset} fold{fold}"
        companion_paths = [result_path(args.out, run_name(m, job_args(args, seed)), learner, dataset, fold)
                           for m in COMPANIONS.get(method, [])]
        if not args.force and all(existing_result_ok(q) for q in [path] + companion_paths):
            print(f"{tag}: done already, skipping", flush=True)
            n_done += 1
            continue
        if exists(timeout_marker(path)) and not args.retry_timeouts:
            print(f"{tag}: timed out before ({timeout_marker(path)}), not retried -> missing", flush=True)
            n_timeout += 1
            continue
        todo.append((tag, name, path, method, learner, dataset, fold, seed))
    if args.max_jobs is not None and len(todo) > args.max_jobs:
        print(f"--max_jobs {args.max_jobs}: {len(todo) - args.max_jobs} job(s) not started", flush=True)
        todo = todo[:args.max_jobs]
    n_par, env_extra = parallel_plan(args, [j[3:] for j in todo])
    if n_par > 1:
        print(f"running {len(todo)} job(s), {n_par} in parallel; per-job env {env_extra}", flush=True)
    timeout_s = args.job_timeout_min * 60 if (n_par > 1 and args.job_timeout_min) else None
    if timeout_s:
        print(f"per-job wall-clock timeout {args.job_timeout_min} min (timed-out jobs are logged, not retried)", flush=True)
    lock = threading.Lock()
    errors_path = join(args.out, "_logs", "process_errors.jsonl")

    def log_error(job, status: str, returncode, seconds: float, log_path: str) -> None:
        tag, name, path, method, learner, dataset, fold, seed = job
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": status, "method": method, "learner": learner,
               "dataset": dataset, "fold": fold, "seed": seed, "returncode": returncode,
               "seconds": round(seconds, 1), "log": log_path}
        with lock:
            os.makedirs(dirname(errors_path), exist_ok=True)
            with open(errors_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    def say(msg: str) -> None:
        with lock:
            print(msg, flush=True)

    def run_job(job) -> str:
        tag, name, path, method, learner, dataset, fold, seed = job
        if n_par == 1:
            prune_lower_cache(args.lower_cache_dir or join(os.path.expanduser("~"), ".cache", "tar_lower_cache"),
                              args.max_lower_cache_gb)
        log_path = join(args.out, "_logs", name, learner, f"{dataset}_fold{fold}.log")
        os.makedirs(dirname(log_path), exist_ok=True)
        say(f"{tag}: running (log: {log_path})")
        t0 = time.time()
        cmd = [sys.executable, abspath(__file__), "--method", method, "--learner", learner, "--dataset", dataset,
               "--fold", str(fold), "--seed", str(seed), *common] + (["--force"] if args.force else [])
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env={**os.environ, **env_extra}, start_new_session=True)
            timed_out = threading.Event()

            def kill():  # the whole process group: the job and anything it started
                timed_out.set()
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            timer = threading.Timer(timeout_s, kill) if timeout_s else None
            if timer:
                timer.daemon = True
                timer.start()
            for line in proc.stdout:
                log.write(line)
                if args.stream or line.lstrip().startswith(("[TAR", "[STACK")) or "Score" in line or "Error" in line:
                    say(("    " if n_par == 1 else f"    {tag} ") + line.rstrip())
            proc.wait()
            if timer:
                timer.cancel()
        if timed_out.is_set():
            rec = log_error(job, "timeout", proc.returncode, time.time() - t0, log_path)
            os.makedirs(dirname(path), exist_ok=True)  # a job killed early never created its result dir
            json.dump(rec, open(timeout_marker(path), "w"), indent=1)
            say(f"{tag}: TIMEOUT after {_fmt_secs(time.time() - t0)} (limit {args.job_timeout_min} min) -> missing, "
                f"not retried; logged in {errors_path}")
            return "timeout"
        if proc.returncode == 0 and exists(path):
            r = json.load(open(path))
            lb = r["reference"].get("joint_tar")
            lb_txt = (f" | leaderboard ft {lb:.4f} | diff {r['score'] - lb:+.4f}" if lb is not None
                      else " | leaderboard ft: n/a")
            say(f"{tag}: DONE score {r['score']:.4f}{lb_txt} | job time {_fmt_secs(time.time() - t0)}")
            return "done"
        log_error(job, "failed", proc.returncode, time.time() - t0, log_path)
        tail = open(log_path).read().splitlines()[-15:]
        say(f"{tag}: FAILED (exit {proc.returncode}) — see {log_path}; last lines:\n"
            + "\n".join("    | " + t for t in tail))
        return "failed"

    if n_par == 1:
        status = [run_job(j) for j in todo]
    else:
        with ThreadPoolExecutor(max_workers=n_par) as pool:  # threads only wait on the job subprocesses
            status = list(pool.map(run_job, todo))
    n_done += status.count("done")
    n_failed += status.count("failed")
    n_timeout += status.count("timeout")
    print(f"jobs: {len(jobs)} | finished: {n_done} | failed: {n_failed} | timed out (missing): {n_timeout}", flush=True)
    return 1 if n_failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--learner", choices=("catboost", "lgbm", "tabm", "tabpfnv2", "tabpfnv2p5"))
    p.add_argument("--dataset")
    p.add_argument("--fold", type=int)
    p.add_argument("--jobs_file", help="run every job listed in this file instead of a single job")
    p.add_argument("--max_jobs", type=int, default=None, help="--jobs_file: start at most this many (unfinished) jobs")
    p.add_argument("--filter", action="append", default=[],
                   help='--jobs_file: keep jobs whose "method learner dataset fold k[ seed s]" contains this (repeat = AND)')
    p.add_argument("--out", required=True, help="results root (on Colab: a Google Drive folder)")
    p.add_argument("--cache_dir", default=None, help="TAR cache root (default: <out>/_cache)")
    p.add_argument("--train_examples", type=int, default=DEFAULTS["train_examples"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"], help="max fine-tuning epochs (upstream: 50)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="seed; in a jobs file a 5th column overrides it")
    p.add_argument("--micro_batch_size", type=int, default=256, help="lower if the GPU runs out of memory (same result)")
    p.add_argument("--no_fast", action="store_true",
                   help="old full-model TAR path (no lower-layer cache, random batches); results go to <method>__nofast")
    p.add_argument("--lower_cache_dir", default=None,
                   help="local-disk dir for the frozen-layer cache (default ~/.cache/tar_lower_cache; not Drive)")
    p.add_argument("--max_lower_cache_gb", type=float, default=40.0,
                   help="--jobs_file: prune the oldest frozen-layer caches above this size before each job")
    p.add_argument("--force", action="store_true", help="re-run even if the result JSON exists")
    p.add_argument("--stream", action="store_true", help="--jobs_file: echo every log line, not just progress")
    p.add_argument("--job_timeout_min", type=float, default=30.0,
                   help="--jobs_file, parallel mode: kill a job after this many minutes (0 = none); logged as timeout in "
                        "<out>/_logs/process_errors.jsonl, not retried, reported as missing (P6 amendment 2)")
    p.add_argument("--retry_timeouts", action="store_true", help="--jobs_file: re-run jobs that timed out before")
    p.add_argument("--parallel_jobs", default="auto",
                   help='--jobs_file: jobs run at once ("auto" = CPU cores // 2 if every job is CTE, else 1)')
    args = p.parse_args()
    set_xfit_seed(args.method)  # P17: before any multabench.ours import (the jobs runner passes it to each job)
    args.cache_dir = args.cache_dir or join(args.out, "_cache")
    if args.jobs_file:
        return run_jobs(args)
    if None in (args.method, args.learner, args.dataset, args.fold):
        p.error("give --method --learner --dataset --fold, or --jobs_file")
    device = None
    for method in [args.method] + COMPANIONS.get(args.method, []):
        margs = argparse.Namespace(**{**vars(args), "method": method, "_device": device})
        path = result_path(args.out, run_name(method, margs), args.learner, args.dataset, args.fold)
        if not args.force and existing_result_ok(path):
            print(f"{path} exists, skipping")
            continue
        result = run_one(margs)
        device = margs._device  # reuse it for the companion
        write_json(path, result)
        lb = result["reference"].get("joint_tar")
        print(f"Score {result['score']:.4f} | leaderboard ft (same dataset/learner/fold) "
              f"{'n/a' if lb is None else f'{lb:.4f}'} | wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
