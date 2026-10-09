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
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from os.path import abspath, dirname, exists, join

REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # works with plain `uv sync` (no .pth file needed)
os.environ.setdefault("WANDB_MODE", "disabled")

METHODS = ("tar_repro",)
DEFAULTS = {"train_examples": 10_000, "epochs": 50, "seed": 42}
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
    from multabench.ours.tar_engine import ENGINE_VERSION  # lazy: keeps this module importable without torch
    r = json.load(open(path))
    engine = ((r.get("tar") or {}).get("config") or {}).get("engine_version")
    if engine == ENGINE_VERSION:
        return True
    old = path[:-len(".json")] + f".{str(engine).replace('/', '-')}.json"
    os.replace(path, old)
    print(f"  result made by engine {engine} (current {ENGINE_VERSION}) kept as {old}; job will re-run", flush=True)
    return False


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


def run_one(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from tabstar.preprocessing.splits import split_to_test
    from tabstar.training.devices import get_device

    from multabench.baselines.preprocessing.sampling import subsample_dataset
    from multabench.constants import DEVICE
    from multabench.datasets.downloading import download_dataset
    from multabench.datasets.multimodal import MultimodalState
    from multabench.datasets.utils import dataset_from_name
    from multabench.ours.learners import make_learner
    from multabench.ours.tar_engine import TarConfig, set_seed
    from multabench.utils.hardware import get_hardware_dict

    t0 = time.time()
    set_seed(args.seed)
    device = get_device(device=DEVICE)
    dataset_id = dataset_from_name(name=args.dataset)
    # Same loading / subsampling / test split as upstream evaluate_on_loaded_dataset with multimodal_state=ft.
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
    model.configure_ours(variant=args.method, cfg=cfg, cache_dir=args.cache_dir, dataset=args.dataset, fold=args.fold,
                         test=(x_test, y_test))
    model.fit(x_train, y_train)
    t_fit = time.time() - t0
    metrics = model.score_all_metrics(X=x_test, y=y_test)
    runtime = time.time() - t0

    tar = model.tar_result
    hq = tar.diagnostics["head_quality"] if tar else {}
    ref = reference_scores(args.dataset, learner_cls.MODEL_NAME, args.fold)
    score = float(metrics.score)
    return {
        "method": args.method, "run_name": run_name(args.method, args), "learner": args.learner,
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
        },
        # Leakage diagnostic: head quality on rows E5 was trained on minus on unseen test rows.
        "leakage_gap_head_train_minus_test": (hq["e5_train"]["score"] - hq["test"]["score"])
        if hq and hq.get("e5_train", {}).get("score") is not None and hq.get("test", {}).get("score") is not None else None,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)), "train_examples": args.train_examples,
        "config": {"epochs": args.epochs, "seed": args.seed, "micro_batch_size": args.micro_batch_size,
                   "fast": not args.no_fast,
                   "pca_components": 30, "cache_dir": args.cache_dir},
        "git": git_info(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:],
        "hardware": json.loads(json.dumps(get_hardware_dict(device), default=str)),
        "torch": torch.__version__, "numpy": np.__version__,
    }


def write_json(path: str, obj: dict) -> None:
    os.makedirs(dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written result that would be "skipped"


def read_jobs(path: str) -> list[tuple[str, str, str, int]]:
    jobs = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            method, learner, dataset, fold = line.split()
            jobs.append((method, learner, dataset, int(fold)))
    return jobs


def job_label(method: str, learner: str, dataset: str, fold: int) -> str:
    return f"{method} {learner} {dataset} fold {fold}"


def _fmt_secs(s: float) -> str:
    return f"{s:.0f}s" if s < 120 else f"{s / 60:.1f} min"


def run_jobs(args: argparse.Namespace) -> int:
    jobs = read_jobs(args.jobs_file)
    if args.filter:  # every substring must appear in "method learner dataset fold k"
        jobs = [j for j in jobs if all(f in job_label(*j) for f in args.filter)]
        print(f"filter {args.filter}: {len(jobs)} matching jobs", flush=True)
    n_done = n_failed = n_run = 0
    common = ["--out", args.out, "--cache_dir", args.cache_dir, "--train_examples", str(args.train_examples),
              "--epochs", str(args.epochs), "--seed", str(args.seed), "--micro_batch_size", str(args.micro_batch_size)]
    common += ["--no_fast"] if args.no_fast else []
    common += ["--lower_cache_dir", args.lower_cache_dir] if args.lower_cache_dir else []
    for i, (method, learner, dataset, fold) in enumerate(jobs, 1):
        name = run_name(method, args)
        path = result_path(args.out, name, learner, dataset, fold)
        tag = f"[{i}/{len(jobs)}] {name} {learner} {dataset} fold{fold}"
        if not args.force and existing_result_ok(path):
            print(f"{tag}: done already, skipping", flush=True)
            n_done += 1
            continue
        if args.max_jobs is not None and n_run >= args.max_jobs:
            print(f"--max_jobs {args.max_jobs} reached; {len(jobs) - i + 1} job(s) not started", flush=True)
            break
        n_run += 1
        log_path = join(args.out, "_logs", name, learner, f"{dataset}_fold{fold}.log")
        os.makedirs(dirname(log_path), exist_ok=True)
        print(f"{tag}: running (log: {log_path})", flush=True)
        t0 = time.time()
        cmd = [sys.executable, abspath(__file__), "--method", method, "--learner", learner, "--dataset", dataset,
               "--fold", str(fold), *common] + (["--force"] if args.force else [])
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                log.write(line)
                if args.stream or "[TAR]" in line or "Score" in line or "Error" in line:
                    print("    " + line.rstrip(), flush=True)
            proc.wait()
        if proc.returncode == 0 and exists(path):
            r = json.load(open(path))
            lb = r["reference"].get("joint_tar")
            lb_txt = (f" | leaderboard ft {lb:.4f} | diff {r['score'] - lb:+.4f}" if lb is not None
                      else " | leaderboard ft: n/a")
            print(f"{tag}: DONE score {r['score']:.4f}{lb_txt} | job time {_fmt_secs(time.time() - t0)}", flush=True)
            n_done += 1
        else:
            tail = open(log_path).read().splitlines()[-15:]
            print(f"{tag}: FAILED (exit {proc.returncode}) — see {log_path}; last lines:", flush=True)
            print("\n".join("    | " + t for t in tail), flush=True)
            n_failed += 1
    print(f"jobs: {len(jobs)} | finished: {n_done} | failed: {n_failed}", flush=True)
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
                   help='--jobs_file: keep jobs whose "method learner dataset fold k" contains this (repeat = AND)')
    p.add_argument("--out", required=True, help="results root (on Colab: a Google Drive folder)")
    p.add_argument("--cache_dir", default=None, help="TAR cache root (default: <out>/_cache)")
    p.add_argument("--train_examples", type=int, default=DEFAULTS["train_examples"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"], help="max fine-tuning epochs (upstream: 50)")
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--micro_batch_size", type=int, default=256, help="lower if the GPU runs out of memory (same result)")
    p.add_argument("--no_fast", action="store_true",
                   help="old full-model TAR path (no lower-layer cache, random batches); results go to <method>__nofast")
    p.add_argument("--lower_cache_dir", default=None,
                   help="local-disk dir for the frozen-layer cache (default ~/.cache/tar_lower_cache; not Drive)")
    p.add_argument("--force", action="store_true", help="re-run even if the result JSON exists")
    p.add_argument("--stream", action="store_true", help="--jobs_file: echo every log line, not just progress")
    args = p.parse_args()
    args.cache_dir = args.cache_dir or join(args.out, "_cache")
    if args.jobs_file:
        return run_jobs(args)
    if None in (args.method, args.learner, args.dataset, args.fold):
        p.error("give --method --learner --dataset --fold, or --jobs_file")
    path = result_path(args.out, run_name(args.method, args), args.learner, args.dataset, args.fold)
    if not args.force and existing_result_ok(path):
        print(f"{path} exists, skipping")
        return 0
    result = run_one(args)
    write_json(path, result)
    lb = result["reference"].get("joint_tar")
    print(f"Score {result['score']:.4f} | leaderboard ft (same dataset/learner/fold) "
          f"{'n/a' if lb is None else f'{lb:.4f}'} | wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
