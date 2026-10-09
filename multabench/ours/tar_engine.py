"""TAR engine: a faithful re-implementation of upstream's target-aware E5 fine-tuning, with a disk cache.

Upstream path being reproduced (tag paper_version):
  baselines/preprocessing/text_embeddings.py::fit_text_encoders_tuned  +  e5/e5_finetune.py::finetune_e5_with_lora
  with the E5TrainArgs defaults that benchmark.py passes for `--multimodal_state ft`.

Kept identical to upstream:
  - one shared e5-small-v2 for all text columns; LoRA r=16, alpha=32, dropout 0.1, on the same 6 modules of the
    top 3 layers; linear head on the (non-normalized) masked mean-pooled last hidden state; cross-entropy loss
  - training cells: one per (row, text column), text "passage: {col}: {val}"; regression target → 20 quantile bins
  - 90/10 train/val split of the rows the learner hands us (upstream split_to_val, stratified on label/bin)
  - HF-Trainer semantics: AdamW lr 1e-4, weight decay 0.001 (not on biases / LayerNorm), linear decay to 0 over
    epochs * steps, grad-norm clip 1.0, batch 256, ≤ 50 epochs, eval every epoch, early stopping on val loss with
    patience 3, best epoch restored
  - final embeddings: "passage: {col}: {val}", masked mean pool + L2 normalize; PCA is done by the caller
Deliberate differences (speed / determinism only):
  - dynamic padding: truncate at 512, pad to the longest sequence in the batch (upstream pads everything to 512)
  - fp16 autocast + GradScaler on GPU (upstream trains in fp32)
  - our own fixed seed for init and shuffling (upstream's Trainer RNG stream can't be reproduced bit-for-bit)
  - inference deduplicates identical texts per column (same text → same embedding)

Besides embeddings, every fit records head quality (AUROC / R² of the fine-tuning head itself) on four row sets:
  e5_train (rows E5 was trained on), e5_val (rows used for early stopping), learner_val (the learner's own
  validation rows, never seen by E5) and test. The e5_train-vs-test gap is our leakage diagnostic: it measures how
  much more the fine-tuned encoder "knows" about its own training rows than about new rows.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from os.path import exists, join
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from multabench.baselines.training.metrics import calculate_metric
from multabench.e5.constants import E5_DIM, E5_NUM_LAYERS, E5_PASSAGE_PREFIX, E5_SMALL_V2, format_e5_passage
from multabench.e5.e5_finetune import E5ForTuning, _get_lora_target_modules, _mean_pool
from multabench.finetune.utils import encoder_finetune_loss
from multabench.preprocessing.discretize import discretize_numerical
from multabench.preprocessing.splits import split_to_val

ENGINE_VERSION = "tar_engine/1"
VARIANTS = {"tar_repro"}
SPLITS = ("e5_train", "e5_val", "learner_val", "test")


@dataclass
class TarConfig:
    """Hyperparameters of the fine-tune. Defaults = upstream E5TrainArgs + fit_text_encoders_tuned."""
    model_name: str = E5_SMALL_V2
    lora_rank: int = 16
    text_layers: int = 3
    lora_dropout: float = 0.1
    learning_rate: float = 1e-4
    epochs: int = 50
    patience: int = 3
    weight_decay: float = 0.001
    batch_size: int = 256
    max_grad_norm: float = 1.0
    max_length: int = 512
    n_bins: int = 20
    fp16: bool = True  # only takes effect on CUDA
    seed: int = 42
    # Not part of the result: only how the work is chunked.
    micro_batch_size: int = 256  # split each 256-batch into chunks of this size (gradient accumulation) if VRAM is short
    eval_batch_size: int = 256

    _NON_RESULT_FIELDS = ("micro_batch_size", "eval_batch_size")

    def result_dict(self, device: torch.device) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in self._NON_RESULT_FIELDS}
        d["fp16"] = bool(self.fp16 and device.type == "cuda")  # what actually ran
        d["engine_version"] = ENGINE_VERSION
        return d


def config_hash(cfg: TarConfig, variant: str, device: torch.device) -> str:
    payload = json.dumps({"variant": variant, **cfg.result_dict(device)}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _rows_hash(*indices: pd.Index) -> str:
    h = hashlib.sha1()
    for idx in indices:
        h.update(np.asarray(idx, dtype=np.int64).tobytes())
        h.update(b"|")
    return h.hexdigest()[:10]


def _text_hash(texts: list[str]) -> np.ndarray:
    return np.array([int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little") for t in texts],
                    dtype=np.uint64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------------- texts

def training_texts(x: DataFrame, text_cols: list[str]) -> list[str]:
    """Row-major (row, col) cells exactly as upstream builds them for fine-tuning (+ TextLabelDataset's prefix)."""
    out = []
    for idx in x.index:
        for col in text_cols:
            v = x.loc[idx, col]
            val = str(v).strip() if pd.notna(v) else ""
            out.append(E5_PASSAGE_PREFIX + (f"{col}: {val}".strip() or " "))
    return out


def encoding_texts(x: DataFrame, col: str) -> list[str]:
    """The strings upstream's transform_text_features passes to the encoder for one column (before the prefix)."""
    return x[col].astype(str).fillna("").tolist()


def prepare_e5_data(x_fit: DataFrame, y_fit: Series, text_cols: list[str], is_cls: bool, n_bins: int) -> dict:
    """Split + texts + labels for fine-tuning, built exactly like fit_text_encoders_tuned / finetune_e5_with_lora."""
    y_e5 = y_fit if is_cls else discretize_numerical(y_fit, n_bins=n_bins)
    x_tr, x_va, y_tr, y_va = split_to_val(x=x_fit, y=y_e5, is_cls=True)  # same call as upstream
    label_enc = LabelEncoder().fit(y_tr.astype(str).to_numpy())  # upstream fits it on the (repeated) train labels
    rep = len(text_cols)
    return {"x_tr": x_tr, "x_va": x_va, "y_tr": y_tr, "y_va": y_va, "classes": label_enc.classes_.tolist(),
            "train_texts": training_texts(x_tr, text_cols), "val_texts": training_texts(x_va, text_cols),
            "train_labels": np.repeat(label_enc.transform(y_tr.astype(str).to_numpy()), rep),
            "val_labels": np.repeat(label_enc.transform(y_va.astype(str).to_numpy()), rep)}


# --------------------------------------------------------------------------------------------- batching

class _Tokenized:
    """Pre-tokenized texts (truncated at max_length, unpadded) + dynamic-padding batch builder."""

    def __init__(self, texts: list[str], tokenizer, max_length: int):
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.ids = enc["input_ids"]
        self.lengths = np.array([len(i) for i in self.ids])
        self.pad_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.ids)

    def batch(self, rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = int(self.lengths[rows].max())
        input_ids = torch.full((len(rows), max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for i, r in enumerate(rows):
            n = self.lengths[r]
            input_ids[i, :n] = torch.tensor(self.ids[r])
            mask[i, :n] = 1
        return input_ids.to(device), mask.to(device)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


# --------------------------------------------------------------------------------------------- model

def build_model(cfg: TarConfig, d_output: int, device: torch.device) -> E5ForTuning:
    """Fresh e5 + LoRA + head, built in the same order and with the same settings as finetune_e5_with_lora."""
    backbone = AutoModel.from_pretrained(cfg.model_name)
    num_layers = E5_NUM_LAYERS[cfg.model_name]
    n_layers = min(max(1, cfg.text_layers), num_layers)
    lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_rank * 2,
                      target_modules=_get_lora_target_modules(list(range(num_layers))[-n_layers:]),
                      lora_dropout=cfg.lora_dropout, bias="none")
    backbone = get_peft_model(backbone, lora)
    model = E5ForTuning(backbone=backbone, d_output=d_output, d_model=E5_DIM[cfg.model_name])
    return model.to(device)


def _trainable_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def _optimizer(model: torch.nn.Module, cfg: TarConfig) -> torch.optim.Optimizer:
    """AdamW with HF-Trainer param groups: no weight decay on biases and LayerNorm weights."""
    no_decay_modules = {n for n, m in model.named_modules() if isinstance(m, torch.nn.LayerNorm)}
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        module_name = n.rsplit(".", 1)[0]
        (no_decay if n.endswith("bias") or module_name in no_decay_modules else decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)


@torch.no_grad()
def _val_loss(model: E5ForTuning, tok: _Tokenized, labels: torch.Tensor, cfg: TarConfig, device: torch.device,
              use_fp16: bool) -> float:
    model.eval()
    order = np.argsort(tok.lengths)
    total = 0.0
    for i in range(0, len(order), cfg.eval_batch_size):
        rows = order[i:i + cfg.eval_batch_size]
        ids, mask = tok.batch(rows, device)
        with _autocast(device, use_fp16):
            logits = model(input_ids=ids, attention_mask=mask)["logits"]
        total += float(encoder_finetune_loss(logits.float(), labels[rows].to(device))) * len(rows)
    return total / len(order)


def train_tar(cfg: TarConfig, train_texts: list[str], train_labels: np.ndarray, val_texts: list[str],
              val_labels: np.ndarray, d_output: int, tokenizer, device: torch.device) -> tuple[E5ForTuning, dict]:
    """LoRA fine-tune with early stopping on val loss; returns the model at its best epoch and a training log."""
    set_seed(cfg.seed)
    use_fp16 = bool(cfg.fp16 and device.type == "cuda")
    model = build_model(cfg, d_output=d_output, device=device)
    tr, va = _Tokenized(train_texts, tokenizer, cfg.max_length), _Tokenized(val_texts, tokenizer, cfg.max_length)
    y_tr, y_va = torch.tensor(train_labels, dtype=torch.long), torch.tensor(val_labels, dtype=torch.long)

    steps_per_epoch = (len(tr) + cfg.batch_size - 1) // cfg.batch_size
    optimizer = _optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=steps_per_epoch * cfg.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
    gen = torch.Generator().manual_seed(cfg.seed)
    micro = max(1, min(cfg.micro_batch_size, cfg.batch_size))

    history, best_loss, best_state, best_epoch, bad = [], float("inf"), None, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(tr), generator=gen).numpy()
        train_loss_sum = 0.0
        for b in range(steps_per_epoch):
            rows = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            while True:
                try:
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss = 0.0
                    for m in range(0, len(rows), micro):  # gradient accumulation == one 256-batch step
                        sub = rows[m:m + micro]
                        ids, mask = tr.batch(sub, device)
                        with _autocast(device, use_fp16):
                            logits = model(input_ids=ids, attention_mask=mask)["logits"]
                        loss = encoder_finetune_loss(logits.float(), y_tr[sub].to(device))
                        scaler.scale(loss * (len(sub) / len(rows))).backward()
                        batch_loss += float(loss) * len(sub)
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro == 1:
                        raise
                    micro = max(1, micro // 2)  # same math, smaller chunks; redo this batch from scratch
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    print(f"  [TAR] CUDA out of memory -> micro_batch_size {micro}", flush=True)
            train_loss_sum += batch_loss
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        val_loss = _val_loss(model, va, y_va, cfg, device, use_fp16)
        history.append({"epoch": epoch, "train_loss": train_loss_sum / len(tr), "val_loss": val_loss,
                        "seconds": round(time.time() - t0, 1)})
        print(f"  [TAR] epoch {epoch:>2}: train_loss {history[-1]['train_loss']:.4f}  val_loss {val_loss:.4f}  "
              f"({history[-1]['seconds']}s)", flush=True)
        if val_loss < best_loss:  # same rule as HF EarlyStoppingCallback (threshold 0)
            best_loss, best_state, best_epoch, bad = val_loss, _trainable_state(model), epoch, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state, strict=False)
    model.eval()
    log = {"history": history, "best_epoch": best_epoch, "best_val_loss": best_loss, "epochs_run": len(history),
           "n_train_cells": len(tr), "n_val_cells": len(va), "steps_per_epoch": steps_per_epoch, "fp16": use_fp16,
           "mean_train_tokens": float(tr.lengths.mean()), "micro_batch_size_used": micro, "truncated_frac": float((tr.lengths >= cfg.max_length).mean())}
    return model, log


@torch.no_grad()
def encode(model: E5ForTuning, tokenizer, texts: list[str], cfg: TarConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(L2-normalized mean-pooled embeddings as upstream's encode_texts_with_e5, head logits) for each text."""
    model.eval()
    use_fp16 = bool(cfg.fp16 and device.type == "cuda")
    tok = _Tokenized(texts, tokenizer, cfg.max_length)
    order = np.argsort(tok.lengths)  # length-sorted batches: much less padding, same results
    emb = np.zeros((len(texts), model.head.in_features), dtype=np.float32)
    logits = np.zeros((len(texts), model.head.out_features), dtype=np.float32)
    for i in range(0, len(order), cfg.eval_batch_size):
        rows = order[i:i + cfg.eval_batch_size]
        ids, mask = tok.batch(rows, device)
        with _autocast(device, use_fp16):
            hidden = model.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = _mean_pool(hidden.float(), mask, l2_normalize=False)
        emb[rows] = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
        logits[rows] = model.head(pooled).float().cpu().numpy()
    return emb, logits


# --------------------------------------------------------------------------------------------- result + cache

@dataclass
class ColumnOutputs:
    emb: np.ndarray        # (n_unique, 384) L2-normalized
    logits: np.ndarray     # (n_unique, d_output) head logits
    text_hash: np.ndarray  # (n_unique,) uint64 hash of the encoded string (sanity check at lookup)
    row_uid: Dict[int, int] = field(default_factory=dict)  # dataset row index -> unique id


@dataclass
class TarResult:
    variant: str
    cache_key: str
    cache_hit: bool
    config: dict
    text_cols: list[str]
    splits: Dict[str, np.ndarray]  # split name -> dataset row indices
    columns: Dict[str, ColumnOutputs]
    head: dict  # label classes, d_output, bin means (regression)
    diagnostics: dict  # head quality per split + training log + timings

    def row_embeddings(self, col: str, rows, texts: Optional[list[str]] = None) -> np.ndarray:
        c = self.columns[col]
        uid = np.array([c.row_uid[int(r)] for r in rows])
        if texts is not None:
            bad = np.nonzero(c.text_hash[uid] != _text_hash(texts))[0]
            if len(bad):
                raise ValueError(f"Cached TAR text mismatch for column {col!r} on {len(bad)} rows (e.g. row {rows[bad[0]]})")
        return c.emb[uid]

    def head_probs(self, rows) -> np.ndarray:
        """Row-level head prediction: softmax over classes/bins, averaged over the row's text cells."""
        probs = []
        for col in self.text_cols:
            c = self.columns[col]
            uid = np.array([c.row_uid[int(r)] for r in rows])
            probs.append(torch.softmax(torch.tensor(c.logits[uid]), dim=-1).numpy())
        return np.mean(probs, axis=0)


def _save(result: TarResult, model: Optional[E5ForTuning], path: str) -> None:
    tmp = tempfile.mkdtemp(prefix=".tmp_", dir=os.path.dirname(path))
    arrays = {}
    for i, col in enumerate(result.text_cols):
        c = result.columns[col]
        rows = np.array(list(c.row_uid.keys()), dtype=np.int64)
        arrays.update({f"c{i}_emb": c.emb, f"c{i}_logits": c.logits, f"c{i}_hash": c.text_hash,
                       f"c{i}_rows": rows, f"c{i}_uid": np.array([c.row_uid[int(r)] for r in rows], dtype=np.int64)})
    for s, idx in result.splits.items():
        arrays[f"split_{s}"] = np.asarray(idx, dtype=np.int64)
    np.savez_compressed(join(tmp, "outputs.npz"), **arrays)
    meta = {"variant": result.variant, "cache_key": result.cache_key, "config": result.config,
            "text_cols": result.text_cols, "head": result.head, "diagnostics": result.diagnostics}
    json.dump(meta, open(join(tmp, "meta.json"), "w"), indent=1)
    if model is not None:
        torch.save(_trainable_state(model), join(tmp, "trainable_state.pt"))  # LoRA + head, ~1.5 MB
    if exists(path):
        shutil.rmtree(path)
    os.replace(tmp, path)


def _load(path: str) -> TarResult:
    meta = json.load(open(join(path, "meta.json")))
    z = np.load(join(path, "outputs.npz"))
    columns = {}
    for i, col in enumerate(meta["text_cols"]):
        columns[col] = ColumnOutputs(emb=z[f"c{i}_emb"], logits=z[f"c{i}_logits"], text_hash=z[f"c{i}_hash"],
                                     row_uid=dict(zip(z[f"c{i}_rows"].tolist(), z[f"c{i}_uid"].tolist())))
    splits = {s: z[f"split_{s}"] for s in SPLITS if f"split_{s}" in z}
    return TarResult(variant=meta["variant"], cache_key=meta["cache_key"], cache_hit=True, config=meta["config"],
                     text_cols=meta["text_cols"], splits=splits, columns=columns, head=meta["head"],
                     diagnostics=meta["diagnostics"])


# --------------------------------------------------------------------------------------------- main entry

def _head_quality(result: TarResult, y_by_split: Dict[str, Series], is_cls: bool) -> dict:
    out = {}
    classes = result.head["classes"]
    for split, y in y_by_split.items():
        rows = np.asarray(y.index)
        probs = result.head_probs(rows)
        if is_cls:
            known = np.isin(y.astype(str).to_numpy(), classes)
            y_int = LabelEncoder().fit(classes).transform(y.astype(str).to_numpy()[known])
            if len(np.unique(y_int)) < 2:
                out[split] = {"score": None, "n": int(known.sum()), "note": "fewer than 2 classes"}
                continue
            m = calculate_metric(y_true=y_int, y_pred=probs[known], d_output=len(classes))
            out[split] = {"score": float(m.score), "n": int(known.sum()), "n_unseen_class": int((~known).sum())}
        else:
            pred = probs @ np.asarray(result.head["bin_means"])
            m = calculate_metric(y_true=pd.to_numeric(y).to_numpy(), y_pred=pred, d_output=1)
            out[split] = {"score": float(m.score), "n": int(len(y))}
    return out


def run_tar(x_fit: DataFrame, y_fit: Series, text_cols: list[str], is_cls: bool, extra: Dict[str, tuple[DataFrame, Series]],
            cfg: TarConfig, device: torch.device, cache_root: Optional[str], dataset: str, fold: int,
            variant: str = "tar_repro") -> TarResult:
    """Fine-tune (or load from cache) and return embeddings + head outputs for x_fit rows and every extra row set.

    x_fit / extras must already be through the learner's model-agnostic preprocessing (text columns as strings).
    extra: {"learner_val": (x, y), "test": (x, y)}; y of extras is used ONLY to report head quality, never to train.
    """
    assert variant in VARIANTS, f"unknown TAR variant {variant!r}"
    all_rows = pd.Index(np.concatenate([np.asarray(x_fit.index)] + [np.asarray(x.index) for x, _ in extra.values() if x is not None]))
    assert all_rows.is_unique, "row sets passed to run_tar must be disjoint with unique indices"
    text_cols = sorted(text_cols)
    extra = {k: v for k, v in extra.items() if v is not None}
    key = f"{dataset}/fold{fold}/{variant}/seed{cfg.seed}/{config_hash(cfg, variant, device)}_" \
          f"{_rows_hash(x_fit.index, *[x.index for x, _ in extra.values()])}"
    path = join(cache_root, key) if cache_root else None
    if path and exists(join(path, "meta.json")):
        print(f"  [TAR] cache hit: {key}", flush=True)
        return _load(path)

    t0 = time.time()
    d = prepare_e5_data(x_fit, y_fit, text_cols, is_cls, cfg.n_bins)
    x_tr, x_va, y_tr = d["x_tr"], d["x_va"], d["y_tr"]
    classes = d["classes"]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    print(f"  [TAR] fine-tuning {cfg.model_name} on {len(x_tr)} rows × {len(text_cols)} text cols = "
          f"{len(d['train_texts'])} cells (val {len(x_va)} rows), d_output={len(classes)}, device={device}", flush=True)
    model, train_log = train_tar(cfg, d["train_texts"], d["train_labels"], d["val_texts"], d["val_labels"],
                                 len(classes), tokenizer, device)
    t_train = time.time() - t0

    # Encode every row we will ever be asked about (fit rows + extras), deduplicated per column.
    t1 = time.time()
    frames = [x_fit] + [x for x, _ in extra.values()]
    columns = {}
    for col in text_cols:
        row_idx = np.concatenate([np.asarray(f.index) for f in frames])
        texts = list(itertools.chain.from_iterable(encoding_texts(f, col) for f in frames))
        uniq, inverse = np.unique(np.array(texts, dtype=object), return_inverse=True)
        uniq = uniq.tolist()
        emb, logits = encode(model, tokenizer, [format_e5_passage(col, t) for t in uniq], cfg, device)
        columns[col] = ColumnOutputs(emb=emb, logits=logits, text_hash=_text_hash(uniq),
                                     row_uid=dict(zip(row_idx.tolist(), inverse.tolist())))
    t_encode = time.time() - t1

    head = {"classes": classes, "d_output": len(classes)}
    if not is_cls:
        yb = pd.Series(y_tr.astype(str).to_numpy(), index=y_tr.index)
        means = pd.to_numeric(y_fit.loc[y_tr.index]).groupby(yb).mean()
        head["bin_means"] = [float(means[c]) for c in classes]
    splits = {"e5_train": np.asarray(x_tr.index), "e5_val": np.asarray(x_va.index)}
    splits.update({k: np.asarray(x.index) for k, (x, _) in extra.items()})
    result = TarResult(variant=variant, cache_key=key, cache_hit=False, config=cfg.result_dict(device),
                       text_cols=text_cols, splits=splits, columns=columns, head=head, diagnostics={})
    y_by_split = {"e5_train": y_fit.loc[x_tr.index], "e5_val": y_fit.loc[x_va.index]}
    y_by_split.update({k: y for k, (_, y) in extra.items()})
    result.diagnostics = {"head_quality": _head_quality(result, y_by_split, is_cls), "train": train_log,
                          "seconds_train": round(t_train, 1), "seconds_encode": round(t_encode, 1),
                          "n_unique_texts": {c: int(len(columns[c].emb)) for c in text_cols},
                          "device": str(device)}
    hq = result.diagnostics["head_quality"]
    print("  [TAR] head quality: " + ", ".join(f"{s}={v['score']:.4f}" for s, v in hq.items() if v["score"] is not None),
          flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _save(result, model, path)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
