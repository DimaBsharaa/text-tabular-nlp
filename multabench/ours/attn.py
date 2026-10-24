"""E52 (PROGRESS.md, development round): a cross-fitted ATTENTION PROBE on the frozen encoder's token states.

Motivation. Every other expert reads a text cell through a mean pool (or TF-IDF counts). When one token decides the
target (a slur for toxicity, one symptom for severity), the mean pool blurs it. Joint TAR can re-weight tokens because
it fine-tunes attention; a frozen encoder cannot. A small learned attention head on top of the frozen token states can:

  for each text column c:  s_t = v_c . tanh(W_c h_t)            (token scores; h_t = standardised token state)
                           a   = softmax_t(s_t) over real tokens  (padding masked)
                           z_c = sum_t a_t h_t
  output                 = Linear([z_c for all columns])          (C logits, or 1 value for regression)

The encoder is never trained. Token states come from one layer: the depth probes' best layer by training-row OOF score
(the same selection the method already makes). Training targets are the experts' own: class indices (cross-entropy) or
the rank-Gaussian y (MSE). Cross-fitting uses the experts' folds (`expert_folds`, same seed): each training row's
feature is predicted by the head trained on the other folds; early stopping uses an inner split of those folds only;
learner-validation and test rows get the mean of the fold heads. No validation or test label is used anywhere.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import os
from os.path import exists, join
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame

from multabench.ours import cte
from multabench.ours import tar_engine as te

ATTN_VERSION = "attn/1"


@dataclasses.dataclass(frozen=True)
class AttnSpec:
    hidden: int = 64
    max_tokens: int = 256        # token states kept per cell (the encoder still sees up to cfg.max_length)
    epochs: int = 40
    patience: int = 4
    lr: float = 2e-3
    weight_decay: float = 1e-2
    dropout: float = 0.1
    batch_size: int = 128
    inner_val: float = 0.1

    def key(self) -> str:
        return hashlib.sha1(repr((ATTN_VERSION, dataclasses.astuple(self))).encode()).hexdigest()[:10]


ATTN_SPEC = AttnSpec()


# ---------------------------------------------------------------------------------------- token states (label-free)

@torch.no_grad()
def encode_token_states(strings: list[str], model_name: str, max_length: int, layer: int, max_tokens: int,
                        device: torch.device, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Token states of hidden_states[layer] (1..L) of the frozen encoder, first max_tokens real tokens of each string.
    Returns (flat float16 (T, width), offsets (n + 1,)), strings in the given order."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa").to(device).eval()
    store = te.TokenStore(strings, tok, max_length)
    keep = np.minimum(store.lengths, max_tokens)
    offsets = np.concatenate([[0], np.cumsum(keep)]).astype(np.int64)
    flat = np.zeros((int(offsets[-1]), model.config.hidden_size), dtype=np.float16)
    order = np.argsort(store.lengths, kind="stable")
    for i in range(0, len(order), batch_size):
        rows = order[i:i + batch_size]
        ids, mask = store.batch(rows, device)
        with te._autocast(device, device.type == "cuda"):
            h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[layer]
        h = h.float().cpu().numpy()
        for j, r in enumerate(rows):
            flat[offsets[r]:offsets[r + 1]] = h[j, :keep[r]]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return flat, offsets


def token_states_cached(x_frames: list[DataFrame], text_cols: list[str], cfg: te.TarConfig, device: torch.device,
                        cache_root: Optional[str], dataset: str, layer: int, spec: AttnSpec = ATTN_SPEC):
    """{col: (flat, offsets, {row index: unique id})}; per-dataset cache of unique cell texts per layer (target-free,
    shared by folds, learners and dev splits), like depth.layer_states_cached."""
    from multabench.e5.constants import format_e5_passage
    out = {}
    for col in sorted(text_cols):
        rows = np.concatenate([np.asarray(f.index) for f in x_frames])
        raw = list(itertools.chain.from_iterable(te.encoding_texts(f, col) for f in x_frames))
        uniq, inverse = np.unique(np.array(raw, dtype=object), return_inverse=True)
        uniq = uniq.tolist()
        key = hashlib.sha1("\x00".join(uniq).encode()).hexdigest()[:12]
        path = join(cache_root, "attn_states", dataset, cfg.model_name.replace("/", "__"), f"L{layer:02d}_t{spec.max_tokens}_len{cfg.max_length}",
                    hashlib.sha1(col.encode()).hexdigest()[:12] + f"_{key}.npz") if cache_root else None
        with cte._store_lock(path):
            if path and exists(path):
                z = np.load(path)
                flat, offsets = z["flat"], z["offsets"]
            else:
                flat, offsets = encode_token_states([format_e5_passage(col, u) for u in uniq], cfg.model_name,
                                                    cfg.max_length, layer, spec.max_tokens, device)
                if path:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp = f"{path}.{os.getpid()}.tmp.npz"
                    np.savez(tmp, flat=flat, offsets=offsets)
                    os.replace(tmp, path)
        out[col] = (flat, offsets, dict(zip(rows.tolist(), inverse.tolist())))
    return out


# ---------------------------------------------------------------------------------------- the head

class AttnProbe(nn.Module):
    def __init__(self, n_cols: int, width: int, hidden: int, n_out: int, dropout: float):
        super().__init__()
        self.score = nn.ModuleList([nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))
                                    for _ in range(n_cols)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(n_cols * width, n_out)

    def forward(self, xs: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        zs = []
        for x, m, sc in zip(xs, masks, self.score):
            s = sc(x).squeeze(-1).masked_fill(m == 0, float("-inf"))
            a = torch.softmax(s, dim=1)
            zs.append((a.unsqueeze(-1) * x).sum(dim=1))
        return self.out(self.drop(torch.cat(zs, dim=1)))


class _Batcher:
    """Pads the token states of a set of rows per column, standardised with training-row token statistics."""

    def __init__(self, states, text_cols, mu_sd, device):
        self.states, self.cols, self.mu_sd, self.device = states, sorted(text_cols), mu_sd, device

    def __call__(self, rows: np.ndarray):
        xs, ms = [], []
        for col in self.cols:
            flat, off, uid = self.states[col]
            us = [uid[int(r)] for r in rows]
            lens = [max(1, int(off[u + 1] - off[u])) for u in us]
            L = max(lens)
            x = np.zeros((len(us), L, flat.shape[1]), dtype=np.float32)
            m = np.zeros((len(us), L), dtype=np.float32)
            for i, u in enumerate(us):
                seg = flat[off[u]:off[u + 1]]
                if len(seg):
                    x[i, :len(seg)] = seg
                    m[i, :len(seg)] = 1
                else:
                    m[i, 0] = 1  # empty cell: one zero token
            mu, sd = self.mu_sd[col]
            x = (x - mu) / sd * m[..., None]
            xs.append(torch.from_numpy(x).to(self.device))
            ms.append(torch.from_numpy(m).to(self.device))
        return xs, ms


def _token_stats(states, text_cols, rows: np.ndarray, max_rows: int = 4000, seed: int = 0):
    """Per-dimension mean / std of token states over (a sample of) the given training rows; label-free."""
    rng = np.random.RandomState(seed)
    rows = rows if len(rows) <= max_rows else rng.choice(rows, max_rows, replace=False)
    out = {}
    for col in sorted(text_cols):
        flat, off, uid = states[col]
        segs = [flat[off[uid[int(r)]]:off[uid[int(r)] + 1]] for r in rows]
        a = np.concatenate([s for s in segs if len(s)] or [np.zeros((1, flat.shape[1]), np.float16)]).astype(np.float32)
        out[col] = (a.mean(axis=0), a.std(axis=0) + 1e-6)
    return out


def _predict(model, batcher, rows, is_cls, bs=512) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            o = model(*batcher(rows[i:i + bs]))
            outs.append((torch.softmax(o, dim=1) if is_cls else o).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 1))


def _train_head(batcher, rows_tr, t_tr, rows_va, t_va, n_cols, width, n_out, is_cls, spec: AttnSpec, seed: int,
                device) -> AttnProbe:
    torch.manual_seed(seed)
    model = AttnProbe(n_cols, width, spec.hidden, n_out, spec.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    loss_fn = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
    to_t = (lambda a: torch.as_tensor(a, dtype=torch.long, device=device)) if is_cls else \
        (lambda a: torch.as_tensor(a, dtype=torch.float32, device=device).view(-1, 1))
    gen = np.random.RandomState(seed)
    best, best_state, bad = np.inf, None, 0
    for _ in range(spec.epochs):
        model.train()
        perm = gen.permutation(len(rows_tr))
        for i in range(0, len(perm), spec.batch_size):
            b = perm[i:i + spec.batch_size]
            loss = loss_fn(model(*batcher(rows_tr[b])), to_t(t_tr[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for i in range(0, len(rows_va), 512):
                b = slice(i, i + 512)
                vl += float(loss_fn(model(*batcher(rows_va[b])), to_t(t_va[b]))) * len(rows_va[b])
                n += len(rows_va[b])
        vl /= max(n, 1)
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= spec.patience:
                break
    model.load_state_dict(best_state)
    return model


def xfit_attention(states, text_cols: list[str], pool_rows: np.ndarray, extra_rows: Dict[str, np.ndarray],
                   y: np.ndarray, is_cls: bool, n_classes: int, c, device: torch.device, spec: AttnSpec = ATTN_SPEC,
                   keep_heads: bool = False):
    """Cross-fitted attention-probe features: pool rows out-of-fold, extras the mean of the fold heads. Returns
    (pool features, {split: features}, {"oof_metric": ...}) with cte.class_features' column budget; with keep_heads
    also the fold heads (`fit`, for the packaged model: new rows get the same mean of the fold heads)."""
    from multabench.ours.experts import expert_folds
    from sklearn.model_selection import train_test_split
    folds = expert_folds(len(y), y, is_cls, c.expert_config())
    top = cte.top_classes(y, n_classes, c) if is_cls else None
    width = next(iter(states.values()))[0].shape[1]
    n_out = n_classes if is_cls else 1
    oof = np.zeros((len(y), n_out))
    ext = {k: np.zeros((len(v), n_out)) for k, v in extra_rows.items()}
    heads = []
    for k, (tr, te_) in enumerate(folds):
        # targets and standardisation from this fold's training rows only
        t_all = y[tr] if is_cls else cte._targets(y[tr], False, 1, c)[:, 0]
        mu_sd = _token_stats(states, text_cols, pool_rows[tr], seed=c.seed + k)
        batcher = _Batcher(states, text_cols, mu_sd, device)
        strat = y[tr] if is_cls and np.bincount(y[tr]).min() >= 2 else None
        try:
            i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=spec.inner_val, random_state=c.seed + k,
                                          stratify=strat)
        except ValueError:
            i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=spec.inner_val, random_state=c.seed + k)
        model = _train_head(batcher, pool_rows[tr][i_tr], t_all[i_tr], pool_rows[tr][i_va], t_all[i_va],
                            len(text_cols), width, n_out, is_cls, spec, c.seed + k, device)
        oof[te_] = _predict(model, batcher, pool_rows[te_], is_cls)
        for name, rows in extra_rows.items():
            ext[name] += _predict(model, batcher, rows, is_cls) / len(folds)
        if keep_heads:
            heads.append({"state": {n: t.detach().cpu().numpy() for n, t in model.state_dict().items()},
                          "mu_sd": mu_sd})
    metric = cte.oof_metric(oof if is_cls else oof, y, is_cls, n_classes)
    feat = lambda P: cte.class_features(P, is_cls, n_classes, top)
    out = feat(oof), {k: feat(v) for k, v in ext.items()}, {"oof_metric": metric}
    if not keep_heads:
        return out
    fit = {"heads": heads, "n_cols": len(text_cols), "width": width, "n_out": n_out, "is_cls": is_cls,
           "n_classes": n_classes, "top": None if top is None else [int(t) for t in top],
           "spec": dataclasses.asdict(spec)}
    return (*out, fit)


def predict_heads(states, text_cols: list[str], rows: np.ndarray, fit: dict, device: torch.device) -> np.ndarray:
    """Features for new rows from saved fold heads: the mean of the heads' outputs (as for learner-val / test rows in
    xfit_attention), each head with its own training-row token standardisation; cte.class_features' budget."""
    spec = AttnSpec(**fit["spec"])
    P = np.zeros((len(rows), fit["n_out"]))  # float64 accumulation, exactly as for the extras in xfit_attention
    for h in fit["heads"]:
        model = AttnProbe(fit["n_cols"], fit["width"], spec.hidden, fit["n_out"], spec.dropout).to(device)
        model.load_state_dict({n: torch.as_tensor(a) for n, a in h["state"].items()})
        P += _predict(model, _Batcher(states, text_cols, h["mu_sd"], device), rows, fit["is_cls"]) / len(fit["heads"])
    top = None if fit["top"] is None else np.asarray(fit["top"])
    return cte.class_features(P, fit["is_cls"], fit["n_classes"], top)
