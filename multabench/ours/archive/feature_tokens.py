"""Structured feature tokens for the `prefix` variants.

token_j = W_name · E5_frozen(name_j) + value_enc_j(x_j), for the top-k structured features (by full-fit importance).
- numerics: piecewise-linear encoding (PLE) over <= 16 pool quantile bins -> Linear(16 -> 384);
- categoricals: one learned embedding per category seen >= MIN_COUNT times in the pool, others/unseen -> "other";
- missing (NaN or upstream's placeholder): a learned per-feature vector replaces the value encoding.
The k tokens are inserted at the input of the first trainable layer (on top of the cached frozen-layer states);
text tokens attend to them; pooling is over text tokens only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame

from multabench.ours import tar_engine as te

N_PLE_BINS = 16
MIN_COUNT = 3
TOP_K = 8


def _missing_value_placeholder() -> str:
    from tabstar.preprocessing.feat_types import MISSING_VALUE
    return MISSING_VALUE


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v == _missing_value_placeholder())


def ple_edges(values: np.ndarray, n_bins: int = N_PLE_BINS) -> np.ndarray:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return np.array([0.0, 0.0])
    return np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))


def ple_encode(x: np.ndarray, edges: np.ndarray, n_bins: int = N_PLE_BINS) -> np.ndarray:
    """Piecewise-linear encoding: component t is 0 below bin t, 1 above it, linear inside. NaN -> 0 (masked later)."""
    out = np.zeros((len(x), n_bins), dtype=np.float32)
    for t in range(min(len(edges) - 1, n_bins)):
        lo, hi = edges[t], edges[t + 1]
        out[:, t] = np.clip((x - lo) / (hi - lo), 0.0, 1.0) if hi > lo else (x >= hi).astype(np.float32)
    out[np.isnan(x)] = 0.0
    return out


class FeatureSpec:
    """Pool-fitted encoding of one structured feature (no learned parameters here)."""

    def __init__(self, name: str, kind: str, pool_values: pd.Series):
        self.name, self.kind = name, kind
        if kind == "num":
            self.edges = ple_edges(pd.to_numeric(pool_values, errors="coerce").to_numpy(dtype=float))
        else:
            counts = pool_values.astype(str)[~pool_values.map(_is_missing)].value_counts()
            self.vocab = {c: i + 1 for i, c in enumerate(sorted(counts[counts >= MIN_COUNT].index))}  # 0 = other

    def encode(self, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """(numeric PLE (n, 16) or category ids (n,), missing mask (n,))."""
        missing = values.map(_is_missing).to_numpy()
        if self.kind == "num":
            x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            return ple_encode(x, self.edges), missing | np.isnan(x)
        ids = np.array([self.vocab.get(str(v), 0) for v in values], dtype=np.int64)
        return ids, missing


class RowFeatures:
    """Encoded structured values for every row the model will see (pool + extras), indexed by row position."""

    def __init__(self, specs: list[FeatureSpec], x: DataFrame):
        self.index = pd.Index(x.index)
        self.num, self.cat, self.missing = {}, {}, {}
        for s in specs:
            enc, miss = s.encode(x[s.name])
            (self.num if s.kind == "num" else self.cat)[s.name] = torch.as_tensor(enc)
            self.missing[s.name] = torch.as_tensor(miss, dtype=torch.bool)

    def positions(self, rows) -> np.ndarray:
        pos = self.index.get_indexer(np.asarray(rows))
        assert (pos >= 0).all()
        return pos


class FeatureTokenizer(nn.Module):
    """Learned part of the feature tokens; trained with the head (same optimizer, lr and early stopping)."""

    def __init__(self, specs: list[FeatureSpec], name_vecs: torch.Tensor, d: int):
        super().__init__()
        self.specs = specs
        self.register_buffer("name_vecs", name_vecs.float())  # frozen E5 representation of the column names
        self.w_name = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.w_name.weight.copy_(torch.eye(d))
        self.num_enc = nn.ModuleDict()
        self.cat_emb = nn.ModuleDict()
        self.missing = nn.ParameterDict()
        for i, s in enumerate(specs):
            key = f"f{i}"
            if s.kind == "num":
                lin = nn.Linear(N_PLE_BINS, d)
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)
                self.num_enc[key] = lin
            else:
                emb = nn.Embedding(len(s.vocab) + 1, d)
                nn.init.normal_(emb.weight, std=0.02)
                self.cat_emb[key] = emb
            self.missing[key] = nn.Parameter(torch.randn(d) * 0.02)

    def forward(self, feats: RowFeatures, pos: np.ndarray, device: torch.device) -> torch.Tensor:
        """(B, k, d) tokens for the rows at positions `pos`."""
        p = torch.as_tensor(pos, dtype=torch.long)
        names = self.w_name(self.name_vecs)  # (k, d)
        toks = []
        for i, s in enumerate(self.specs):
            key = f"f{i}"
            if s.kind == "num":
                val = self.num_enc[key](feats.num[s.name][p].to(device))
            else:
                val = self.cat_emb[key](feats.cat[s.name][p].to(device))
            miss = feats.missing[s.name][p].to(device).unsqueeze(-1)
            val = torch.where(miss, self.missing[key].expand_as(val), val)
            toks.append(names[i].unsqueeze(0) + val)
        return torch.stack(toks, dim=1)


@torch.no_grad()
def frozen_name_vectors(model: te.E5ForTuning, tokenizer, names: list[str], cfg: te.TarConfig,
                        device: torch.device) -> torch.Tensor:
    """Spec 7.8: masked mean of the frozen embeddings + lower layers for "passage: {name}" (layer-9 input space)."""
    bert, k = te._bert(model), te.n_frozen_layers(cfg)
    tokens = te.TokenStore([f"passage: {n}" for n in names], tokenizer, cfg.max_length)
    model.eval()
    ids, mask = tokens.batch(np.arange(len(names)), device)
    h = te._run_layers(bert.encoder.layer[:k], bert.embeddings(input_ids=ids), mask).float()
    m = mask.unsqueeze(-1).float()
    return ((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu()


class PrefixSource:
    """Batches for the prefix variants: cached frozen-layer states of each cell's text + its row's feature tokens,
    through the trainable layers. `ids` index a cell table (string uid + row position per cell)."""
    fast = True

    def __init__(self, model: te.E5ForTuning, cache: te.LowerCache, cfg: te.TarConfig, cell_string_uid: np.ndarray,
                 cell_row_pos: np.ndarray, feats: RowFeatures):
        self.model = model
        self.layers = te._bert(model).encoder.layer[te.n_frozen_layers(cfg):]
        self.cache, self.feats = cache, feats
        self.cell_string_uid, self.cell_row_pos = cell_string_uid, cell_row_pos
        self.lengths = cache.lengths[cell_string_uid]  # per cell (for bucketing / length-sorted eval)
        self.features_visible = True  # False = text cannot attend to the feature tokens (test 5)

    def last_hidden(self, ids, device, use_fp16):
        h, mask = self.cache.batch(self.cell_string_uid[ids], device)
        ftok = self.model.feature_tokenizer(self.feats, self.cell_row_pos[ids], device)
        fmask = torch.full((len(ids), ftok.shape[1]), int(self.features_visible), dtype=mask.dtype, device=device)
        with te._autocast(device, use_fp16):
            out = te._run_layers(self.layers, torch.cat([ftok.float(), h], dim=1), torch.cat([fmask, mask], dim=1))
        return out[:, ftok.shape[1]:], mask  # text tokens only -> pooling over text only
