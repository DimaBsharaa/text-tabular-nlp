"""Verbalized structured context for the `context` variants.

context = "; ".join(top-8 features by importance as "name=1.23 (decile 4/10)" | "name=<raw, <= 30 chars>" |
"name=missing"); cell input = "passage: {context} | {col}: {val}". The context is capped at 80 tokens and the text
is truncated so that [CLS] + all + [SEP] stays <= 512 tokens. Deciles are computed on the E5 pool only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from multabench.ours import tar_engine as te
from multabench.ours.archive.feature_tokens import _is_missing

CONTEXT_MAX_TOKENS = 80
CAT_MAX_CHARS = 30
PASSAGE = "passage:"


class Verbalizer:
    def __init__(self, features: list[str], cat_cols: list[str], x_pool: DataFrame):
        self.features = list(features)
        self.kinds = {f: ("cat" if f in cat_cols else "num") for f in self.features}
        self.deciles = {}
        for f in self.features:
            if self.kinds[f] == "num":
                v = pd.to_numeric(x_pool[f], errors="coerce").dropna().to_numpy(dtype=float)
                self.deciles[f] = np.quantile(v, np.linspace(0.1, 0.9, 9)) if len(v) else np.zeros(9)

    def _one(self, f: str, v) -> str:
        if _is_missing(v):
            return f"{f}=missing"
        if self.kinds[f] == "cat":
            return f"{f}={str(v)[:CAT_MAX_CHARS]}"
        x = float(v)
        if np.isnan(x):
            return f"{f}=missing"
        d = int(np.searchsorted(self.deciles[f], x, side="right")) + 1  # 1..10
        return f"{f}={x:.3g} (decile {d}/10)"

    def context(self, row: pd.Series) -> str:
        return "; ".join(self._one(f, row[f]) for f in self.features)

    def contexts(self, x: DataFrame) -> list[str]:
        return [self.context(r) for _, r in x[self.features].iterrows()]


def context_ids(tokenizer, context: str, text_part: str, max_length: int) -> list[int]:
    """[CLS] passage: <context, <= 80 tokens> | <text, truncated> [SEP], total <= max_length."""
    enc = lambda s: tokenizer(s, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    p_ids, c_ids = enc(PASSAGE), enc(context)[:CONTEXT_MAX_TOKENS]
    t_ids = enc(f"| {text_part}")  # truncated at max_length first (it is cut further below); same leading tokens
    room = max_length - 2 - len(p_ids) - len(c_ids)
    return [tokenizer.cls_token_id] + p_ids + c_ids + t_ids[:max(room, 0)] + [tokenizer.sep_token_id]


def full_string(context: str, text_part: str) -> str:
    return f"{PASSAGE} {context} | {text_part}"


class ContextTokenStore(te.TokenStore):
    """TokenStore over context strings with the capped token construction above. A sentinel string carrying the
    caps is appended, so the lower-layer cache key changes if the caps ever change."""

    def __init__(self, contexts: list[str], text_parts: list[str], tokenizer, max_length: int):
        strings, ids = [], []
        seen = {}
        for c, t in zip(contexts, text_parts):
            s = full_string(c, t)
            if s not in seen:
                seen[s] = len(strings)
                strings.append(s)
                ids.append(context_ids(tokenizer, c, t, max_length))
        sentinel = f"__context_caps__ ctx={CONTEXT_MAX_TOKENS} cat={CAT_MAX_CHARS} len={max_length}"
        strings.append(sentinel)
        ids.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
        self.strings, self.index, self.ids = strings, seen | {sentinel: len(strings) - 1}, ids
        self.lengths = np.array([len(i) for i in ids], dtype=np.int64)
        self.pad_id = tokenizer.pad_token_id

    def uids_for(self, contexts: list[str], text_parts: list[str]) -> np.ndarray:
        return np.array([self.index[full_string(c, t)] for c, t in zip(contexts, text_parts)], dtype=np.int64)
