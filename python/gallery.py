# gallery.py
from __future__ import annotations
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class DetectionGallery:
    """
    In-memory gallery for embeddings with metadata.
    - embeddings: numpy array shape (N, D) rows are L2-normalized
    - keys: parallel list of keys (usually track ids)
    - metas: parallel list of metadata dicts
    """

    def __init__(self, max_items: int = 5000):
        self.max_items = int(max_items)
        self.embeddings: Optional[np.ndarray] = None  # shape (N, D)
        self.keys: List[int] = []
        self.metas: List[Dict[str, Any]] = []

    def _ensure_shape_and_normalize(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype='float32')
        if emb.ndim == 1:
            emb = emb[None, :]
        # normalize rows
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-7
        emb = emb / norms
        return emb

    def _prune_if_needed(self):
        if self.embeddings is None:
            return
        n = len(self.keys)
        if n <= self.max_items:
            return
        # prune by oldest last_seen timestamp by default
        last_seen = [m.get('last_seen', 0) for m in self.metas]
        order = np.argsort(last_seen)  # ascending
        to_keep = order[-self.max_items:]
        self.embeddings = self.embeddings[to_keep]
        self.keys = [self.keys[i] for i in to_keep]
        self.metas = [self.metas[i] for i in to_keep]

    def add(self,
            key: int,
            embedding: np.ndarray,
            meta: Optional[Dict[str, Any]] = None,
            timestamp: Optional[float] = None,
            dedupe_threshold: float = 0.995,
            replace_similar: bool = False):
        """
        Add one embedding (or batch) with optional metadata.

        - key: usually track id
        - embedding: 1D or 2D numpy array (if 2D, will be appended row-wise)
        - meta: metadata dict
        - timestamp: unix time, if None uses now()
        - dedupe_threshold: if adding entry for the same key and similarity >= threshold,
          we do not append a duplicate entry (but we update last_seen in meta).
        - replace_similar: if True and similar entry exists for same key, we replace that row's embedding/meta.
        """
        ts = timestamp if timestamp is not None else time.time()
        emb = self._ensure_shape_and_normalize(np.asarray(embedding))
        num = emb.shape[0]

        if self.embeddings is None:
            self.embeddings = emb.copy()
            # replicate key/meta per-row
            for i in range(num):
                self.keys.append(int(key))
                m = dict(meta or {})
                m.setdefault('last_seen', ts)
                self.metas.append(m)
            self._prune_if_needed()
            return

        # dedup logic per-row: for same key, check similarity to existing rows with same key
        for i in range(num):
            row = emb[i : i + 1]  # shape (1, D)
            same_key_indices = [idx for idx, k in enumerate(self.keys) if k == int(key)]
            best_sim = -1.0
            best_idx = None
            if same_key_indices:
                # compute sims only against same-key rows
                existing = self.embeddings[same_key_indices]  # shape (M, D)
                sims = (existing @ row.T).squeeze()
                idx_rel = int(np.argmax(sims))
                best_sim = float(sims[idx_rel])
                best_idx = same_key_indices[idx_rel]

            if best_idx is not None and best_sim >= dedupe_threshold:
                # similar snapshot exists; update last_seen and optionally replace embedding/meta
                try:
                    self.metas[best_idx].update(dict(meta or {}))
                    self.metas[best_idx]['last_seen'] = ts
                    if replace_similar:
                        self.embeddings[best_idx] = row
                except Exception:
                    pass
                continue  # skip appending duplicate

            # append new row
            self.embeddings = np.vstack([self.embeddings, row])
            self.keys.append(int(key))
            m = dict(meta or {})
            m.setdefault('last_seen', ts)
            self.metas.append(m)

        self._prune_if_needed()

    def add_batch(self,
                  keys: List[int],
                  embeddings: np.ndarray,
                  metas: Optional[List[Dict[str, Any]]] = None,
                  timestamp: Optional[float] = None,
                  dedupe_threshold: float = 0.995):
        """
        Batch add convenience. keys length must match number of embeddings rows.
        """
        ts = timestamp if timestamp is not None else time.time()
        embeddings = self._ensure_shape_and_normalize(np.asarray(embeddings))
        nrows = embeddings.shape[0]
        if len(keys) != nrows:
            raise ValueError("keys length must match embeddings rows")
        metas = metas or [None] * nrows
        for k, e, m in zip(keys, embeddings, metas):
            self.add(k, e, meta=m, timestamp=ts, dedupe_threshold=dedupe_threshold)

    def query(self,
              embedding: np.ndarray,
              topk: int = 5,
              min_similarity: Optional[float] = None,
              filter_fn: Optional[Any] = None) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Query the gallery with a single embedding row. Returns list of (key, similarity, meta) sorted by similarity desc.
        - filter_fn(meta, key) -> bool to allow custom filtering (e.g. exclude active keys)
        """
        if self.embeddings is None:
            return []
        q = np.asarray(embedding, dtype='float32').reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-7)
        sims = (self.embeddings @ q).astype('float32')  # shape (N,)
        idx = np.argsort(sims)[::-1]
        out = []
        for i in idx[:topk]:
            sim = float(sims[i])
            key = self.keys[i]
            meta = self.metas[i]
            if min_similarity is not None and sim < float(min_similarity):
                continue
            if filter_fn is not None:
                try:
                    if not filter_fn(meta, key):
                        continue
                except Exception:
                    continue
            out.append((int(key), float(sim), dict(meta)))
        return out

    def prune_older_than(self, seconds: float):
        """Remove entries with meta['last_seen'] older than now - seconds"""
        if self.embeddings is None:
            return
        cutoff = time.time() - float(seconds)
        keep_idx = [i for i, m in enumerate(self.metas) if m.get('last_seen', 0) >= cutoff]
        if len(keep_idx) == len(self.keys):
            return
        if len(keep_idx) == 0:
            self.embeddings = None
            self.keys = []
            self.metas = []
            return
        self.embeddings = self.embeddings[keep_idx]
        self.keys = [self.keys[i] for i in keep_idx]
        self.metas = [self.metas[i] for i in keep_idx]

    def size(self) -> int:
        return len(self.keys)

    def clear(self):
        self.embeddings = None
        self.keys = []
        self.metas = []

    def get_all(self):
        return list(zip(self.keys, list(self.embeddings) if self.embeddings is not None else [], self.metas))
