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

    Notes on behavior improvements (backward-compatible):
    - internal `key_to_indices` mapping for fast same-key lookup
    - global dedupe check: if any existing row has similarity >= dedupe_threshold,
      update that row's last_seen (and optionally replace embedding/meta) and skip appending.
      This uses the existing `dedupe_threshold` argument to `add` / `add_batch`.
    - handles embedding-dimension mismatches by padding/truncating incoming embeddings
      to the gallery dimension (if gallery already initialized).
    """

    def __init__(self, max_items: int = 5000):
        self.max_items = int(max_items)
        self.embeddings: Optional[np.ndarray] = None  # shape (N, D)
        self.keys: List[int] = []
        self.metas: List[Dict[str, Any]] = []
        # helper map: key -> list of row indices in embeddings/keys/metas
        self.key_to_indices: Dict[int, List[int]] = {}

    def _ensure_shape_and_normalize(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype='float32')
        if emb.ndim == 1:
            emb = emb[None, :]
        # normalize rows
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-7
        emb = emb / norms
        return emb

    def _rebuild_key_index(self):
        """Rebuild the key->indices mapping from scratch (call after pruning or bulk replace)."""
        self.key_to_indices = {}
        for idx, k in enumerate(self.keys):
            self.key_to_indices.setdefault(int(k), []).append(idx)

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
        to_keep = list(to_keep)  # numpy -> list
        # maintain order of the kept indices in increasing time order -> keep stable
        to_keep.sort()
        self.embeddings = self.embeddings[to_keep]
        self.keys = [self.keys[i] for i in to_keep]
        self.metas = [self.metas[i] for i in to_keep]
        self._rebuild_key_index()

    def _pad_or_truncate_to_dim(self, vec: np.ndarray, dim: int) -> np.ndarray:
        vec = np.asarray(vec, dtype='float32')
        if vec.ndim == 1:
            if vec.shape[0] == dim:
                return vec
            if vec.shape[0] < dim:
                pad = np.zeros((dim - vec.shape[0],), dtype='float32')
                out = np.concatenate([vec, pad])
                out = out / (np.linalg.norm(out) + 1e-7)
                return out
            else:
                out = vec[:dim]
                out = out / (np.linalg.norm(out) + 1e-7)
                return out
        else:
            # 2D case -> process row-wise
            rows = []
            for r in vec:
                rows.append(self._pad_or_truncate_to_dim(r, dim))
            return np.stack(rows, axis=0)

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
        - dedupe_threshold: if any existing embedding (across gallery) has similarity >= threshold,
          we update that existing row's last_seen (and optionally replace embedding/meta) and skip appending.
          This helps avoid duplicate near-identical rows across different keys.
        - replace_similar: if True and similar entry exists, we replace that row's embedding/meta.
        """
        ts = timestamp if timestamp is not None else time.time()
        emb_in = np.asarray(embedding, dtype='float32')
        # If gallery is already initialized, ensure incoming vectors match dimension by padding/truncating
        if self.embeddings is not None:
            target_dim = self.embeddings.shape[1]
            if emb_in.ndim == 1:
                emb = self._pad_or_truncate_to_dim(emb_in, target_dim)[None, :]
            else:
                emb = self._pad_or_truncate_to_dim(emb_in, target_dim)
        else:
            # no gallery yet - just normalize shape and use its dimension
            emb = emb_in if emb_in.ndim > 1 else emb_in[None, :]

        emb = self._ensure_shape_and_normalize(emb)
        num = emb.shape[0]

        # If gallery is empty, initialize
        if self.embeddings is None:
            self.embeddings = emb.copy()
            for i in range(num):
                self.keys.append(int(key))
                m = dict(meta or {})
                m.setdefault('last_seen', ts)
                self.metas.append(m)
            self._rebuild_key_index()
            self._prune_if_needed()
            return

        # GLOBAL dedupe: if any existing row has similarity >= dedupe_threshold, update that row's meta and optionally replace embedding
        if dedupe_threshold is not None:
            # compute sims against all existing embeddings in one go
            try:
                q = emb  # shape (num, D)
                existing = self.embeddings  # shape (N, D)
                sims_all = existing @ q.T  # shape (N, num)
            except Exception:
                sims_all = None
            if sims_all is not None:
                # for each incoming row, check best match across all existing rows
                for i in range(num):
                    col = sims_all[:, i]
                    if col.size == 0:
                        continue
                    best_idx = int(np.argmax(col))
                    best_sim = float(col[best_idx])
                    if best_sim >= float(dedupe_threshold):
                        # similar existing row found - update its meta last_seen and optionally replace
                        try:
                            self.metas[best_idx].update(dict(meta or {}))
                            self.metas[best_idx]['last_seen'] = ts
                            if replace_similar:
                                # replace embedding row (keep same key mapping)
                                self.embeddings[best_idx] = emb[i : i + 1]
                            # skip appending this incoming row
                            continue
                        except Exception:
                            # if anything fails, fall back to appending
                            pass
                    # else append below

        # Append rows one-by-one (so key mapping is correct)
        for i in range(num):
            row = emb[i : i + 1]  # shape (1, D)
            self.embeddings = np.vstack([self.embeddings, row])
            self.keys.append(int(key))
            m = dict(meta or {})
            m.setdefault('last_seen', ts)
            self.metas.append(m)

        # rebuild index and prune if necessary
        self._rebuild_key_index()
        self._prune_if_needed()

    def add_batch(self,
                  keys: List[int],
                  embeddings: np.ndarray,
                  metas: Optional[List[Dict[str, Any]]] = None,
                  timestamp: Optional[float] = None,
                  dedupe_threshold: float = 0.995):
        """
        Batch add convenience. keys length must match number of embeddings rows.
        Uses the same dedupe semantics as `add`.
        """
        ts = timestamp if timestamp is not None else time.time()
        embeddings = np.asarray(embeddings, dtype='float32')
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
            self.key_to_indices = {}
            return
        self.embeddings = self.embeddings[keep_idx]
        self.keys = [self.keys[i] for i in keep_idx]
        self.metas = [self.metas[i] for i in keep_idx]
        self._rebuild_key_index()

    def size(self) -> int:
        return len(self.keys)

    def clear(self):
        self.embeddings = None
        self.keys = []
        self.metas = []
        self.key_to_indices = {}

    def get_all(self):
        """
        Return list of tuples (key, embedding_row, meta).
        embedding_row will be numpy array rows if available; or empty list if gallery is empty.
        """
        if self.embeddings is None:
            return list(zip(self.keys, [], self.metas))
        # return copies for safety
        emb_list = [np.array(r, copy=True) for r in list(self.embeddings)]
        return list(zip(self.keys, emb_list, self.metas.copy()))
