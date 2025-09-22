# gallery.py
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class DetectionGallery:
    """
    Simple in-memory gallery for embeddings with metadata.
    - embeddings: NxD numpy array (rows L2-normalized)
    - keys: parallel list of track_id or custom id
    - metas: parallel list of metadata dicts
    """

    def __init__(self, max_items: int = 5000):
        self.max_items = int(max_items)
        self.embeddings = None  # np.ndarray shape (N, D) or None
        self.keys: List[int] = []
        self.metas: List[Dict[str, Any]] = []

    def _ensure_shape(self, emb: np.ndarray):
        emb = np.asarray(emb, dtype='float32')
        if emb.ndim == 1:
            emb = emb[None, :]
        # l2-normalize rows
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-7
        emb = emb / norms
        return emb

    def add(self, key: int, embedding: np.ndarray, meta: Optional[Dict[str, Any]] = None, timestamp: Optional[float] = None):
        """
        Add one embedding (or batch). key is track id (or detection id). meta is arbitrary.
        If same key already exists, we append a new row (you may instead update existing one).
        """
        ts = timestamp if timestamp is not None else time.time()
        emb = self._ensure_shape(embedding)
        if self.embeddings is None:
            self.embeddings = emb.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self.keys.append(int(key))
        m = dict(meta or {})
        m.setdefault('last_seen', ts)
        self.metas.append(m)
        self._prune_if_needed()

    def _prune_if_needed(self):
        if self.embeddings is None:
            return
        n = len(self.keys)
        if n <= self.max_items:
            return
        # prune oldest entries (by meta['last_seen']) â€” simple policy
        last_seen = [m.get('last_seen', 0) for m in self.metas]
        order = np.argsort(last_seen)  # ascending
        to_keep = order[-self.max_items:]
        self.embeddings = self.embeddings[to_keep]
        self.keys = [self.keys[i] for i in to_keep]
        self.metas = [self.metas[i] for i in to_keep]

    def query(self, embedding: np.ndarray, topk: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Return list of tuples (key, similarity, meta) sorted by similarity desc.
        If gallery is empty -> []
        """
        if self.embeddings is None:
            return []
        q = self._ensure_shape(embedding)  # shape (1, D)
        sims = np.dot(self.embeddings, q.T).squeeze()  # cosine since normalized
        idx = np.argsort(sims)[::-1][:topk]
        return [(self.keys[i], float(sims[i]), self.metas[i]) for i in idx]

    def prune_older_than(self, seconds: float):
        """Remove entries whose meta['last_seen'] is older than now - seconds."""
        if self.embeddings is None:
            return
        cutoff = time.time() - float(seconds)
        keep = [i for i, m in enumerate(self.metas) if m.get('last_seen', 0) >= cutoff]
        if len(keep) == len(self.keys):
            return
        if len(keep) == 0:
            self.embeddings = None
            self.keys = []
            self.metas = []
            return
        self.embeddings = self.embeddings[keep]
        self.keys = [self.keys[i] for i in keep]
        self.metas = [self.metas[i] for i in keep]

    def size(self) -> int:
        return len(self.keys)