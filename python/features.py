# features.py
from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Optional heavy deps (used if available)
try:
    import cv2
    import torch
    from PIL import Image
except Exception:
    cv2 = None
    torch = None
    Image = None


# -------------------------
# Enhanced IdentityMemory
# -------------------------
class IdentityMemory:
    """
    Enhanced identity memory storing multiple recent embeddings per track,
    with EMA + best-shot selection + pruning support.

    API:
        mem = IdentityMemory(dim=128, momentum=0.95, buffer_size=5)
        mem.update(track_id, embedding, timestamp=None, quality=1.0)
        rep = mem.get(track_id)         # representative embedding (best-shot)
        ema = mem.get_ema(track_id)     # EMA embedding
        mem.prune_old(max_age_seconds)
    """
    def __init__(self, dim: int = 128, momentum: float = 0.95, buffer_size: int = 5):
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.buffer_size = int(buffer_size)
        # records: track_id -> {'ema': np.array, 'buffer': [(emb, ts, quality)], 'rep': np.array}
        self.records: Dict[int, Dict[str, Any]] = {}

    def update(self,
               track_id: int,
               embedding: Optional[np.ndarray],
               timestamp: Optional[float] = None,
               quality: float = 1.0):
        if embedding is None:
            return
        if timestamp is None:
            timestamp = time.time()

        emb = np.asarray(embedding, dtype='float32')
        if emb.ndim == 1:
            pass
        elif emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb.reshape(-1)
        else:
            # try to flatten if weird shape
            emb = emb.reshape(-1)

        # normalize
        n = np.linalg.norm(emb) + 1e-7
        emb = emb / n

        rec = self.records.get(track_id)
        if rec is None:
            rec = {}
            rec['ema'] = emb.copy()
            rec['buffer'] = [(emb.copy(), timestamp, float(quality))]
            rec['rep'] = emb.copy()
            self.records[track_id] = rec
            return

        # update EMA
        m = rec.get('ema')
        new_ema = self.momentum * m + (1.0 - self.momentum) * emb
        rec['ema'] = new_ema / (np.linalg.norm(new_ema) + 1e-7)

        # buffer append (recent "good" shots)
        rec['buffer'].append((emb.copy(), timestamp, float(quality)))
        if len(rec['buffer']) > self.buffer_size:
            rec['buffer'].pop(0)

        # compute representative embedding:
        # currently selecting best-shot by quality * recency weight, fallback to EMA average
        now = timestamp
        best_score = -1.0
        best_emb = None
        for e, ts, q in rec['buffer']:
            # recency weight = 1 / (1 + age_seconds) (older -> smaller)
            age = max(0.0, now - ts)
            recency = 1.0 / (1.0 + age)
            score = float(q) * recency
            if score > best_score:
                best_score = score
                best_emb = e
        if best_emb is None:
            # fallback to EMA
            rec['rep'] = rec['ema'].copy()
        else:
            rec['rep'] = best_emb.copy()

    def get(self, track_id: int) -> Optional[np.ndarray]:
        rec = self.records.get(track_id)
        if rec is None:
            return None
        return rec.get('rep')

    def get_ema(self, track_id: int) -> Optional[np.ndarray]:
        rec = self.records.get(track_id)
        if rec is None:
            return None
        return rec.get('ema')

    def prune_old(self, max_age_seconds: float):
        """
        Remove track records whose latest buffer entry is older than max_age_seconds.
        """
        now = time.time()
        to_delete = []
        for tid, rec in self.records.items():
            buf = rec.get('buffer', [])
            if not buf:
                to_delete.append(tid)
                continue
            last_ts = buf[-1][1]
            if (now - last_ts) > max_age_seconds:
                to_delete.append(tid)
        for tid in to_delete:
            del self.records[tid]


# -------------------------
# Crop quality scoring helper
# -------------------------
def compute_crop_quality(frame: Any,
                         bbox_xywh: Tuple[int, int, int, int],
                         conf: Optional[float] = None,
                         mask: Any = None,
                         min_size_for_good: float = 40.0) -> float:
    """
    Compute a [0..1] quality score for a crop.
    Heuristics combined:
      - detection confidence (if provided)
      - sharpness / blur (variance of Laplacian if cv2 available)
      - size relative to min_size_for_good (tiny crops are low quality)
      - mask coverage (if mask provided): fraction of mask area in bbox

    Returns 0..1 where 1 is best.
    Non-destructive if cv2 is missing: uses size + conf primarily.
    """
    try:
        score = 0.0
        weight_total = 0.0

        # detection confidence (if any)
        if conf is not None:
            c = float(conf)
            c = max(0.0, min(1.0, c))
            score += 0.5 * c
            weight_total += 0.5

        x, y, w, h = bbox_xywh
        w = float(max(0.0, w))
        h = float(max(0.0, h))
        area = w * h

        # size factor: scaled sigmoid-like: sizes >= min_size_for_good => good
        size_factor = min(1.0, (max(w, h) / max(1.0, min_size_for_good)))
        # map 0..inf -> 0..1 with diminishing returns
        size_score = size_factor / (1.0 + 0.5 * size_factor)
        score += 0.3 * size_score
        weight_total += 0.3

        # sharpness / blur
        if cv2 is not None:
            try:
                x0 = int(max(0, round(x))); y0 = int(max(0, round(y)))
                x1 = int(round(x + w)); y1 = int(round(y + h))
                crop = frame[y0:y1, x0:x1]
                if crop is not None and getattr(crop, "size", None) is not None and crop.size != 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    var = float(lap.var())
                    # map var -> 0..1 using soft-saturating function
                    sharpness = min(1.0, var / 1000.0)
                    score += 0.15 * sharpness
                    weight_total += 0.15
            except Exception:
                pass

        # mask coverage (if available)
        if mask is not None:
            try:
                # mask expected to be a full-frame binary mask or crop-level mask
                if isinstance(mask, np.ndarray):
                    # if same size as frame -> crop it; else assume it's crop mask
                    if mask.shape[:2] == frame.shape[:2]:
                        x0 = int(max(0, round(x))); y0 = int(max(0, round(y)))
                        x1 = int(round(x + w)); y1 = int(round(y + h))
                        sub = mask[y0:y1, x0:x1]
                    else:
                        sub = mask
                    if sub is not None and sub.size != 0:
                        frac = float((sub > 0).sum()) / float(sub.size)
                        score += 0.05 * float(frac)
                        weight_total += 0.05
            except Exception:
                pass

        if weight_total <= 0.0:
            return 0.5  # neutral value when nothing available

        final = score / weight_total
        # clamp 0..1
        final = max(0.0, min(1.0, final))
        return final
    except Exception:
        return 0.5


# -------------------------
# MaskAwareFeatureExtractor (thin wrapper)
# -------------------------
class MaskAwareFeatureExtractor:
    """
    Stable interface for extracting appearance embeddings.
    If you pass a heavy `full_extractor` it will be used; otherwise, fallback is used.
    """
    def __init__(self, device: str = "cpu", full_extractor: Any = None):
        self.device = device
        if full_extractor is not None:
            self.impl = full_extractor
        else:
            # attempt to import a heavier implementation if author kept it under 'features'
            try:
                from features import MaskAwareFeatureExtractor as FE  # type: ignore
                # if it exists, instantiate the user's heavy impl
                self.impl = FE(device=device)
            except Exception:
                # fallback to tiny extractor
                self.impl = _TinyFallbackExtractor(device=device)

    def extract_features(self,
                         frame: Any,
                         bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None,
                         use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        return self.impl.extract_features(frame, bbox_xywh,
                                          mask=mask,
                                          use_augmentation=use_augmentation,
                                          previous_features=previous_features)


# -------------------------
# Tiny fallback extractor
# -------------------------
class _TinyFallbackExtractor:
    """
    Cheap fallback extracting a color-histogram-based vector with temporal smoothing.
    Returns a 128-d normalized numpy vector.
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _color_hist(self, crop: Any) -> np.ndarray:
        if cv2 is not None:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
            v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
            feat = np.concatenate([h, s, v]).astype("float32")
            feat /= (feat.sum() + 1e-7)
            return feat
        else:
            return np.ones(96, dtype="float32") / 96.0

    def extract_features(self, frame: Any, bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None, use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        try:
            x, y, w, h = bbox_xywh
            x0 = max(0, int(round(x))); y0 = max(0, int(round(y)))
            x1 = int(round(x + w)); y1 = int(round(y + h))
            crop = frame[y0:y1, x0:x1]
            if crop is None or getattr(crop, "size", None) == 0:
                return None
            color = self._color_hist(crop)
            if previous_features and len(previous_features) > 0:
                prev = np.mean(np.stack(previous_features[-4:], axis=0), axis=0)
                color = 0.6 * color + 0.4 * prev[:color.shape[0]]
            out = np.zeros(128, dtype="float32")
            L = min(len(color), 128)
            out[:L] = color[:L]
            out /= (np.linalg.norm(out) + 1e-7)
            return out
        except Exception:
            return None


# -------------------------
# build_entity_metas
# -------------------------
def build_entity_metas(frame: Any,
                       confused_groups: List[Any],
                       extractor: MaskAwareFeatureExtractor,
                       masks: Optional[Dict[Tuple[float, float, float, float], Any]] = None,
                       previous_features_map: Optional[Dict[int, List[np.ndarray]]] = None
                       ) -> Dict[Tuple[float, float, float, float], Dict]:
    """
    For each confused group (object with .bbox and .possible_ids), compute an embedding and
    return mapping:
        { group.bbox_xyxy : {'embedding': np.ndarray, 'frame': frame, 'bbox_xywh': (x,y,w,h), 'mask': mask_crop, ... } }
    """
    out: Dict[Tuple[float, float, float, float], Dict] = {}
    for g in confused_groups:
        bbox_xyxy = tuple(g.bbox)
        x1, y1, x2, y2 = bbox_xyxy
        w = max(0, x2 - x1); h = max(0, y2 - y1)
        bbox_xywh = (int(round(x1)), int(round(y1)), int(round(w)), int(round(h)))
        meta: Dict[str, Any] = {'frame': frame, 'bbox_xywh': bbox_xywh}

        if masks and bbox_xyxy in masks:
            meta['mask'] = masks[bbox_xyxy]

        # assemble previous features if any (flattened list from candidate possible_ids)
        prev_feats = []
        if previous_features_map:
            for tid in getattr(g, 'possible_ids', []):
                pf = previous_features_map.get(tid)
                if pf:
                    prev_feats.extend(pf[-6:])
        if prev_feats:
            meta['previous_features'] = prev_feats

        # attempt extraction
        try:
            emb = extractor.extract_features(frame, bbox_xywh, mask=meta.get('mask'),
                                             use_augmentation=True, previous_features=meta.get('previous_features'))
            if emb is not None:
                emb = np.array(emb, dtype='float32')
                emb /= (np.linalg.norm(emb) + 1e-7)
                meta['embedding'] = emb
                out[bbox_xyxy] = meta
        except Exception:
            continue
    return out