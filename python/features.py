from __future__ import annotations
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Optional heavy deps (used if available)
try:
    import cv2
except Exception:
    cv2 = None

try:
    import torch
    import torchvision
    from torchvision import transforms, models
except Exception:
    torch = None
    torchvision = None
    models = None

# -------------------------
# Enhanced IdentityMemory (unchanged behavior)
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
        # selecting best-shot by quality * recency weight (recent & high quality wins)
        now = timestamp
        best_score = -1.0
        best_emb = None
        for e, ts, q in rec['buffer']:
            age = max(0.0, now - ts)
            recency = 1.0 / (1.0 + age)
            score = float(q) * recency
            if score > best_score:
                best_score = score
                best_emb = e
        if best_emb is None:
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
# Crop quality scoring helper (same)
# -------------------------
def compute_crop_quality(frame: Any,
                         bbox_xywh: Tuple[int, int, int, int],
                         conf: Optional[float] = None,
                         mask: Any = None,
                         min_size_for_good: float = 40.0) -> float:
    """
    Compute a [0..1] quality score for a crop.
    """
    try:
        score = 0.0
        weight_total = 0.0

        if conf is not None:
            c = float(conf)
            c = max(0.0, min(1.0, c))
            score += 0.5 * c
            weight_total += 0.5

        x, y, w, h = bbox_xywh
        w = float(max(0.0, w)); h = float(max(0.0, h))
        size_factor = min(1.0, (max(w, h) / max(1.0, min_size_for_good)))
        size_score = size_factor / (1.0 + 0.5 * size_factor)
        score += 0.3 * size_score
        weight_total += 0.3

        if cv2 is not None:
            try:
                x0 = int(max(0, round(x))); y0 = int(max(0, round(y)))
                x1 = int(round(x + w)); y1 = int(round(y + h))
                crop = frame[y0:y1, x0:x1]
                if crop is not None and getattr(crop, "size", None) is not None and crop.size != 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    var = float(lap.var())
                    sharpness = min(1.0, var / 1000.0)
                    score += 0.15 * sharpness
                    weight_total += 0.15
            except Exception:
                pass

        if mask is not None:
            try:
                if isinstance(mask, np.ndarray):
                    if frame is not None and mask.shape[:2] == frame.shape[:2]:
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
            return 0.5
        final = score / weight_total
        final = max(0.0, min(1.0, final))
        return final
    except Exception:
        return 0.5


# -------------------------
# Utility: deterministic random projection helper to fixed dim
# -------------------------
def _deterministic_random_projection(vec: np.ndarray, out_dim: int = 128, seed: int = 42) -> np.ndarray:
    v = np.asarray(vec, dtype='float32').reshape(-1)
    rng = np.random.RandomState(seed)
    proj = rng.normal(size=(v.size, out_dim)).astype('float32')
    out = v.dot(proj)
    out /= (np.linalg.norm(out) + 1e-7)
    return out


# -------------------------
# Torch-based extractor (optional, used if torch + torchvision present)
# -------------------------
class _TorchMobileNetExtractor:
    """
    MobileNetV2-based feature extractor that produces a deterministic 128-D embedding
    by projecting the last feature map vector. Uses pretrained weights if available.
    """
    def __init__(self, device: str = "cpu", out_dim: int = 128):
        if torch is None or models is None:
            raise ImportError("torch/torchvision not available for _TorchMobileNetExtractor")
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')
        # try to load mobilenetv2
        try:
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            # remove classifier, keep features + pooling
            self.model.classifier = torch.nn.Identity()
            self.model.eval()
            self.model.to(self.device)
            self.out_dim = out_dim
            # deterministic projection seed
            self.proj_seed = 1234
        except Exception:
            # fallback to simple CPU identity mapping
            raise

        # preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

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
            inp = self.transform(crop)
            inp = inp.unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model.features(inp)  # feature map
                # global average pooling (mobilenet features have shape [B, C, H, W])
                pooled = feat.mean(dim=[2, 3]).squeeze(0).cpu().numpy()  # shape (C,)
            # deterministic projection to out_dim
            emb = _deterministic_random_projection(pooled, out_dim=self.out_dim, seed=self.proj_seed)
            return emb
        except Exception:
            return None


# -------------------------
# CombinedFallbackExtractor: stronger CPU fallback
# -------------------------
class _CombinedFallbackExtractor:
    """
    Combines color histograms + gradient orientation histogram + simple grid texture
    then applies deterministic projection to 128-D. Lightweight, no external models required.
    """
    def __init__(self, out_dim: int = 128, proj_seed: int = 1):
        self.out_dim = int(out_dim)
        self.proj_seed = int(proj_seed)

    def _color_hist(self, crop: np.ndarray) -> np.ndarray:
        # HSV hist with 32 bins per channel -> 96 dims
        try:
            if cv2 is not None and len(crop.shape) == 3:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
                s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
                v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
                feat = np.concatenate([h, s, v]).astype('float32')
                feat /= (feat.sum() + 1e-7)
                return feat
        except Exception:
            pass
        return np.ones(96, dtype='float32') / 96.0

    def _grad_orient_hist(self, crop: np.ndarray) -> np.ndarray:
        # Compute gradient orientations and histogram them into 32 bins
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            ang = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)  # 0..360
            # bin into 32 bins over 0..360
            bins = np.linspace(0, 360.0, 33)
            hist, _ = np.histogram(ang.flatten(), bins=bins, weights=MobileNet_V2_Weights.DEFAULT)
            hist = hist.astype('float32')
            hist /= (hist.sum() + 1e-7)
            return hist
        except Exception:
            return np.ones(32, dtype='float32') / 32.0

    def _grid_texture(self, crop: np.ndarray, grid: Tuple[int, int] = (4, 4)) -> np.ndarray:
        # split crop into grid and compute mean+std per cell -> 2 * grid_cells dims
        try:
            h, w = crop.shape[:2]
            gx, gy = grid
            cell_h = max(1, h // gy)
            cell_w = max(1, w // gx)
            feats = []
            for iy in range(gy):
                for ix in range(gx):
                    x0 = ix * cell_w
                    y0 = iy * cell_h
                    x1 = min(w, x0 + cell_w)
                    y1 = min(h, y0 + cell_h)
                    cell = crop[y0:y1, x0:x1]
                    if cell is None or getattr(cell, "size", None) == 0:
                        feats.extend([0.0, 0.0])
                        continue
                    if len(cell.shape) == 3:
                        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = cell
                    feats.append(float(gray.mean()))
                    feats.append(float(gray.std()))
            arr = np.array(feats, dtype='float32')
            if arr.size == 0:
                return np.ones(32, dtype='float32') / 32.0
            # normalize to 0..1 roughly
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7)
            return arr
        except Exception:
            return np.ones(32, dtype='float32') / 32.0

    def extract_features(self, frame: Any, bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None, use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        try:
            x, y, w, h = bbox_xywh
            x0 = max(0, int(round(x))); y0 = max(0, int(round(y)))
            x1 = int(round(x + w)); y1 = int(round(y + h))
            # guard bbox validity
            if x1 <= x0 or y1 <= y0:
                return None
            crop = frame[y0:y1, x0:x1]
            if crop is None or getattr(crop, "size", None) == 0:
                return None
            # optionally augment (simple flip) to increase robustness: average features of original+flip
            crops = [crop]
            if use_augmentation:
                try:
                    crops.append(cv2.flip(crop, 1))
                except Exception:
                    pass

            feats = []
            for c in crops:
                ch = self._color_hist(c)    # 96
                gh = self._grad_orient_hist(c)  # 32
                tx = self._grid_texture(c, grid=(4, 4))  # 32
                fv = np.concatenate([ch, gh, tx]).astype('float32')  # ~160
                feats.append(fv)

            fv = np.mean(np.stack(feats, axis=0), axis=0)
            # temporal smoothing: include previous_features if provided (average)
            if previous_features and len(previous_features) > 0:
                prev_stack = np.stack(previous_features[-4:], axis=0)
                # pad/truncate prev features to fv length for safe combination (project later)
                prev_mean = np.mean(prev_stack, axis=0)
                # if prev_mean length differs, expand/truncate using simple resizing
                if prev_mean.size != fv.size:
                    # simple resize by interpolation/resampling
                    if prev_mean.size > fv.size:
                        prev_mean = prev_mean[:fv.size]
                    else:
                        # pad with repeated values
                        pad = np.tile(prev_mean, int(np.ceil(fv.size / prev_mean.size)))[:fv.size]
                        prev_mean = pad
                fv = 0.6 * fv + 0.4 * prev_mean

            # deterministic projection to fixed out_dim
            emb = _deterministic_random_projection(fv, out_dim=self.out_dim, seed=self.proj_seed)
            return emb
        except Exception:
            return None


# -------------------------
# Primary extractor wrapper (keeps API)
# -------------------------
class MaskAwareFeatureExtractor:
    """
    Primary extractor to use in tracker code.
    It will use, in order of preference:
      - user-provided `full_extractor` if passed to constructor
      - torch-based MobileNet extractor (if torch + torchvision available)
      - combined CPU fallback extractor
    The extract_features API remains the same as before.
    """
    def __init__(self, device: str = "cpu", full_extractor: Any = None, out_dim: int = 128):
        self.device = device
        self.out_dim = int(out_dim)
        # if a full_extractor is directly provided, prefer it (keeps compatibility)
        if full_extractor is not None:
            self.impl = full_extractor
            self.impl_name = getattr(full_extractor, "__class__", type(full_extractor)).__name__
            self._has_torch_impl = False
        else:
            # try torch extractor first
            torch_ok = (torch is not None and models is not None)
            impl = None
            if torch_ok:
                try:
                    impl = _TorchMobileNetExtractor(device=device, out_dim=self.out_dim)
                    self._has_torch_impl = True
                except Exception:
                    impl = None
                    self._has_torch_impl = False
            if impl is None:
                # fallback combined extractor
                impl = _CombinedFallbackExtractor(out_dim=self.out_dim, proj_seed=1)
                self._has_torch_impl = False
            self.impl = impl
            self.impl_name = type(self.impl).__name__

    def extract_features(self,
                         frame: Any,
                         bbox_xywh: Tuple[int, int, int, int],
                         mask: Any = None,
                         use_augmentation: bool = False,
                         previous_features: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Backwards-compatible: returns numpy array embedding or None.
        """
        return self.impl.extract_features(frame, bbox_xywh,
                                          mask=mask,
                                          use_augmentation=use_augmentation,
                                          previous_features=previous_features)

    def extract_features_with_quality(self,
                                      frame: Any,
                                      bbox_xywh: Tuple[int, int, int, int],
                                      mask: Any = None,
                                      use_augmentation: bool = False,
                                      previous_features: Optional[List[np.ndarray]] = None,
                                      det_confidence: Optional[float] = None) -> Tuple[Optional[np.ndarray], float]:
        """
        Returns (embedding or None, quality_score 0..1).
        It uses compute_crop_quality() plus a fallback confidence if available.
        """
        emb = self.extract_features(frame, bbox_xywh, mask=mask,
                                    use_augmentation=use_augmentation,
                                    previous_features=previous_features)
        q = compute_crop_quality(frame, bbox_xywh, conf=det_confidence, mask=mask)
        return emb, q


# -------------------------
# build_entity_metas (unchanged - uses extractor)
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
