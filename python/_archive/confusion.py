# confusion.py
from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Protocol

# try importing optional helper modules (features.py and gallery.py)
try:
    from features import MaskAwareFeatureExtractor, IdentityMemory
except Exception:
    MaskAwareFeatureExtractor = None
    IdentityMemory = None

try:
    from gallery import DetectionGallery
except Exception:
    DetectionGallery = None


# ---- Tunables (move to config or pass to backends) ----
OVERLAP_THRESHOLD = 0.5
MIDPOINT_DIST_THRESHOLD = 60
IOU_ACCEPT = 0.25
SIGMA_MULTIPLIER = 2.0
NORMALIZED_DIST_THRESHOLD = 3.0
IOU_WEIGHT = 1.0
DIST_WEIGHT = 1.0


# ---- Helpers (IoU / midpoint) ----
def calculate_iou(bbox1: Tuple[float, float, float, float],
                  bbox2: Tuple[float, float, float, float]) -> float:
    x1 = max(bbox1[0], bbox2[0]); y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2]); y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0, (bbox1[2] - bbox1[0])) * max(0, (bbox1[3] - bbox1[1]))
    a2 = max(0, (bbox2[2] - bbox2[0])) * max(0, (bbox2[3] - bbox2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def calculate_midpoint(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ---- Data containers ----
@dataclass
class ConfusedEntity:
    possible_ids: List[int]
    bbox: Tuple[float, float, float, float]

    @property
    def midpoint(self) -> Tuple[float, float]:
        return calculate_midpoint(self.bbox)


@dataclass
class CandidateScore:
    id: int
    score: float
    iou: float
    norm_dist: float
    intersects_expanded: bool
    pred_bbox: Tuple[float, float, float, float]
    expanded_bbox: Tuple[float, float, float, float]
    mean_std: float
    reason: Optional[str]


# ---- Robust prediction -> bbox extractor (works with different kalman styles) ----
def _bbox_from_tracker_prediction(kalman_obj: Any, sigma_multiplier: float = SIGMA_MULTIPLIER
                                 ) -> Tuple[Tuple[float, float, float, float],
                                            Tuple[float, float, float, float],
                                            float]:
    try:
        inner = getattr(kalman_obj, "kalman", kalman_obj)

        if hasattr(inner, "predict_without_updating") and hasattr(inner, "expanded_predicted_bbox"):
            pred = tuple(inner.predict_without_updating())
            expanded = tuple(inner.expanded_predicted_bbox(sigma_multiplier))
            if hasattr(inner, "average_position_std"):
                s1, s2 = inner.average_position_std()
            else:
                s1, s2 = 1.0, 1.0
            return pred, expanded, (s1 + s2) / 2.0

        if hasattr(kalman_obj, "predict"):
            if hasattr(kalman_obj, "predict_float"):
                px, py = kalman_obj.predict_float()
            else:
                px, py = kalman_obj.predict()
            if hasattr(kalman_obj, "position_std"):
                sx, sy = kalman_obj.position_std()
            else:
                sx, sy = 10.0, 10.0
            halfw = max(2.0, sx * sigma_multiplier)
            halfh = max(2.0, sy * sigma_multiplier)
            pred = (px - halfw, py - halfh, px + halfw, py + halfh)
            expanded = (px - 2 * halfw, py - 2 * halfh, px + 2 * halfw, py + 2 * halfh)
            return tuple(pred), tuple(expanded), (sx + sy) / 2.0
    except Exception:
        pass
    return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 1000.0


# ---- Backend interface ----
class ReIDBackend(Protocol):
    def score_candidates(self, entity: ConfusedEntity, kalman_trackers: Dict[int, Any], entity_meta: Dict = None) -> List[CandidateScore]:
        ...


# ---- Heuristic backend (unchanged) ----
class IoUKalmanBackend:
    def __init__(self,
                 iou_accept: float = IOU_ACCEPT,
                 sigma_multiplier: float = SIGMA_MULTIPLIER,
                 normalized_dist_threshold: float = NORMALIZED_DIST_THRESHOLD,
                 iou_weight: float = IOU_WEIGHT,
                 dist_weight: float = DIST_WEIGHT):
        self.iou_accept = float(iou_accept)
        self.sigma_multiplier = float(sigma_multiplier)
        self.norm_thr = float(normalized_dist_threshold)
        self.iou_weight = float(iou_weight)
        self.dist_weight = float(dist_weight)

    def score_candidates(self, entity: ConfusedEntity, kalman_trackers: Dict[int, Any], entity_meta: Dict = None) -> List[CandidateScore]:
        out: List[CandidateScore] = []
        for cand_id in entity.possible_ids:
            track = kalman_trackers.get(cand_id)
            if track is None:
                continue
            pred_bbox, expanded_bbox, mean_std = _bbox_from_tracker_prediction(track, self.sigma_multiplier)
            iou_score = calculate_iou(entity.bbox, pred_bbox)
            pred_mid = calculate_midpoint(pred_bbox)
            dist = math.dist(entity.midpoint, pred_mid)
            norm_dist = dist / (1.0 + mean_std)
            intersects_expanded = not (
                entity.bbox[2] < expanded_bbox[0] or entity.bbox[0] > expanded_bbox[2]
                or entity.bbox[3] < expanded_bbox[1] or entity.bbox[1] > expanded_bbox[3]
            )
            score = self.iou_weight * iou_score - self.dist_weight * (norm_dist / (1.0 + norm_dist))
            reason = None
            if iou_score >= self.iou_accept:
                reason = f"iou_ok({iou_score:.2f})"
            elif intersects_expanded:
                reason = f"intersects_expanded(std~{mean_std:.2f})"
            elif norm_dist <= self.norm_thr:
                reason = f"close_by_normdist({norm_dist:.2f})"
            out.append(CandidateScore(
                id=cand_id,
                score=score,
                iou=iou_score,
                norm_dist=norm_dist,
                intersects_expanded=intersects_expanded,
                pred_bbox=tuple(pred_bbox),
                expanded_bbox=tuple(expanded_bbox),
                mean_std=mean_std,
                reason=reason
            ))
        out.sort(key=lambda x: x.score, reverse=True)
        return out


# ---- Dummy DL backend as example (keeps previous behaviour) ----
class DummyDLBackend:
    def __init__(self, similarity_threshold: float = 0.5, identity_memory: Any = None):
        self.sim_threshold = float(similarity_threshold)
        self.identity_memory = identity_memory

    @staticmethod
    def _cosine(a: Optional[Any], b: Optional[Any]) -> float:
        import numpy as _np
        if a is None or b is None:
            return -1.0
        a = _np.asarray(a, dtype='float32'); b = _np.asarray(b, dtype='float32')
        na = _np.linalg.norm(a) + 1e-7; nb = _np.linalg.norm(b) + 1e-7
        return float(_np.dot(a, b) / (na * nb))

    def score_candidates(self, entity: ConfusedEntity, kalman_trackers: Dict[int, Any], entity_meta: Dict = None) -> List[CandidateScore]:
        out: List[CandidateScore] = []
        meta = entity_meta or {}
        emb = None
        # entity_meta may be mapping bbox->meta produced by features.build_entity_metas
        if isinstance(meta, dict) and entity.bbox in meta and isinstance(meta[entity.bbox], dict):
            emb = meta[entity.bbox].get('embedding')
        if emb is None:
            return []
        for cand_id in entity.possible_ids:
            track = kalman_trackers.get(cand_id)
            cand_emb = None
            if track is not None and hasattr(track, "embedding") and getattr(track, "embedding", None) is not None:
                cand_emb = getattr(track, "embedding")
            elif self.identity_memory is not None:
                cand_emb = self.identity_memory.get(cand_id)
            sim = self._cosine(emb, cand_emb)
            reason = f"dl_sim({sim:.2f})" if sim >= self.sim_threshold else None
            out.append(CandidateScore(
                id=cand_id,
                score=float(sim),
                iou=0.0,
                norm_dist=999.0,
                intersects_expanded=False,
                pred_bbox=(0.0, 0.0, 0.0, 0.0),
                expanded_bbox=(0.0, 0.0, 0.0, 0.0),
                mean_std=0.0,
                reason=reason
            ))
        out.sort(key=lambda x: x.score, reverse=True)
        return out


# ---- NEW: GalleryBackend ----
class GalleryBackend:
    """
    Backend that queries a DetectionGallery for past appearance embeddings and proposes old track IDs.
    - gallery: an instance of DetectionGallery (in-memory or wrapper over FAISS/etc.)
    - extractor: optional MaskAwareFeatureExtractor used to compute embedding if entity_meta lacks embedding
    - identity_memory: optional IdentityMemory (fallback for per-track embeddings)
    - sim_threshold: cosine similarity gating (0..1)
    - time_window: seconds; ignore gallery entries older than this
    - topk: how many gallery items to consider
    """
    def __init__(self,
                 gallery: Any,
                 extractor: Optional[Any] = None,
                 identity_memory: Optional[Any] = None,
                 sim_threshold: float = 0.65,
                 time_window: float = 60.0,
                 topk: int = 5,
                 require_not_active: bool = False):
        if DetectionGallery is None and gallery is None:
            raise ImportError("GalleryBackend requires a DetectionGallery instance (gallery.py).")
        self.gallery = gallery
        self.extractor = extractor
        self.identity_memory = identity_memory
        self.sim_threshold = float(sim_threshold)
        self.time_window = float(time_window)
        self.topk = int(topk)
        self.require_not_active = bool(require_not_active)

    @staticmethod
    def _cosine(a, b):
        import numpy as _np
        if a is None or b is None:
            return -1.0
        a = _np.asarray(a, dtype='float32'); b = _np.asarray(b, dtype='float32')
        na = _np.linalg.norm(a) + 1e-7; nb = _np.linalg.norm(b) + 1e-7
        return float(_np.dot(a, b) / (na * nb))

    def _get_embedding_from_meta_or_extractor(self, entity: ConfusedEntity, entity_meta: Dict) -> Optional[Any]:
        if not entity_meta:
            return None
        # entity_meta expected as mapping bbox->meta (features.build_entity_metas)
        meta_for_bbox = None
        if isinstance(entity_meta, dict) and entity.bbox in entity_meta:
            meta_for_bbox = entity_meta[entity.bbox]
        # meta_for_bbox may be {'embedding':..., 'frame':..., 'bbox_xywh':...}
        if isinstance(meta_for_bbox, dict) and 'embedding' in meta_for_bbox:
            return meta_for_bbox['embedding']
        # attempt compute via extractor if frame and bbox_xywh are present
        if self.extractor is not None and isinstance(meta_for_bbox, dict):
            frame = meta_for_bbox.get('frame')
            bbox_xywh = meta_for_bbox.get('bbox_xywh')
            prev = meta_for_bbox.get('previous_features')
            if frame is not None and bbox_xywh is not None:
                try:
                    emb = self.extractor.extract_features(frame, bbox_xywh, mask=meta_for_bbox.get('mask'),
                                                          use_augmentation=False, previous_features=prev)
                    return emb
                except Exception:
                    return None
        return None

    def score_candidates(self, entity: ConfusedEntity, kalman_trackers: Dict[int, Any], entity_meta: Dict = None) -> List[CandidateScore]:
        """
        For the confused entity, query gallery for nearest embeddings and produce CandidateScore objects.
        Note: candidate ids returned are the gallery keys (original track ids or keys you used when adding).
        """
        out: List[CandidateScore] = []
        # try to obtain embedding for this confused entity
        emb = self._get_embedding_from_meta_or_extractor(entity, entity_meta or {})
        if emb is None:
            # nothing to query with
            return []

        # query gallery
        try:
            results = self.gallery.query(emb, topk=self.topk)
        except Exception:
            # gallery query failed (bad gallery), return empty
            return []

        now = time.time()
        for key, sim, meta in results:
            # gating by recency
            last_seen = meta.get('last_seen', 0.0) if isinstance(meta, dict) else 0.0
            if (now - last_seen) > self.time_window:
                continue
            # if user requested we only propose keys that are not currently active (avoid duplicate proposals),
            # skip if active
            if self.require_not_active and (key in kalman_trackers):
                continue

            # compute iou w/ stored bbox if available (useful info)
            stored_bbox = None
            if isinstance(meta, dict):
                stored_bbox = meta.get('bbox')
            if stored_bbox is None:
                iou_val = 0.0
                pred_bbox = (0.0, 0.0, 0.0, 0.0)
                expanded_bbox = pred_bbox
                norm_dist = 999.0
            else:
                iou_val = calculate_iou(entity.bbox, tuple(stored_bbox))
                pred_bbox = tuple(stored_bbox)
                expanded_bbox = tuple(stored_bbox)
                # compute midpoint distance normalized by entity size
                ent_mid = entity.midpoint
                stored_mid = calculate_midpoint(pred_bbox)
                dist = math.dist(ent_mid, stored_mid)
                ent_w = max(1.0, (entity.bbox[2] - entity.bbox[0]))
                ent_h = max(1.0, (entity.bbox[3] - entity.bbox[1]))
                norm_dist = dist / (0.5 * (ent_w + ent_h))

            reason = f"gallery({sim:.2f})" if sim >= self.sim_threshold else None

            out.append(CandidateScore(
                id=int(key),
                score=float(sim),
                iou=float(iou_val),
                norm_dist=float(norm_dist),
                intersects_expanded=False,
                pred_bbox=tuple(pred_bbox),
                expanded_bbox=tuple(expanded_bbox),
                mean_std=0.0,
                reason=reason
            ))

        out.sort(key=lambda x: x.score, reverse=True)
        return out


# ---- Orchestrator: combine backends / produce resolved assignments ----
class ReIDManager:
    """
    Manages detection of confused groups and resolution using provided ReID backends.
    Usage:
      manager = ReIDManager(backends=[IoUKalmanBackend(), GalleryBackend(gallery, extractor, memory)])
      confused = manager.find_confused_groups(list_of_entities)
      entity_metas = features.build_entity_metas(frame, confused, extractor, masks, previous_features_map)
      resolved = manager.resolve(confused, kalman_trackers, entity_metas)
    """
    def __init__(self, backends: Optional[List[ReIDBackend]] = None,
                 overlap_threshold: float = OVERLAP_THRESHOLD,
                 midpoint_dist_threshold: float = MIDPOINT_DIST_THRESHOLD):
        self.backends = backends or [IoUKalmanBackend()]
        self.overlap_threshold = overlap_threshold
        self.midpoint_dist_threshold = midpoint_dist_threshold

    def find_confused_groups(self, entities: List[Any]) -> List[ConfusedEntity]:
        confused_groups: List[ConfusedEntity] = []
        processed = set()
        n = len(entities)
        for i, e1 in enumerate(entities):
            if i in processed:
                continue
            group = [e1]
            processed.add(i)
            for j in range(i + 1, n):
                if j in processed:
                    continue
                e2 = entities[j]
                if calculate_iou(e1.bbox, e2.bbox) > self.overlap_threshold:
                    group.append(e2)
                    processed.add(j)
                else:
                    m1 = calculate_midpoint(e1.bbox)
                    m2 = calculate_midpoint(e2.bbox)
                    if math.dist(m1, m2) <= self.midpoint_dist_threshold:
                        group.append(e2)
                        processed.add(j)
            if len(group) > 1:
                possible_ids = []
                for g in group:
                    if hasattr(g, "possible_ids"):
                        possible_ids += list(getattr(g, "possible_ids") or [])
                possible_ids = list(dict.fromkeys(possible_ids))
                xs = [c.bbox[0] for c in group] + [c.bbox[2] for c in group]
                ys = [c.bbox[1] for c in group] + [c.bbox[3] for c in group]
                union_bbox = (min(xs), min(ys), max(xs), max(ys))
                confused_groups.append(ConfusedEntity(possible_ids, union_bbox))
        return confused_groups

    def resolve(self, confused_entities: List[ConfusedEntity],
                kalman_trackers: Dict[int, Any],
                entity_metas: Optional[Dict[Tuple[float, float, float, float], Dict]] = None
               ) -> List[Tuple[Optional[int], Tuple[float, float, float, float], Optional[float], str]]:
        """
        For each confused entity, ask each backend for candidate scores.
        Select highest scoring candidate which has a 'reason' (i.e., passed gating).
        entity_metas: optional mapping from entity.bbox -> metadata (e.g., embeddings for DL)
        Returns list of (best_id_or_None, entity_bbox, score_or_None, reason)
        """
        results = []
        entity_metas = entity_metas or {}
        for ent in confused_entities:
            aggregated: Dict[int, CandidateScore] = {}
            for backend in self.backends:
                try:
                    scores = backend.score_candidates(ent, kalman_trackers, entity_metas)
                except TypeError:
                    scores = backend.score_candidates(ent, kalman_trackers)
                for cs in scores:
                    if cs.id not in aggregated or cs.score > aggregated[cs.id].score:
                        aggregated[cs.id] = cs

            candidates = list(aggregated.values())
            candidates.sort(key=lambda x: x.score, reverse=True)
            best = None
            for c in candidates:
                # only accept candidates that have a gating reason (backend sets reason when positive)
                if c.reason is not None:
                    best = c
                    break

            if best is not None:
                results.append((best.id, ent.bbox, best.score, best.reason))
            else:
                results.append((None, ent.bbox, None, "no_candidate"))
        return results
