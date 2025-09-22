# confusion.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Protocol

# Tunable defaults (move to config or pass to backends if you prefer)
OVERLAP_THRESHOLD = 0.5      # used for initial confusion grouping (IoU)
MIDPOINT_DIST_THRESHOLD = 60
IOU_ACCEPT = 0.25
SIGMA_MULTIPLIER = 2.0
NORMALIZED_DIST_THRESHOLD = 3.0
IOU_WEIGHT = 1.0
DIST_WEIGHT = 1.0


# ---- Basic helpers (kept for compatibility) ----
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
    # return float midpoints
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# ---- Data containers ----
@dataclass
class ConfusedEntity:
    possible_ids: List[int]
    bbox: Tuple[float, float, float, float]

    @property
    def midpoint(self) -> Tuple[float, float]:
        return calculate_midpoint(self.bbox)

    def get_label(self) -> str:
        if self.possible_ids:
            return "? ".join(map(str, self.possible_ids)) + "?"
        return "?"


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


# ---- Tracker-prediction → bbox helper (robust to several kalman wrappers) ----
def _bbox_from_tracker_prediction(kalman_obj: Any, sigma_multiplier: float = SIGMA_MULTIPLIER
                                 ) -> Tuple[Tuple[float, float, float, float],
                                            Tuple[float, float, float, float],
                                            float]:
    """
    Robustly extract a predicted bbox, an expanded bbox (for tolerant matching), and a mean std estimate.
    Accepts:
      - Track wrapper with .kalman
      - BBoxGroupKalmanTracker-like object with predict_without_updating() and expanded_predicted_bbox(sigma)
      - Point-kalman with predict() / predict_float() and position_std()
    Returns (pred_bbox, expanded_bbox, mean_std)
    """
    try:
        # If wrapper: unwrap .kalman
        k = getattr(kalman_obj, "kalman", kalman_obj)

        # If bbox-group kalman style
        if hasattr(k, "predict_without_updating") and hasattr(k, "expanded_predicted_bbox"):
            pred = k.predict_without_updating()
            expanded = k.expanded_predicted_bbox(sigma_multiplier)
            avg_sx, avg_sy = (1.0, 1.0)
            if hasattr(k, "average_position_std"):
                avg_sx, avg_sy = k.average_position_std()
            return tuple(pred), tuple(expanded), (avg_sx + avg_sy) / 2.0

        # fallback to point-kalman style
        if hasattr(kalman_obj, "predict"):
            # prefer non-committing float prediction if present
            if hasattr(kalman_obj, "predict_float"):
                px, py = kalman_obj.predict_float()
            else:
                px, py = kalman_obj.predict()
            if hasattr(kalman_obj, "position_std"):
                sx, sy = kalman_obj.position_std()
            else:
                sx, sy = (10.0, 10.0)
            halfw = max(2.0, sx * sigma_multiplier)
            halfh = max(2.0, sy * sigma_multiplier)
            pred = (px - halfw, py - halfh, px + halfw, py + halfh)
            expanded = (px - 2 * halfw, py - 2 * halfh, px + 2 * halfw, py + 2 * halfh)
            return tuple(pred), tuple(expanded), (sx + sy) / 2.0
    except Exception:
        pass

    # safe default that will rarely match
    return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 1000.0


# ---- Backend interface: implement new re-id strategies by subclassing/implementing Protocol ----
class ReIDBackend(Protocol):
    """
    ReID backend interface. Given a confused entity and kalman trackers,
    return a list of CandidateScore (one per candidate id) - backend may also return [].
    The ReIDManager will aggregate / select across backends.
    """
    def score_candidates(self, entity: ConfusedEntity, kalman_trackers: Dict[int, Any]) -> List[CandidateScore]:
        ...


# ---- Example heuristic backend (keeps your current heuristic behavior) ----
class IoUKalmanBackend:
    """
    Heuristic matcher: uses IoU, predicted bbox, expanded bbox and normalized midpoint distance.
    Designed to behave like your previous resolve_confusion logic.
    """
    def __init__(self,
                 iou_accept=IOU_ACCEPT,
                 sigma_multiplier: float = SIGMA_MULTIPLIER,
                 normalized_dist_threshold: float = NORMALIZED_DIST_THRESHOLD,
                 iou_weight: float = IOU_WEIGHT,
                 dist_weight: float = DIST_WEIGHT):
        self.iou_accept = iou_accept
        self.sigma_multiplier = sigma_multiplier
        self.norm_thr = normalized_dist_threshold
        self.iou_weight = iou_weight
        self.dist_weight = dist_weight

    def score_candidates(self, entity: ConfusedEntity, kalman_trackers: Dict[int, Any]) -> List[CandidateScore]:
        out: List[CandidateScore] = []
        for cand_id in entity.possible_ids:
            if cand_id not in kalman_trackers:
                continue
            tracker_obj = kalman_trackers[cand_id]
            pred_bbox, expanded_bbox, mean_std = _bbox_from_tracker_prediction(tracker_obj, self.sigma_multiplier)

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
                reason = f"ioU_ok ({iou_score:.2f})"
            elif intersects_expanded:
                reason = f"intersects_expanded (std~{mean_std:.1f})"
            elif norm_dist <= self.norm_thr:
                reason = f"close_by_normdist ({norm_dist:.2f})"

            cs = CandidateScore(
                id=cand_id,
                score=score,
                iou=iou_score,
                norm_dist=norm_dist,
                intersects_expanded=intersects_expanded,
                pred_bbox=tuple(pred_bbox),
                expanded_bbox=tuple(expanded_bbox),
                mean_std=mean_std,
                reason=reason
            )
            out.append(cs)
        # sort descending by score for convenient use later
        out.sort(key=lambda x: x.score, reverse=True)
        return out


# ---- Example placeholder DL backend you can implement later ----
class DummyDLBackend:
    """
    Example skeleton for a deep-learning assisted backend.
    To use in practice:
      - compute embeddings for each track (e.g., appearance vector) and place them in kalman_trackers[tid].embedding
      - compute an embedding for the confused entity (crop image -> embed) and pass it as entity_meta
    This backend expects entity_meta={'embedding': [...]} and track objects with '.embedding' attribute.
    """
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # small, safe cosine similarity
        import math
        if a is None or b is None or len(a) == 0 or len(b) == 0:
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return -1.0
        return dot / (na * nb)

    def score_candidates(self, entity: ConfusedEntity, kalman_trackers: Dict[int, Any], entity_meta: Dict = None) -> List[CandidateScore]:
        embeddings = entity_meta.get("embedding") if entity_meta else None
        if embeddings is None:
            return []
        out = []
        for cand_id in entity.possible_ids:
            t = kalman_trackers.get(cand_id)
            if t is None or not hasattr(t, "embedding"):
                continue
            sim = self._cosine(embeddings, t.embedding)
            reason = f"dl_sim ({sim:.2f})" if sim >= self.similarity_threshold else None
            cs = CandidateScore(
                id=cand_id,
                score=sim,
                iou=0.0,
                norm_dist=999.0,
                intersects_expanded=False,
                pred_bbox=(0.0,0.0,0.0,0.0),
                expanded_bbox=(0.0,0.0,0.0,0.0),
                mean_std=0.0,
                reason=reason
            )
            out.append(cs)
        out.sort(key=lambda x: x.score, reverse=True)
        return out


# ---- Orchestrator: combine backends / produce resolved assignments ----
class ReIDManager:
    """
    Manages detection of confused groups and resolution using provided ReID backends.
    Usage:
      manager = ReIDManager(backends=[IoUKalmanBackend(), DummyDLBackend()])
      confused = manager.find_confused_groups(list_of_entities)
      resolved = manager.resolve(confused, kalman_trackers, entity_metas=dict_of_meta_if_any)
    Returns resolved list of tuples (best_id_or_None, entity_bbox, score_or_None, reason)
    """
    def __init__(self, backends: Optional[List[ReIDBackend]] = None,
                 overlap_threshold: float = OVERLAP_THRESHOLD,
                 midpoint_dist_threshold: float = MIDPOINT_DIST_THRESHOLD):
        self.backends = backends or [IoUKalmanBackend()]
        self.overlap_threshold = overlap_threshold
        self.midpoint_dist_threshold = midpoint_dist_threshold

    def find_confused_groups(self, entities: List[Any]) -> List[ConfusedEntity]:
        """
        Group by IoU > overlap_threshold or midpoints closer than midpoint_dist_threshold.
        Entities should have .bbox and optionally .possible_ids attributes.
        """
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
                # collect possible ids if tracks provided
                possible_ids = []
                for g in group:
                    if hasattr(g, "possible_ids"):
                        possible_ids += list(getattr(g, "possible_ids") or [])
                # union & unique preserving order
                possible_ids = list(dict.fromkeys(possible_ids))
                xs = [c.bbox[0] for c in group] + [c.bbox[2] for c in group]
                ys = [c.bbox[1] for c in group] + [c.bbox[3] for c in group]
                union_bbox = (min(xs), min(ys), max(xs), max(ys))
                confused_groups.append(ConfusedEntity(possible_ids, union_bbox))
        return confused_groups

    def resolve(self, confused_entities: List[ConfusedEntity],
                kalman_trackers: Dict[int, Any],
                entity_metas: Optional[Dict[Tuple[float,float,float,float], Dict]] = None
               ) -> List[Tuple[Optional[int], Tuple[float,float,float,float], Optional[float], str]]:
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
                # some backends accept entity_meta as 3rd arg (like DummyDLBackend)
                try:
                    scores = backend.score_candidates(ent, kalman_trackers)  # type: ignore
                except TypeError:
                    # fallback if backend expects (entity, trackers, meta)
                    scores = backend.score_candidates(ent, kalman_trackers, entity_metas.get(ent.bbox, None))  # type: ignore
                for cs in scores:
                    if cs.id not in aggregated or cs.score > aggregated[cs.id].score:
                        aggregated[cs.id] = cs

            # pick best with a reason (gated)
            candidates = list(aggregated.values())
            candidates.sort(key=lambda x: x.score, reverse=True)
            best = None
            for c in candidates:
                if c.reason is not None:
                    best = c
                    break

            if best is not None:
                results.append((best.id, ent.bbox, best.score, best.reason))
            else:
                results.append((None, ent.bbox, None, "no_candidate"))
        return results