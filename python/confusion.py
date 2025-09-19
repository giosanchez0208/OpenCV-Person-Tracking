import math

# Tunable params
OVERLAP_THRESHOLD = 0.5      # used for initial confusion grouping (IoU)
MIDPOINT_DIST_THRESHOLD = 60 # also group by proximity if IoU low but midpoint close
IOU_ACCEPT = 0.25            # accept match if IoU >= this
SIGMA_MULTIPLIER = 2.0       # expand predicted bbox by sigma * stddev for tolerant matching
NORMALIZED_DIST_THRESHOLD = 3.0  # accept if normalized distance <= this
# weight for combined score (higher favors IoU more)
IOU_WEIGHT = 1.0
DIST_WEIGHT = 1.0

class ConfusedEntity:
    def __init__(self, possible_ids, bbox):
        self.possible_ids = possible_ids  # list of candidate track ids
        self.bbox = bbox
        self.midpoint = calculate_midpoint(bbox)

    def get_label(self):
        return "? ".join(map(str, self.possible_ids)) + "?"

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0]); y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2]); y2 = min(bbox1[3], bbox2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = max(0, (bbox1[2] - bbox1[0])) * max(0, (bbox1[3] - bbox1[1]))
    a2 = max(0, (bbox2[2] - bbox2[0])) * max(0, (bbox2[3] - bbox2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def calculate_midpoint(bbox):
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def _bbox_from_tracker_prediction(kalman_obj, sigma_multiplier=SIGMA_MULTIPLIER):
    """
    Helper: return (pred_bbox, expanded_bbox, mean_std)
    Accepts:
      - a BBoxGroupKalmanTracker instance (has predict_without_updating & expanded_predicted_bbox)
      - a Track instance with .kalman (like earlier code)
      - a PointKalmanTracker (returns small bbox around point using std)
    This tries multiple attribute names to be robust to how your kalman_trackers dict is structured.
    """
    # BBoxGroupKalmanTracker or Track wrapper
    try:
        # Track wrapper: has .kalman
        if hasattr(kalman_obj, "kalman"):
            k = kalman_obj.kalman
        else:
            k = kalman_obj
        # If it has predict_without_updating and expanded_predicted_bbox, treat as bbox tracker
        if hasattr(k, "predict_without_updating") and hasattr(k, "expanded_predicted_bbox"):
            pred = k.predict_without_updating()
            expanded = k.expanded_predicted_bbox(sigma_multiplier)
            avg_sx, avg_sy = k.average_position_std() if hasattr(k, "average_position_std") else (1.0, 1.0)
            return pred, expanded, (avg_sx + avg_sy) / 2.0
    except Exception:
        pass

    # PointKalmanTracker-like fallback: use .predict() or .predict_float() for point, and P for std
    try:
        # If object is a PointKalmanTracker itself
        if hasattr(kalman_obj, "predict_without_updating") is False and hasattr(kalman_obj, "predict"):
            # call a non-committing prediction if available
            if hasattr(kalman_obj, "predict_float"):
                px, py = kalman_obj.predict_float()
            else:
                px, py = kalman_obj.predict()
            # std estimate if available
            if hasattr(kalman_obj, "position_std"):
                sx, sy = kalman_obj.position_std()
            else:
                sx, sy = (10.0, 10.0)
            # make a small bbox around the point and expanded bbox
            halfw = max(2, int(round(sx * sigma_multiplier)))
            halfh = max(2, int(round(sy * sigma_multiplier)))
            pred = (px - halfw, py - halfh, px + halfw, py + halfh)
            expanded = (px - 2*halfw, py - 2*halfh, px + 2*halfw, py + 2*halfh)
            return pred, expanded, (sx + sy) / 2.0
    except Exception:
        pass

    # If none of the above work, return a dummy wide bbox (so it won't match accidentally)
    return (0,0,0,0), (0,0,0,0), 1000.0

def detect_confusion(entities):
    """
    Group entities into confused_groups when they strongly overlap by IoU OR
    are very close by midpoint. Returns list of lists (groups of ConfusedEntity-like objects).
    Input 'entities' should be list of objects with .bbox attribute (like your original).
    """
    confused_groups = []
    processed = set()
    n = len(entities)
    for i, e1 in enumerate(entities):
        if i in processed:
            continue
        group = [e1]
        processed.add(i)
        for j in range(i+1, n):
            if j in processed:
                continue
            e2 = entities[j]
            if calculate_iou(e1.bbox, e2.bbox) > OVERLAP_THRESHOLD:
                group.append(e2)
                processed.add(j)
            else:
                # also group if midpoints are very near (handles small boxes / partial occlusions)
                m1 = calculate_midpoint(e1.bbox)
                m2 = calculate_midpoint(e2.bbox)
                if math.dist(m1, m2) <= MIDPOINT_DIST_THRESHOLD:
                    group.append(e2)
                    processed.add(j)
        if len(group) > 1:
            # build a ConfusedEntity containing all possible ids collated (if sources added possible_ids)
            # candidate possible_ids merging: keep union of .possible_ids if present, else empty list
            possible_ids = []
            for g in group:
                if hasattr(g, "possible_ids"):
                    possible_ids += list(g.possible_ids)
            possible_ids = list(dict.fromkeys(possible_ids))  # unique preserving order
            # create single composite ConfusedEntity locating at union bbox
            # union bbox:
            xs = [c.bbox[0] for c in group] + [c.bbox[2] for c in group]
            ys = [c.bbox[1] for c in group] + [c.bbox[3] for c in group]
            union_bbox = (min(xs), min(ys), max(xs), max(ys))
            confused_groups.append(ConfusedEntity(possible_ids, union_bbox))
    return confused_groups

def resolve_confusion(confused_entities, kalman_trackers,
                      iou_accept=IOU_ACCEPT,
                      sigma_multiplier=SIGMA_MULTIPLIER,
                      normalized_dist_threshold=NORMALIZED_DIST_THRESHOLD):
    """
    Resolve each confused entity against candidate kalman trackers.

    - kalman_trackers: dict mapping id -> kalman/track object
    Returns list of tuples (best_id, bbox, score, reason)
    """
    resolved = []
    for entity in confused_entities:
        # gather candidate predictions
        candidate_scores = []
        for cand_id in entity.possible_ids:
            if cand_id not in kalman_trackers:
                continue
            tracker_obj = kalman_trackers[cand_id]
            pred_bbox, expanded_bbox, mean_std = _bbox_from_tracker_prediction(tracker_obj, sigma_multiplier)

            # IoU between entity bbox and predicted bbox
            iou_score = calculate_iou(entity.bbox, pred_bbox)

            # normalized distance between midpoints (dist / (1 + mean_std))
            pred_mid = calculate_midpoint(pred_bbox)
            dist = math.dist(entity.midpoint, pred_mid)
            norm_dist = dist / (1.0 + mean_std)  # normalise by track uncertainty

            # boolean gating rules
            intersects_expanded = not (
                entity.bbox[2] < expanded_bbox[0] or entity.bbox[0] > expanded_bbox[2]
                or entity.bbox[3] < expanded_bbox[1] or entity.bbox[1] > expanded_bbox[3]
            )

            # combined score: higher = better
            score = IOU_WEIGHT * iou_score - DIST_WEIGHT * (norm_dist / (1.0 + norm_dist))

            reason = None
            # decide accept / reject
            if iou_score >= iou_accept:
                reason = f"ioU_ok ({iou_score:.2f})"
            elif intersects_expanded:
                reason = f"intersects_expanded (std~{mean_std:.1f})"
            elif norm_dist <= normalized_dist_threshold:
                reason = f"close_by_normdist ({norm_dist:.2f})"

            candidate_scores.append({
                "id": cand_id,
                "score": score,
                "iou": iou_score,
                "norm_dist": norm_dist,
                "intersects_expanded": intersects_expanded,
                "pred_bbox": pred_bbox,
                "expanded_bbox": expanded_bbox,
                "mean_std": mean_std,
                "reason": reason
            })

        # choose best candidate that passed any gate
        # sort by score descending to pick best if multiple pass gates
        candidate_scores.sort(key=lambda x: x["score"], reverse=True)
        best = None
        for c in candidate_scores:
            if c["reason"] is not None:
                best = c
                break

        if best is not None:
            resolved.append((best["id"], entity.bbox, best["score"], best["reason"]))
        else:
            # no candidate deemed acceptable; leave unresolved (could be new person)
            resolved.append((None, entity.bbox, None, "no_candidate"))

    return resolved
