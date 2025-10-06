# track.py
import time
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Any
from python.kalman import BBoxGroupKalmanTracker

# ReID / confusion
from python.confusion import calculate_midpoint, ReIDManager, IoUKalmanBackend, GalleryBackend

# gallery + features
from python.gallery import DetectionGallery
from python.features import MaskAwareFeatureExtractor, IdentityMemory, build_entity_metas, compute_crop_quality

# Tunables
DISTANCE_LIMIT = 1000
IOU_THRESHOLD = 0.3
MAX_AGE = 20
SIZE_CHANGE_THRESHOLD = 2.0

# re-id gating / gallery behavior
REID_SIM_THRESHOLD = 0.65
REID_TIME_WINDOW = 60.0
GALLERY_TOPK = 5
SNAPSHOT_HIT_STREAK = 3   # when to snapshot an embedding into gallery for confirmed tracks
DEDUPE_SIM = 0.995        # gallery dedupe threshold


class MultiObjectTracker:
    def __init__(self,
                 gallery_max_items: int = 5000,
                 extractor_device: str = "cpu",
                 embedding_dim: int = 128,
                 require_gallery_not_active: bool = True):
        self.tracked_entities = {}
        self.next_id = 0

        # Build gallery + extractor + memory and plug GalleryBackend into ReIDManager
        self.gallery = DetectionGallery(max_items=gallery_max_items)
        self.extractor = MaskAwareFeatureExtractor(device=extractor_device, full_extractor=None, out_dim=embedding_dim)
        self.identity_memory = IdentityMemory(dim=embedding_dim, momentum=0.95, buffer_size=5)

        gallery_backend = GalleryBackend(
            gallery=self.gallery,
            extractor=self.extractor,
            identity_memory=self.identity_memory,
            sim_threshold=REID_SIM_THRESHOLD,
            time_window=REID_TIME_WINDOW,
            topk=GALLERY_TOPK,
            require_not_active=require_gallery_not_active
        )

        # create a ReIDManager with geometric heuristic + gallery backend
        self.reid_manager = ReIDManager(backends=[IoUKalmanBackend(), gallery_backend])

    def calculate_iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        if x2_int < x1_int or y2_int < y1_int:
            return 0.0
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def build_cost_matrix(self, tracked_entities, new_detections):
        n_tracks = len(tracked_entities)
        n_detections = len(new_detections)
        cost_matrix = np.full((n_tracks, n_detections), 1000.0)
        for i, entity in enumerate(tracked_entities):
            predicted_bbox = entity.kalman.predict()
            predicted_midpoint = (
                (predicted_bbox[0] + predicted_bbox[2]) / 2,
                (predicted_bbox[1] + predicted_bbox[3]) / 2
            )
            entity_area = entity.get_bbox_area()
            for j, new_bbox in enumerate(new_detections):
                iou = self.calculate_iou(predicted_bbox, new_bbox)
                new_midpoint = (
                    (new_bbox[0] + new_bbox[2]) / 2,
                    (new_bbox[1] + new_bbox[3]) / 2
                )
                distance = math.dist(predicted_midpoint, new_midpoint)
                new_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
                size_ratio = max(entity_area, new_area) / (min(entity_area, new_area) + 1e-6)
                cost = 0
                cost += -iou * 100
                cost += distance / 10
                if size_ratio > SIZE_CHANGE_THRESHOLD:
                    cost += (size_ratio - SIZE_CHANGE_THRESHOLD) * 50
                cost -= min(entity.hit_streak, 10) * 2
                if distance > DISTANCE_LIMIT or (iou < IOU_THRESHOLD and distance > 200):
                    cost = 1000.0
                cost_matrix[i, j] = cost
        return cost_matrix

    def update(self, new_detections, frame=None, det_confs: Optional[List[float]] = None, det_masks: Optional[List[Any]] = None):
        """
        new_detections: list of bbox tuples (x1,y1,x2,y2)
        frame: optional full-frame numpy image used to compute embeddings / entity_metas
        det_confs: optional list of detection confidences aligned to new_detections
        det_masks: optional list of masks aligned to new_detections
        """
        if det_confs is None:
            det_confs = [None] * len(new_detections)
        if det_masks is None:
            det_masks = [None] * len(new_detections)

        # handle empty detection list: age & delete old tracks (snapshot reps into gallery)
        if not new_detections:
            for entity_id in list(self.tracked_entities.keys()):
                ent = self.tracked_entities[entity_id]
                if ent.increment_age() > MAX_AGE:
                    # snapshot representative embedding (preferred) then delete
                    rep = self.identity_memory.get(ent.id)
                    if rep is not None:
                        try:
                            self.gallery.add(key=ent.id, embedding=rep,
                                             meta={'bbox': ent.bbox, 'last_seen': time.time()},
                                             dedupe_threshold=DEDUPE_SIM, replace_similar=False)
                        except Exception:
                            pass
                    del self.tracked_entities[entity_id]
            return

        # if no existing tracks, create tracks for all detections (but attempt gallery re-id for each first)
        if not self.tracked_entities:
            for i, bbox in enumerate(new_detections):
                resumed = False
                if frame is not None:
                    emb = self._compute_embedding_for_bbox(frame, bbox)
                    if emb is not None:
                        # query gallery immediately for resume
                        candidates = self.gallery.query(emb, topk=GALLERY_TOPK,
                                                        min_similarity=REID_SIM_THRESHOLD,
                                                        filter_fn=(lambda m, k: True))
                        for cand_key, sim, cand_meta in candidates:
                            # require not currently active
                            if cand_key in self.tracked_entities:
                                continue
                            # require recency
                            last_seen = cand_meta.get('last_seen', 0)
                            if (time.time() - last_seen) > REID_TIME_WINDOW:
                                continue
                            # accept candidate -> resume
                            te = TrackedEntity(cand_key, bbox)
                            te.embedding = emb
                            te.kalman = BBoxGroupKalmanTracker(bbox[:4])
                            self.tracked_entities[cand_key] = te
                            self.next_id = max(self.next_id, cand_key + 1)
                            # update identity memory with this new sighting
                            q = compute_crop_quality(frame, (int(bbox[0]), int(bbox[1]),
                                                             int(max(0, bbox[2] - bbox[0])),
                                                             int(max(0, bbox[3] - bbox[1]))),
                                                     conf=det_confs[i], mask=det_masks[i])
                            self.identity_memory.update(cand_key, emb, timestamp=time.time(), quality=q)
                            resumed = True
                            break
                if not resumed:
                    # plain new id
                    te = TrackedEntity(self.next_id, bbox)
                    if frame is not None:
                        emb2 = self._compute_embedding_for_bbox(frame, bbox)
                        if emb2 is not None:
                            q2 = compute_crop_quality(frame, (int(bbox[0]), int(bbox[1]),
                                                              int(max(0, bbox[2] - bbox[0])),
                                                              int(max(0, bbox[3] - bbox[1]))),
                                                      conf=det_confs[i], mask=det_masks[i])
                            self.identity_memory.update(te.id, emb2, timestamp=time.time(), quality=q2)
                            te.embedding = self.identity_memory.get(te.id)
                    self.tracked_entities[self.next_id] = te
                    self.next_id += 1
            return

        # run data association between existing tracks and detections
        tracked_list = list(self.tracked_entities.values())
        cost_matrix = self.build_cost_matrix(tracked_list, new_detections)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_detections = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 900:
                entity = tracked_list[row]
                entity.update(new_detections[col])
                matched_tracks.add(entity.id)
                matched_detections.add(col)
                # update embedding for matched track if frame is available
                if frame is not None:
                    emb = self._compute_embedding_for_bbox(frame, entity.bbox)
                    if emb is not None:
                        conf = det_confs[col] if col < len(det_confs) else None
                        mask = det_masks[col] if col < len(det_masks) else None
                        bbox_xywh = (int(round(entity.bbox[0])), int(round(entity.bbox[1])),
                                     int(round(max(0, entity.bbox[2] - entity.bbox[0]))),
                                     int(round(max(0, entity.bbox[3] - entity.bbox[1]))))
                        q = compute_crop_quality(frame, bbox_xywh, conf=conf, mask=mask)
                        self.identity_memory.update(entity.id, emb, timestamp=time.time(), quality=q)
                        entity.embedding = self.identity_memory.get(entity.id)
                        # snapshot to gallery when stable
                        if entity.hit_streak >= SNAPSHOT_HIT_STREAK:
                            rep = self.identity_memory.get(entity.id)
                            if rep is not None:
                                try:
                                    self.gallery.add(key=entity.id, embedding=rep,
                                                     meta={'bbox': entity.bbox, 'last_seen': time.time()},
                                                     dedupe_threshold=DEDUPE_SIM, replace_similar=False)
                                except Exception:
                                    pass

        # age & delete unmatched tracks (snapshot rep before deletion)
        for entity in tracked_list:
            if entity.id not in matched_tracks:
                if entity.increment_age() > MAX_AGE:
                    ent = self.tracked_entities.get(entity.id)
                    if ent is not None:
                        rep = self.identity_memory.get(ent.id)
                        if rep is not None:
                            try:
                                self.gallery.add(key=ent.id, embedding=rep,
                                                 meta={'bbox': ent.bbox, 'last_seen': time.time()},
                                                 dedupe_threshold=DEDUPE_SIM, replace_similar=False)
                            except Exception:
                                pass
                    try:
                        del self.tracked_entities[entity.id]
                    except KeyError:
                        pass

        # For unmatched detections: query gallery BEFORE creating brand-new ids
        for j, bbox in enumerate(new_detections):
            if j in matched_detections:
                continue

            resumed = False
            if frame is not None:
                emb = self._compute_embedding_for_bbox(frame, bbox)
                if emb is not None:
                    # filter_fn: exclude keys that are currently active if backend wants that behavior
                    def _filter(meta, key):
                        if key in self.tracked_entities:
                            return False
                        # recency check will be applied below as well
                        return True

                    candidates = self.gallery.query(emb, topk=GALLERY_TOPK,
                                                    min_similarity=REID_SIM_THRESHOLD,
                                                    filter_fn=_filter if True else None)
                    for cand_key, sim, cand_meta in candidates:
                        # enforce recency
                        last_seen = cand_meta.get('last_seen', 0)
                        if (time.time() - last_seen) > REID_TIME_WINDOW:
                            continue
                        # if candidate is not active, resume it
                        if cand_key not in self.tracked_entities:
                            te = TrackedEntity(cand_key, bbox)
                            te.embedding = emb
                            te.kalman = BBoxGroupKalmanTracker(bbox[:4])
                            self.tracked_entities[cand_key] = te
                            self.next_id = max(self.next_id, cand_key + 1)
                            # update identity memory with this new sighting
                            bbox_xywh = (int(round(bbox[0])), int(round(bbox[1])),
                                         int(round(max(0, bbox[2] - bbox[0]))),
                                         int(round(max(0, bbox[3] - bbox[1]))))
                            q = compute_crop_quality(frame, bbox_xywh, conf=det_confs[j], mask=det_masks[j])
                            self.identity_memory.update(cand_key, emb, timestamp=time.time(), quality=q)
                            resumed = True
                            break

            if not resumed:
                # fallback create brand-new id
                te = TrackedEntity(self.next_id, bbox)
                if frame is not None:
                    emb2 = self._compute_embedding_for_bbox(frame, bbox)
                    if emb2 is not None:
                        bbox_xywh = (int(round(bbox[0])), int(round(bbox[1])),
                                     int(round(max(0, bbox[2] - bbox[0]))),
                                     int(round(max(0, bbox[3] - bbox[1]))))
                        q2 = compute_crop_quality(frame, bbox_xywh, conf=det_confs[j], mask=det_masks[j])
                        self.identity_memory.update(te.id, emb2, timestamp=time.time(), quality=q2)
                        te.embedding = self.identity_memory.get(te.id)
                self.tracked_entities[self.next_id] = te
                self.next_id += 1

        # run confusion detection/resolution (unchanged)
        self.detect_confusion(frame=frame)

    def detect_confusion(self, frame=None):
        entities = list(self.tracked_entities.values())
        holders = []
        for e in entities:
            h = type("H", (), {})()
            h.bbox = e.bbox
            h.possible_ids = [e.id]
            holders.append(h)

        confused_groups = self.reid_manager.find_confused_groups(holders)

        # Build previous_features_map for build_entity_metas if tracks have a history
        previous_features_map = {}
        for t in entities:
            if getattr(t, "embedding", None) is not None:
                previous_features_map[t.id] = [t.embedding]

        entity_metas = {}
        if frame is not None and len(confused_groups) > 0:
            try:
                entity_metas = build_entity_metas(frame, confused_groups, self.extractor,
                                                  masks=None, previous_features_map=previous_features_map)
            except Exception:
                entity_metas = {}

        resolved = self.reid_manager.resolve(confused_groups, {t.id: t for t in entities}, entity_metas)

        # reset all confusion
        for t in entities:
            t.set_confusion([])

        for best_id, bbox, score, reason in resolved:
            if best_id is None:
                for t in entities:
                    if self.calculate_iou(t.bbox, bbox) > 0.1:
                        others = [o.id for o in entities if o.id != t.id and self.calculate_iou(o.bbox, bbox) > 0.1]
                        if others:
                            t.set_confusion(sorted(list(set(t.confused_with + others))))
            else:
                for t in entities:
                    if t.id != best_id and self.calculate_iou(t.bbox, bbox) > 0.1:
                        t.set_confusion(sorted(list(set(t.confused_with + [best_id]))))

    def _compute_embedding_for_bbox(self, frame, bbox_xyxy):
        try:
            x1, y1, x2, y2 = bbox_xyxy[:4]
            x = int(round(x1)); y = int(round(y1))
            w = int(round(max(0, x2 - x1))); h = int(round(max(0, y2 - y1)))
            bbox_xywh = (x, y, w, h)
            emb = self.extractor.extract_features(frame, bbox_xywh, mask=None, use_augmentation=False)
            if emb is None:
                return None
            arr = np.array(emb, dtype='float32')
            n = np.linalg.norm(arr) + 1e-7
            arr = arr / n
            return arr
        except Exception:
            return None

    def get_tracked_entities(self):
        return list(self.tracked_entities.values())


class TrackedEntity:
    """Individual tracked entity with Kalman filter."""
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox
        self.midpoint = self.calculate_midpoint()
        self.age = 0
        self.hit_streak = 0
        self.kalman = BBoxGroupKalmanTracker(self.bbox[:4])
        self.confused_with = []
        self.size_history = [self.get_bbox_area()]
        self.embedding = None  # representative embedding (set via IdentityMemory)

    def calculate_midpoint(self):
        return calculate_midpoint(self.bbox)

    def get_bbox_area(self):
        x1, y1, x2, y2 = self.bbox[:4]
        return abs((x2 - x1) * (y2 - y1))

    def update(self, new_bbox):
        self.bbox = new_bbox
        self.midpoint = self.calculate_midpoint()
        try:
            self.kalman.update(new_bbox[:4])
        except Exception:
            try:
                self.kalman.update(tuple(new_bbox[:4]))
            except Exception:
                pass
        self.age = 0
        self.hit_streak += 1
        self.confused_with = []
        self.size_history.append(self.get_bbox_area())
        if len(self.size_history) > 10:
            self.size_history.pop(0)

    def increment_age(self):
        self.age += 1
        self.hit_streak = 0
        try:
            predicted_bbox = self.kalman.predict()
            self.bbox = tuple(predicted_bbox) + tuple(self.bbox[4:]) if len(self.bbox) > 4 else tuple(predicted_bbox)
        except Exception:
            pass
        self.midpoint = self.calculate_midpoint()
        return self.age

    def set_confusion(self, other_ids):
        self.confused_with = other_ids

    def get_label(self):
        if self.confused_with:
            all_ids = [self.id] + self.confused_with
            return "? ".join(map(str, sorted(all_ids))) + "?"
        return str(self.id)
