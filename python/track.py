# track.py
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from python.kalman import BBoxGroupKalmanTracker
# import the new ReID manager / backend
from python.confusion import calculate_midpoint, ReIDManager, IoUKalmanBackend

DISTANCE_LIMIT = 1000
IOU_THRESHOLD = 0.3
MAX_AGE = 20
SIZE_CHANGE_THRESHOLD = 2.0

class MultiObjectTracker:
    """
    Manages multiple TrackedEntity objects and performs optimal assignment
    using the Hungarian algorithm.
    """
    
    def __init__(self):
        self.tracked_entities = {}
        self.next_id = 0
        # create a ReIDManager with the heuristic backend by default.
        # Later you can pass a DL backend in this list.
        self.reid_manager = ReIDManager(backends=[IoUKalmanBackend()])

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

    def update(self, new_detections):
        if not new_detections:
            for entity_id in list(self.tracked_entities.keys()):
                if self.tracked_entities[entity_id].increment_age() > MAX_AGE:
                    del self.tracked_entities[entity_id]
            return

        if not self.tracked_entities:
            for bbox in new_detections:
                self.tracked_entities[self.next_id] = TrackedEntity(self.next_id, bbox)
                self.next_id += 1
            return

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

        for entity in tracked_list:
            if entity.id not in matched_tracks:
                if entity.increment_age() > MAX_AGE:
                    del self.tracked_entities[entity.id]

        for j, bbox in enumerate(new_detections):
            if j not in matched_detections:
                self.tracked_entities[self.next_id] = TrackedEntity(self.next_id, bbox)
                self.next_id += 1

        # Use the ReIDManager to detect+resolve confusion among current tracks.
        self.detect_confusion()

    def detect_confusion(self):
        """
        Use ReIDManager to find confused groups and resolve them.
        The manager returns assignments (best_id or None), we use that to set per-track
        confused_with lists so UI/labels can show the confusion state.
        """
        entities = list(self.tracked_entities.values())
        # first create simple holder objects with .bbox and .possible_ids for grouping
        holders = []
        for e in entities:
            h = type("H", (), {})()
            h.bbox = e.bbox
            # possible_ids for grouping resolution: include the track itself and any confused_with from older logic
            h.possible_ids = [e.id]
            holders.append(h)

        confused_groups = self.reid_manager.find_confused_groups(holders)
        # optional: you can provide entity-specific metadata (e.g., crops / embeddings) here:
        entity_metas = {}  # e.g., {group.bbox: {"embedding": ...}}

        resolved = self.reid_manager.resolve(confused_groups, {t.id: t for t in entities}, entity_metas)

        # reset all confusion
        for t in entities:
            t.set_confusion([])

        # set confusion lists according to resolution results
        for best_id, bbox, score, reason in resolved:
            if best_id is None:
                # no candidate: mark all tracks overlapping union bbox as confused (conservative)
                for t in entities:
                    if self.calculate_iou(t.bbox, bbox) > 0.1:
                        # mark t as confused with anyone else overlapping
                        others = [o.id for o in entities if o.id != t.id and self.calculate_iou(o.bbox, bbox) > 0.1]
                        if others:
                            t.set_confusion(sorted(list(set(t.confused_with + others))))
            else:
                # mark track best_id as not confused, but other tracks that overlap union bbox as confused_with best_id
                for t in entities:
                    if t.id != best_id and self.calculate_iou(t.bbox, bbox) > 0.1:
                        # t is confused with best_id
                        t.set_confusion(sorted(list(set(t.confused_with + [best_id]))))

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
        
    def calculate_midpoint(self):
        return calculate_midpoint(self.bbox)
    
    def get_bbox_area(self):
        x1, y1, x2, y2 = self.bbox[:4]
        return abs((x2 - x1) * (y2 - y1))
    
    def update(self, new_bbox):
        self.bbox = new_bbox
        self.midpoint = self.calculate_midpoint()
        self.kalman.update(new_bbox[:4])
        self.age = 0
        self.hit_streak += 1
        self.confused_with = []
        self.size_history.append(self.get_bbox_area())
        if len(self.size_history) > 10:
            self.size_history.pop(0)
    
    def increment_age(self):
        self.age += 1
        self.hit_streak = 0
        predicted_bbox = self.kalman.predict()
        # attempt to preserve trailing fields if present
        try:
            self.bbox = tuple(predicted_bbox) + tuple(self.bbox[4:])
        except Exception:
            self.bbox = tuple(predicted_bbox)
        self.midpoint = self.calculate_midpoint()
        return self.age
    
    def set_confusion(self, other_ids):
        self.confused_with = other_ids
    
    def get_label(self):
        if self.confused_with:
            all_ids = [self.id] + self.confused_with
            return "? ".join(map(str, sorted(all_ids))) + "?"
        return str(self.id)
