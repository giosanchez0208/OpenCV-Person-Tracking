# python/kalman.py
from dataclasses import dataclass
import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Optional

def bbox_to_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy

@dataclass
class SingleKalman:
    id: int
    kf: cv2.KalmanFilter
    w: float
    h: float
    last_update: float
    time_since_update: int = 0
    hits: int = 0

    @classmethod
    def create(cls, initial_bbox: List[float], id: int, dt: float = 1.0):
        # state: [cx, cy, vx, vy] (4), measurement: [cx, cy] (2)
        kf = cv2.KalmanFilter(4, 2)
        # Transition matrix for constant velocity
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Tunable covariances
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        cx, cy = bbox_to_center(initial_bbox)
        # initial state (cx, cy, vx=0, vy=0)
        kf.statePost = np.array([[cx], [cy], [0.], [0.]], dtype=np.float32)

        w = initial_bbox[2] - initial_bbox[0]
        h = initial_bbox[3] - initial_bbox[1]

        return cls(id=id, kf=kf, w=w, h=h, last_update=time.time(), time_since_update=0, hits=1)

    def predict(self) -> List[float]:
        state = self.kf.predict()
        cx = float(state[0])
        cy = float(state[1])
        # produce bbox centered at predicted center using last known w,h
        x1 = cx - self.w / 2.0
        y1 = cy - self.h / 2.0
        x2 = cx + self.w / 2.0
        y2 = cy + self.h / 2.0
        # increment time since update; will be reset on update()
        self.time_since_update += 1
        return [x1, y1, x2, y2]

    def update(self, measured_bbox: List[float]):
        cx, cy = bbox_to_center(measured_bbox)
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kf.correct(measurement)
        # update size estimate with a small running update (simple)
        new_w = measured_bbox[2] - measured_bbox[0]
        new_h = measured_bbox[3] - measured_bbox[1]
        # smoothing the width/height so small jitter doesn't flip size wildly
        alpha = 0.6
        self.w = alpha * new_w + (1 - alpha) * self.w
        self.h = alpha * new_h + (1 - alpha) * self.h

        self.last_update = time.time()
        self.time_since_update = 0
        self.hits += 1

class KalmanManager:
    def __init__(self, dist_threshold: float = 100.0, max_age: int = 30, dt: float = 1.0):
        """
        dist_threshold: maximum pixel distance between predicted center and detection center to consider a match
        max_age: number of frames to keep a tracker without being updated
        """
        self.trackers: List[SingleKalman] = []
        self.next_id: int = 1
        self.dist_threshold = dist_threshold
        self.max_age = max_age
        self.dt = dt

    def predict_all(self) -> List[Dict]:
        """Predict positions for all trackers, return list of dicts {id, bbox}"""
        preds = []
        for t in self.trackers:
            pred_bbox = t.predict()
            preds.append({"id": t.id, "bbox": pred_bbox})
        return preds

    def _center(self, bbox: List[float]) -> Tuple[float, float]:
        return bbox_to_center(bbox)

    def update(self, detections: List[List[float]]) -> List[Dict]:
        """
        detections: list of bboxes [x1,y1,x2,y2]
        returns: list of dicts { 'id': int, 'bbox': [x1,y1,x2,y2] }
        """
        # 1) predict step
        preds = [t.predict() for t in self.trackers]
        # prepare centers
        pred_centers = [self._center(b) for b in preds]
        det_centers = [self._center(d) for d in detections]

        # 2) compute all pairwise distances
        pairs = []
        for ti, pc in enumerate(pred_centers):
            for di, dc in enumerate(det_centers):
                dist = np.hypot(pc[0] - dc[0], pc[1] - dc[1])
                pairs.append((dist, ti, di))
        # sort ascending distances
        pairs.sort(key=lambda x: x[0])

        assigned_t = set()
        assigned_d = set()
        matches = []  # list of (tidx, didx)

        for dist, ti, di in pairs:
            if dist > self.dist_threshold:
                break
            if ti in assigned_t or di in assigned_d:
                continue
            assigned_t.add(ti)
            assigned_d.add(di)
            matches.append((ti, di))

        # 3) update matched trackers
        for ti, di in matches:
            tracker = self.trackers[ti]
            tracker.update(detections[di])

        # 4) create trackers for unmatched detections
        unmatched_dets = [i for i in range(len(detections)) if i not in assigned_d]
        for di in unmatched_dets:
            tr = SingleKalman.create(detections[di], id=self.next_id, dt=self.dt)
            self.trackers.append(tr)
            self.next_id += 1

        # 5) increase time_since_update for unmatched trackers (already incremented in predict),
        # remove old trackers
        to_keep = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                to_keep.append(t)
        self.trackers = to_keep

        # 6) prepare output list combining the latest bbox estimate and id
        out = []
        for t in self.trackers:
            # reconstruct bbox from current state (predict once to get center)
            state = t.kf.statePost
            cx = float(state[0])
            cy = float(state[1])
            x1 = cx - t.w/2.0
            y1 = cy - t.h/2.0
            x2 = cx + t.w/2.0
            y2 = c
