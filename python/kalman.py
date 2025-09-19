import numpy as np
import itertools

# ---------------------------
# Utility functions
# ---------------------------

def iou(boxA, boxB):
    """
    Compute IoU between two boxes.
    box = (x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter
    if union == 0:
        return 0.0
    return inter / union

def expand_bbox(bbox, dx, dy):
    x1, y1, x2, y2 = bbox
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

# ---------------------------
# Small extensions to your PointKalmanTracker
# ---------------------------

class PointKalmanTracker:
    def __init__(self, initial_pos):
        # state: [x, y, vx, vy, ax, ay]
        self.x = np.array([[initial_pos[0]], [initial_pos[1]], [0], [0], [0], [0]], dtype=float)

        # state transition matrix: predicts next state from current state.
        self.F = np.array([[1, 0, 1, 0, 0.5, 0],
                           [0, 1, 0, 1, 0, 0.5],
                           [0, 0, 1, 0, 1, 0],
                           [0, 0, 0, 1, 0, 1],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]], dtype=float)

        # measurement matrix: maps state to measurement space.
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=float)

        # process covariance: uncertainty in the model's prediction.
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5, 1.0, 1.0])

        # measurement covariance: uncertainty in the measurements.
        self.R = np.eye(2) * 2.0

        # error covariance: uncertainty in the state estimate.
        self.P = np.eye(6) * 100

    def predict(self):
        """
        Project the state and its uncertainty forward in time.
        Returns integer (x, y) for convenience.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (int(self.x[0, 0]), int(self.x[1, 0]))

    def predict_float(self):
        """
        Same as predict but returns floats and leaves state updated.
        Useful if you need subpixel uncertainty analysis.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return (float(self.x[0, 0]), float(self.x[1, 0]))

    def update(self, measurement):
        z = np.array([[measurement[0]], [measurement[1]]], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_velocity(self):
        return (self.x[2, 0], self.x[3, 0])

    def get_acceleration(self):
        return (self.x[4, 0], self.x[5, 0])

    def position_covariance(self):
        """
        Return 2x2 covariance of (x,y) extracted from P.
        This is the uncertainty used to compute disappearance leeway.
        """
        return self.P[0:2, 0:2]

    def position_std(self):
        """
        Return (std_x, std_y) from position covariance (sqrt of diag).
        """
        cov = self.position_covariance()
        sx = float(np.sqrt(max(0.0, cov[0, 0])))
        sy = float(np.sqrt(max(0.0, cov[1, 1])))
        return sx, sy

# ---------------------------
# BBoxGroupKalmanTracker with uncertainty helpers
# ---------------------------

class BBoxGroupKalmanTracker:
    def __init__(self, initial_bbox):
        """
        Initializes four Kalman trackers, one for each corner of the bounding box.
        initial_bbox: (x1, y1, x2, y2) -> top-left, bottom-right
        """
        x1, y1, x2, y2 = initial_bbox
        self.trackers = {
            "tl": PointKalmanTracker((x1, y1)),  # top-left
            "tr": PointKalmanTracker((x2, y1)),  # top-right
            "bl": PointKalmanTracker((x1, y2)),  # bottom-left
            "br": PointKalmanTracker((x2, y2)),  # bottom-right
        }

    def predict(self):
        """
        Predict the next positions of all four corners.
        Returns bbox in (x1, y1, x2, y2) format.
        (This advances each corner's internal state.)
        """
        tl = self.trackers["tl"].predict()
        tr = self.trackers["tr"].predict()
        bl = self.trackers["bl"].predict()
        br = self.trackers["br"].predict()

        # reconstruct bbox: top-left and bottom-right
        x1 = int(min(tl[0], bl[0]))
        y1 = int(min(tl[1], tr[1]))
        x2 = int(max(tr[0], br[0]))
        y2 = int(max(bl[1], br[1]))

        return (x1, y1, x2, y2)

    def predict_without_updating(self):
        """
        Utility: get predicted float positions without modifying internal state.
        Useful if calling many match hypotheses without committing.
        (We do a temporary math predict here — note: simpler than full Kalman predict replication.)
        """
        # For simplicity, use the current state vector transformed by F to simulate one-step predict
        predicted = {}
        for k, t in self.trackers.items():
            x_pred = t.F @ t.x
            predicted[k] = (float(x_pred[0, 0]), float(x_pred[1, 0]))
        x1 = min(predicted["tl"][0], predicted["bl"][0])
        y1 = min(predicted["tl"][1], predicted["tr"][1])
        x2 = max(predicted["tr"][0], predicted["br"][0])
        y2 = max(predicted["bl"][1], predicted["br"][1])
        return (x1, y1, x2, y2)

    def update(self, new_bbox):
        """
        Update all four corner trackers with the new bbox measurement.
        new_bbox: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = new_bbox
        self.trackers["tl"].update((x1, y1))
        self.trackers["tr"].update((x2, y1))
        self.trackers["bl"].update((x1, y2))
        self.trackers["br"].update((x2, y2))

    def get_bbox(self):
        """
        Return the current best estimate of the bbox
        based on the four tracked corners.
        """
        tl = self.trackers["tl"].x
        tr = self.trackers["tr"].x
        bl = self.trackers["bl"].x
        br = self.trackers["br"].x

        x1 = int(min(tl[0, 0], bl[0, 0]))
        y1 = int(min(tl[1, 0], tr[1, 0]))
        x2 = int(max(tr[0, 0], br[0, 0]))
        y2 = int(max(bl[1, 0], br[1, 0]))

        return (x1, y1, x2, y2)

    def corner_position_stds(self):
        """
        Return dict of stds for each corner: { 'tl': (sx, sy), ... }
        """
        return {k: t.position_std() for k, t in self.trackers.items()}

    def average_position_std(self):
        """
        Average of corner stddevs -> single (sx, sy) representing bbox uncertainty.
        """
        sxs = []
        sys = []
        for sx, sy in self.corner_position_stds().values():
            sxs.append(sx); sys.append(sy)
        if not sxs:
            return (1.0, 1.0)
        return (float(np.mean(sxs)), float(np.mean(sys)))

    def expanded_predicted_bbox(self, sigma_multiplier=2.0):
        """
        Return predicted bbox expanded by sigma_multiplier * average stddev.
        Useful to represent disappearance leeway.
        """
        pred = self.predict_without_updating()
        avg_sx, avg_sy = self.average_position_std()
        # Convert std in state-units to pixels; std already in pixels because state stores positions
        dx = max(1, int(round(sigma_multiplier * avg_sx)))
        dy = max(1, int(round(sigma_multiplier * avg_sy)))
        return expand_bbox(pred, dx, dy)

    def get_velocities(self):
        """Return velocities for all four corners."""
        return {k: t.get_velocity() for k, t in self.trackers.items()}

    def get_accelerations(self):
        """Return accelerations for all four corners."""
        return {k: t.get_acceleration() for k, t in self.trackers.items()}

# ---------------------------
# Track container and matching logic
# ---------------------------

class Track:
    """
    Container for a tracked object: holds an id, the BBoxGroupKalmanTracker,
    and bookkeeping fields like unseen_frames.
    """
    _next_id = 0
    def __init__(self, initial_bbox):
        self.id = Track._next_id
        Track._next_id += 1
        self.kalman = BBoxGroupKalmanTracker(initial_bbox)
        self.unseen_frames = 0  # consecutive frames without detection
        self.age = 0  # total frames since creation

    def predict(self):
        self.age += 1
        return self.kalman.predict()

    def update(self, bbox):
        self.unseen_frames = 0
        self.kalman.update(bbox)

    def mark_missed(self):
        self.unseen_frames += 1

    def bbox(self):
        return self.kalman.get_bbox()

def match_detections_to_tracks(detections, tracks,
                               iou_threshold=0.3,
                               sigma_multiplier=2.0,
                               max_unseen=5):
    """
    Greedy matching between detection boxes and track predicted boxes.
    - detections: list of (x1,y1,x2,y2)
    - tracks: list of Track objects (already predicted for this frame)
    Returns (matches, unmatched_dets, unmatched_tracks) where matches is list of (det_idx, track_idx)
    Behavior:
      - Primary score: IoU between detection and predicted bbox
      - Additional acceptance: if detection intersects track's expanded predicted bbox
        (expanded via sigma_multiplier * mean stddev extracted from Kalman P)
    """
    # Build predicted boxes and expanded boxes
    preds = [t.kalman.predict_without_updating() for t in tracks]
    expanded = [tracks[i].kalman.expanded_predicted_bbox(sigma_multiplier) for i in range(len(tracks))]

    # Compute candidate pairs and IoU
    pairs = []
    for di, det in enumerate(detections):
        for ti, pred in enumerate(preds):
            score = iou(det, pred)
            intersects_expanded = (
                not (det[2] < expanded[ti][0] or det[0] > expanded[ti][2] or det[3] < expanded[ti][1] or det[1] > expanded[ti][3])
            )
            # Accept candidate if IoU >= small positive or intersects expanded region
            if score >= 0.0 or intersects_expanded:
                # We use IoU as primary rank; but prefer intersects_expanded even when IoU low
                # We'll compute an augmented score that slightly boosts intersecting expanded regions
                augmented = score + (0.1 if intersects_expanded else 0.0)
                pairs.append((augmented, di, ti))

    # Greedy assign: sort descending by augmented score
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_dets = set()
    matched_tracks = set()
    matches = []

    for score, di, ti in pairs:
        if di in matched_dets or ti in matched_tracks:
            continue
        # final acceptance decision: require IoU >= threshold OR intersects expanded
        pred = preds[ti]
        det = detections[di]
        primary_iou = iou(det, pred)
        intersects_expanded = not (det[2] < expanded[ti][0] or det[0] > expanded[ti][2] or det[3] < expanded[ti][1] or det[1] > expanded[ti][3])
        if primary_iou >= iou_threshold or intersects_expanded:
            matches.append((di, ti))
            matched_dets.add(di)
            matched_tracks.add(ti)

    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    return matches, unmatched_dets, unmatched_tracks

def tracking_step(detections, track_list,
                  iou_threshold=0.3,
                  sigma_multiplier=2.0,
                  max_unseen=5):
    """
    Full per-frame pipeline:
     - 1) Predict all tracks (commit their internal predict)
     - 2) Match detections <-> tracks
     - 3) Update matched tracks; create new tracks for unmatched detections
     - 4) Mark missed tracks and remove old ones
    Returns updated track_list
    """
    # 1) Predict (advance state)
    for t in track_list:
        t.predict()

    # 2) Match
    matches, unmatched_dets, unmatched_tracks = match_detections_to_tracks(
        detections, track_list, iou_threshold=iou_threshold,
        sigma_multiplier=sigma_multiplier, max_unseen=max_unseen
    )

    # 3) Update matched tracks
    for di, ti in matches:
        bbox = detections[di]
        track_list[ti].update(bbox)

    # 4) Unmatched detections -> new tracks
    for di in unmatched_dets:
        new_track = Track(detections[di])
        track_list.append(new_track)

    # 5) Unmatched tracks -> mark missed and maybe delete
    survivors = []
    for i, t in enumerate(track_list):
        if i in [ti for _, ti in matches]:
            survivors.append(t)
        else:
            t.mark_missed()
            if t.unseen_frames <= max_unseen:
                survivors.append(t)
            # else drop the track (garbage collect)

    return survivors

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # initial tracks
    tracks = [Track((100, 100, 160, 200))]

    # frame loop (pseudo)
    # detections is a list of bboxes detected this frame
    detections = [(102, 98, 162, 202), (300, 300, 350, 370)]
    tracks = tracking_step(detections, tracks, iou_threshold=0.35, sigma_multiplier=2.5, max_unseen=7)

    for t in tracks:
        print("Track", t.id, "bbox", t.bbox(), "unseen", t.unseen_frames)
