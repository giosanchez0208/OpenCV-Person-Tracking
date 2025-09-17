import math
import numpy as np
from python.kalman import KalmanTracker
from python.confusion import ConfusedEntity, calculate_midpoint

DISTANCE_LIMIT = 1000
MAX_AGE = 20  # frames to keep a tracker without updates

class TrackedEntity:
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox  # (x1, y1, x2, y2, conf)
        self.midpoint = self.calculate_midpoint()
        self.age = 0
        self.hit_streak = 0
        self.kalman = KalmanTracker(self.midpoint)
        self.confused_with = []  # list of other entity IDs
        
    def calculate_midpoint(self):
        return calculate_midpoint(self.bbox)
    
    def predict_next_location(self, new_midpoints):
        # Use Kalman prediction
        predicted_pos = self.kalman.predict()
        
        best_match_idx = None
        min_dist = float("inf")

        for i, (nx, ny) in enumerate(new_midpoints):
            dist = math.dist(predicted_pos, (nx, ny))
            if dist < min_dist and dist <= DISTANCE_LIMIT:
                min_dist = dist
                best_match_idx = i

        return best_match_idx

    def update(self, new_bbox):
        self.bbox = new_bbox
        self.midpoint = self.calculate_midpoint()
        self.kalman.update(self.midpoint)
        self.age = 0
        self.hit_streak += 1
        self.confused_with = []  # clear confusion on update

    def increment_age(self):
        self.age += 1
        self.kalman.predict()  # keep predicting even without measurements
        return self.age
    
    def set_confusion(self, other_ids):
        self.confused_with = other_ids
    
    def get_label(self):
        if self.confused_with:
            all_ids = [self.id] + self.confused_with
            return "? ".join(map(str, sorted(all_ids))) + "?"
        return str(self.id)