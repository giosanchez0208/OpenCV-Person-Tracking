import math
import numpy as np

DISTANCE_LIMIT = 50
MAX_AGE = 5  # frames to keep a tracker without updates

class TrackedEntity:
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox  # Should be (x1, y1, x2, y2, conf)
        self.midpoint = self.calculate_midpoint()
        self.age = 0
        self.hit_streak = 0

    def calculate_midpoint(self):
        x1, y1, x2, y2, conf = self.bbox
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        return (mx, my)
    
    def predict_next_location(self, new_midpoints):
        # IMPLEMENT KALMAN LATER nearest neighbor implementation for now
        # IMPLEMENT CONFUSION LATER surest neighbor ONLY implementation for now  
        # IMPLEMENT DYNAMIC APPROACH LATER    

        cx, cy = self.midpoint
        best_match_idx = None
        min_dist = float("inf")

        for i, (nx, ny) in enumerate(new_midpoints):
            dist = math.dist((cx, cy), (nx, ny))
            if dist < min_dist and dist <= DISTANCE_LIMIT:
                min_dist = dist
                best_match_idx = i

        return best_match_idx

    def update(self, new_bbox):
        self.bbox = new_bbox
        self.midpoint = self.calculate_midpoint()
        self.age = 0
        self.hit_streak += 1

    def increment_age(self):
        self.age += 1
        return self.age