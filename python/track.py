import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from python.kalman import BBoxGroupKalmanTracker
from python.confusion import calculate_midpoint

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
        
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
        x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int < x1_int or y2_int < y1_int:
            return 0.0
        
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def build_cost_matrix(self, tracked_entities, new_detections):
        """
        Build a cost matrix for the Hungarian algorithm.
        Lower cost = better match.
        
        Rows: existing tracked entities
        Columns: new detections
        """
        n_tracks = len(tracked_entities)
        n_detections = len(new_detections)
        
        # Initialize with high cost (no match)
        cost_matrix = np.full((n_tracks, n_detections), 1000.0)
        
        for i, entity in enumerate(tracked_entities):
            # Get predicted bbox for this entity
            predicted_bbox = entity.kalman.predict()
            predicted_midpoint = (
                (predicted_bbox[0] + predicted_bbox[2]) / 2,
                (predicted_bbox[1] + predicted_bbox[3]) / 2
            )
            
            # Get entity's average size
            entity_area = entity.get_bbox_area()
            
            for j, new_bbox in enumerate(new_detections):
                # Calculate IoU
                iou = self.calculate_iou(predicted_bbox, new_bbox)
                
                # Calculate distance
                new_midpoint = (
                    (new_bbox[0] + new_bbox[2]) / 2,
                    (new_bbox[1] + new_bbox[3]) / 2
                )
                distance = math.dist(predicted_midpoint, new_midpoint)
                
                # Calculate size ratio
                new_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
                size_ratio = max(entity_area, new_area) / (min(entity_area, new_area) + 1e-6)
                
                # Build composite cost (lower is better)
                cost = 0
                
                # IoU component (negative because higher IoU is better)
                cost += -iou * 100
                
                # Distance component (normalized)
                cost += distance / 10
                
                # Size consistency component
                if size_ratio > SIZE_CHANGE_THRESHOLD:
                    cost += (size_ratio - SIZE_CHANGE_THRESHOLD) * 50
                
                # Confidence based on hit streak (prefer established tracks)
                cost -= min(entity.hit_streak, 10) * 2
                
                # Apply hard constraints
                if distance > DISTANCE_LIMIT or (iou < IOU_THRESHOLD and distance > 200):
                    cost = 1000.0  # Impossible match
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def update(self, new_detections):
        """
        Update all tracked entities with new detections using optimal assignment.
        
        new_detections: list of bounding boxes [(x1, y1, x2, y2, ...), ...]
        """
        if not new_detections:
            # No new detections, age all tracks
            for entity_id in list(self.tracked_entities.keys()):
                if self.tracked_entities[entity_id].increment_age() > MAX_AGE:
                    del self.tracked_entities[entity_id]
            return
        
        if not self.tracked_entities:
            # No existing tracks, create new ones for all detections
            for bbox in new_detections:
                self.tracked_entities[self.next_id] = TrackedEntity(self.next_id, bbox)
                self.next_id += 1
            return
        
        # Build cost matrix
        tracked_list = list(self.tracked_entities.values())
        cost_matrix = self.build_cost_matrix(tracked_list, new_detections)
        
        # Solve assignment problem using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Track which detections and tracks were matched
        matched_tracks = set()
        matched_detections = set()
        
        # Process matches
        for row, col in zip(row_indices, col_indices):
            # Check if this is a valid match (cost not too high)
            if cost_matrix[row, col] < 900:  # Threshold for valid match
                entity = tracked_list[row]
                entity.update(new_detections[col])
                matched_tracks.add(entity.id)
                matched_detections.add(col)
        
        # Handle unmatched tracks (increment age)
        for entity in tracked_list:
            if entity.id not in matched_tracks:
                if entity.increment_age() > MAX_AGE:
                    del self.tracked_entities[entity.id]
        
        # Handle unmatched detections (create new tracks)
        for j, bbox in enumerate(new_detections):
            if j not in matched_detections:
                self.tracked_entities[self.next_id] = TrackedEntity(self.next_id, bbox)
                self.next_id += 1
        
        # Detect and mark confusion (when tracks are very close)
        self.detect_confusion()
    
    def detect_confusion(self):
        """
        Detect when multiple tracks are confused (overlapping significantly).
        """
        entities = list(self.tracked_entities.values())
        
        for i, entity1 in enumerate(entities):
            confused_with = []
            for j, entity2 in enumerate(entities):
                if i != j:
                    iou = self.calculate_iou(entity1.bbox, entity2.bbox)
                    if iou > 0.5:  # Significant overlap
                        confused_with.append(entity2.id)
            
            entity1.set_confusion(confused_with)
    
    def get_tracked_entities(self):
        """Return list of all currently tracked entities."""
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
        """Calculate the area of the bounding box"""
        x1, y1, x2, y2 = self.bbox[:4]
        return abs((x2 - x1) * (y2 - y1))
    
    def update(self, new_bbox):
        """Update the entity with a new detection."""
        self.bbox = new_bbox
        self.midpoint = self.calculate_midpoint()
        self.kalman.update(new_bbox[:4])
        self.age = 0
        self.hit_streak += 1
        self.confused_with = []
        
        # Update size history
        self.size_history.append(self.get_bbox_area())
        if len(self.size_history) > 10:
            self.size_history.pop(0)
    
    def increment_age(self):
        """Age the track when no detection is matched."""
        self.age += 1
        self.hit_streak = 0
        # Still predict to keep Kalman filter updated
        predicted_bbox = self.kalman.predict()
        # Update bbox with prediction
        self.bbox = predicted_bbox + self.bbox[4:]  # Keep any additional bbox data
        self.midpoint = self.calculate_midpoint()
        return self.age
    
    def set_confusion(self, other_ids):
        """Mark this entity as confused with others."""
        self.confused_with = other_ids
    
    def get_label(self):
        """Get display label for this entity."""
        if self.confused_with:
            all_ids = [self.id] + self.confused_with
            return "? ".join(map(str, sorted(all_ids))) + "?"
        return str(self.id)