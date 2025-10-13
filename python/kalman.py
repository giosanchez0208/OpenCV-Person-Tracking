import numpy as np
from scipy.linalg import solve, LinAlgError

from typing import Tuple, Optional

def create_state_vector(bbox):

    x1, y1, x2, y2 = bbox

class BBoxKalmanFilter:    # zero velocity

    """    dx1, dy1, dx2, dy2 = 0.0, 0.0, 0.0, 0.0

    Kalman filter for tracking full bounding box (x1, y1, x2, y2) with velocities.
    state_vector = np.array([x1, y1, x2, y2, dx1, dy1, dx2, dy2], dtype=float)

        return state_vector

    State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
    - (x1, y1): top-left corner
    - (x2, y2): bottom-right corner  
    - (vx1, vy1): velocity of top-left
    - (vx2, vy2): velocity of bottom-right
    """
    
    def __init__(self, initial_bbox: Tuple[float, float, float, float], 
                 process_noise: float = 1.0, 
                 measurement_noise: float = 10.0):
        """
        Args:
            initial_bbox: (x1, y1, x2, y2)
            process_noise: Process uncertainty (how much we trust the motion model)
            measurement_noise: Measurement uncertainty (how much we trust detections)
        """
        x1, y1, x2, y2 = initial_bbox
        
        # State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.x = np.array([x1, y1, x2, y2, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x1 = x1 + vx1
            [0, 1, 0, 0, 0, 1, 0, 0],  # y1 = y1 + vy1
            [0, 0, 1, 0, 0, 0, 1, 0],  # x2 = x2 + vx2
            [0, 0, 0, 1, 0, 0, 0, 1],  # y2 = y2 + vy2
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx1 = vx1
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy1 = vy1
            [0, 0, 0, 0, 0, 0, 1, 0],  # vx2 = vx2
            [0, 0, 0, 0, 0, 0, 0, 1],  # vy2 = vy2
        ], dtype=np.float32)
        
        # Measurement matrix (we measure positions, not velocities)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float32) * measurement_noise
        
        # State covariance (initial uncertainty)
        self.P = np.eye(8, dtype=np.float32) * 1000
        
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
    
    def predict(self) -> Tuple[float, float, float, float]:
        """
        Predict the next state and return predicted bbox.
        
        Returns:
            Predicted bbox (x1, y1, x2, y2)
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        self.time_since_update += 1
        
        return self._state_to_bbox()
    
    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        """
        Update the state with a new measurement.
        
        Args:
            bbox: Measured bbox (x1, y1, x2, y2)
        """
        z = np.array(bbox, dtype=np.float32).reshape(4, 1)
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        self.hits += 1
        self.time_since_update = 0
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """Get current bbox estimate."""
        return self._state_to_bbox()
    
    def get_velocity(self) -> Tuple[float, float, float, float]:
        """Get bbox velocity (vx1, vy1, vx2, vy2)."""
        return (float(self.x[4, 0]), float(self.x[5, 0]), 
                float(self.x[6, 0]), float(self.x[7, 0]))
    
    def get_center_velocity(self) -> Tuple[float, float]:
        """Get center point velocity."""
        vx1, vy1, vx2, vy2 = self.get_velocity()
        return ((vx1 + vx2) / 2, (vy1 + vy2) / 2)
    
    def get_uncertainty(self) -> float:
        """
        Get position uncertainty (average std dev of bbox corners).
        Lower values = more confident prediction.
        """
        position_variance = np.diag(self.P[:4, :4])
        return float(np.mean(np.sqrt(position_variance)))
    
    def prediction_quality(self) -> float:
        """
        Quality score for this prediction (0-1, higher is better).
        Based on hit rate and uncertainty.
        """
        hit_ratio = self.hits / max(self.age, 1)
        uncertainty_penalty = 1.0 / (1.0 + self.get_uncertainty() / 100)
        age_penalty = 1.0 / (1.0 + self.time_since_update / 5)
        
        return hit_ratio * uncertainty_penalty * age_penalty
    
    def _state_to_bbox(self) -> Tuple[float, float, float, float]:
        """Convert state vector to bbox tuple."""
        x1 = float(self.x[0, 0])
        y1 = float(self.x[1, 0])
        x2 = float(self.x[2, 0])
        y2 = float(self.x[3, 0])
        
        # Ensure valid bbox (x1 < x2, y1 < y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        return (x1, y1, x2, y2)
    
    def iou_with_prediction(self, bbox: Tuple[float, float, float, float]) -> float:
        """
        Calculate IoU between given bbox and current prediction.
        Useful for matching detected boxes to predictions.
        """
        predicted = self.get_state()
        return compute_iou(predicted, bbox)


def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union between two bboxes.
    
    Args:
        box1, box2: (x1, y1, x2, y2)
    
    Returns:
        IoU score (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    
    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_bbox_distance(box1: Tuple[float, float, float, float],
                          box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Euclidean distance between bbox centers.
    
    Args:
        box1, box2: (x1, y1, x2, y2)
    
    Returns:
        Distance in pixels
    """
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    
    return float(np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2))


def compute_combined_score(kalman_filter: BBoxKalmanFilter,
                           detected_bbox: Tuple[float, float, float, float],
                           embedding_similarity: float,
                           iou_weight: float = 0.3,
                           distance_weight: float = 0.2,
                           embedding_weight: float = 0.5) -> float:
    """
    Combine Kalman prediction, IoU, distance, and embedding similarity into single score.
    
    Args:
        kalman_filter: Kalman filter with prediction
        detected_bbox: Newly detected bbox
        embedding_similarity: Cosine similarity of embeddings (0-1)
        iou_weight: Weight for IoU component
        distance_weight: Weight for distance component (inverted)
        embedding_weight: Weight for embedding similarity
    
    Returns:
        Combined score (0-1, higher is better match)
    """
    predicted_bbox = kalman_filter.get_state()
    
    # IoU score
    iou_score = compute_iou(predicted_bbox, detected_bbox)
    
    # Distance score (normalize and invert - closer is better)
    distance = compute_bbox_distance(predicted_bbox, detected_bbox)
    bbox_diagonal = np.sqrt((detected_bbox[2] - detected_bbox[0])**2 + 
                            (detected_bbox[3] - detected_bbox[1])**2)
    normalized_distance = min(distance / max(bbox_diagonal, 1), 1.0)
    distance_score = 1.0 - normalized_distance
    
    # Combine scores
    combined = (iou_weight * iou_score + 
                distance_weight * distance_score + 
                embedding_weight * embedding_similarity)
    
    # Boost by prediction quality
    quality_boost = 1.0 + (kalman_filter.prediction_quality() - 0.5) * 0.2
    
    return float(np.clip(combined * quality_boost, 0, 1))
