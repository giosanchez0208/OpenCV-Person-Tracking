from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Tuple, Any, List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_ENTITY_MEMORY = 64

@dataclass
class Entity:
    id: int
    bbox: Tuple[float, ...]
    state_vector_history: Deque[Tuple[float, ...]] = field(default_factory=lambda: deque(maxlen=MAX_ENTITY_MEMORY))
    resnet_embedding_history: Deque[Any] = field(default_factory=lambda: deque(maxlen=MAX_ENTITY_MEMORY))
    last_seen: int = 0
    kalman_filter: Optional[Any] = None  # BBoxKalmanFilter (avoid circular import)
    predicted_bbox: Optional[Tuple[float, float, float, float]] = None
    missed_detections: int = 0  # Count frames without detection (for occlusion handling)
    
    def add_state_vector(self, sv: Tuple[float, ...]) -> None:
        self.state_vector_history.append(sv)

    def add_embedding(self, emb: Any) -> None:
        self.resnet_embedding_history.append(emb)

    def get_state_vector_history(self) -> List[Tuple[float, ...]]:
        return list(self.state_vector_history)

    def get_embedding_history(self) -> List[Any]:
        return list(self.resnet_embedding_history)
    
    def is_tracked(self) -> bool:
        """Entity is actively tracked (has Kalman filter)."""
        return self.kalman_filter is not None
    
    def track_quality(self) -> float:
        """Get tracking quality (0-1, higher is better)."""
        if self.kalman_filter is None:
            return 0.0
        return self.kalman_filter.prediction_quality()


@dataclass
class Memory:
    curr_entities: Optional[List[Entity]] = None


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    if emb1 is None or emb2 is None:
        return 0.0
    
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two bboxes."""
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
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
    """Compute Euclidean distance between bbox centers."""
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    
    return float(np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2))


def compute_match_score(entity: Entity, 
                        detected_bbox: Tuple[float, ...],
                        embedding: Optional[np.ndarray],
                        verbose: bool = False) -> Tuple[float, dict]:
    """
    Compute collaborative match score combining Kalman prediction and appearance.
    Adaptively weights components based on tracking quality and context.
    
    Returns:
        (score, debug_info_dict)
    """
    debug_info = {
        'embedding_sim': 0.0,
        'iou': 0.0,
        'distance': 0.0,
        'distance_normalized': 0.0,
        'weights': {'embedding': 0.0, 'iou': 0.0, 'distance': 0.0},
        'kalman_quality': 0.0,
        'final_score': 0.0
    }
    
    # Get embedding similarity
    embedding_sim = 0.0
    if embedding is not None and entity.resnet_embedding_history:
        entity_embedding = entity.resnet_embedding_history[-1]
        embedding_sim = cosine_similarity(embedding, entity_embedding)
    debug_info['embedding_sim'] = embedding_sim
    
    # If no Kalman filter, rely purely on appearance
    if not entity.is_tracked() or entity.predicted_bbox is None:
        debug_info['weights']['embedding'] = 1.0
        debug_info['final_score'] = embedding_sim
        return embedding_sim, debug_info
    
    # Compute spatial metrics
    predicted_bbox = entity.predicted_bbox
    iou = compute_iou(predicted_bbox, detected_bbox[:4])
    distance = compute_bbox_distance(predicted_bbox, detected_bbox[:4])
    
    # Normalize distance by bbox diagonal
    bbox_diagonal = np.sqrt((detected_bbox[2] - detected_bbox[0])**2 + 
                            (detected_bbox[3] - detected_bbox[1])**2)
    distance_normalized = min(distance / max(bbox_diagonal, 1), 1.0)
    distance_score = 1.0 - distance_normalized
    
    debug_info['iou'] = iou
    debug_info['distance'] = distance
    debug_info['distance_normalized'] = distance_normalized
    
    # Get Kalman quality
    kalman_quality = entity.track_quality()
    debug_info['kalman_quality'] = kalman_quality
    
    # ADAPTIVE WEIGHTING STRATEGY:
    # 1. High quality Kalman + close proximity → trust motion more (occlusion recovery)
    # 2. Low quality Kalman or far distance → trust appearance more
    # 3. Recent reappearance (missed_detections > 0) → balance both cautiously
    
    # Base weights
    embedding_weight = 0.5
    iou_weight = 0.3
    distance_weight = 0.2
    
    # Adjust based on Kalman quality (high quality = trust motion more)
    if kalman_quality > 0.7:
        embedding_weight = 0.35
        iou_weight = 0.4
        distance_weight = 0.25
    elif kalman_quality < 0.3:
        embedding_weight = 0.7
        iou_weight = 0.2
        distance_weight = 0.1
    
    # Adjust based on proximity (very close = likely same person even if appearance differs)
    if iou > 0.5:  # High overlap
        iou_weight += 0.1
        embedding_weight -= 0.1
    elif distance_normalized > 0.5:  # Far away
        embedding_weight += 0.15
        iou_weight -= 0.1
        distance_weight -= 0.05
    
    # Adjust for recent reappearance (just came back from occlusion)
    if entity.missed_detections > 0:
        # Trust Kalman prediction more for reappearance
        embedding_weight -= 0.1
        iou_weight += 0.05
        distance_weight += 0.05
    
    # Normalize weights to sum to 1
    total_weight = embedding_weight + iou_weight + distance_weight
    embedding_weight /= total_weight
    iou_weight /= total_weight
    distance_weight /= total_weight
    
    debug_info['weights'] = {
        'embedding': embedding_weight,
        'iou': iou_weight,
        'distance': distance_weight
    }
    
    # Compute combined score
    combined_score = (embedding_weight * embedding_sim + 
                     iou_weight * iou + 
                     distance_weight * distance_score)
    
    # Boost by Kalman quality (high quality predictions are more reliable)
    quality_boost = 1.0 + (kalman_quality - 0.5) * 0.3
    final_score = float(np.clip(combined_score * quality_boost, 0, 1))
    
    debug_info['final_score'] = final_score
    
    if verbose:
        print(f"      [Score] emb={embedding_sim:.3f} iou={iou:.3f} dist={distance_score:.3f} "
              f"→ weights=[{embedding_weight:.2f}, {iou_weight:.2f}, {distance_weight:.2f}] "
              f"→ quality_boost={quality_boost:.2f} → FINAL={final_score:.3f}")
    
    return final_score, debug_info


def find_best_match(new_embedding: np.ndarray, 
                    detected_bbox: Tuple[float, ...],
                    entities: List[Entity], 
                    already_matched: set, 
                    similarity_threshold: float, 
                    verbose: bool = False) -> Tuple[Optional[int], float, dict]:
    """
    Find best matching entity using collaborative Kalman + embedding scoring.
    
    Returns:
        (matched_id, score, debug_info)
    """
def find_best_match(new_embedding: np.ndarray, 
                    detected_bbox: Tuple[float, ...],
                    entities: List[Entity], 
                    already_matched: set, 
                    similarity_threshold: float, 
                    verbose: bool = False) -> Tuple[Optional[int], float, dict]:
    """
    Find best matching entity using collaborative Kalman + embedding scoring.
    
    Returns:
        (matched_id, score, debug_info)
    """
    if not entities:
        return None, 0.0, {}
    
    best_match_id = None
    best_score = 0.0
    best_debug_info = {}
    
    if verbose:
        print(f"[Matching] Comparing detection against {len(entities)} entities (excluding {len(already_matched)} matched)...")
    
    for entity in entities:
        if entity.id in already_matched:
            if verbose:
                print(f"  Entity ID {entity.id}: ALREADY MATCHED (skipped)")
            continue
        
        # Compute collaborative match score
        score, debug_info = compute_match_score(entity, detected_bbox, new_embedding, verbose)
        
        if verbose:
            print(f"  Entity ID {entity.id}: score={score:.3f}, last_seen={entity.last_seen}, "
                  f"missed={entity.missed_detections}, kalman={'YES' if entity.is_tracked() else 'NO'}")
        
        if score > best_score:
            best_score = score
            best_match_id = entity.id
            best_debug_info = debug_info
    
    if verbose:
        print(f"[Matching] Best match: ID {best_match_id} with score {best_score:.3f} (threshold: {similarity_threshold})")
    
    if best_score >= similarity_threshold:
        return best_match_id, best_score, best_debug_info
    
    return None, best_score, best_debug_info


def match_detections_to_entities(
    detections: List[Tuple[Tuple[float, ...], Optional[np.ndarray]]],  # List of (bbox, embedding)
    entities: List[Entity],
    similarity_threshold: float,
    verbose: bool = False
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Globally optimal assignment using Hungarian algorithm.
    Solves the ID swapping problem when people cross paths.
    
    Args:
        detections: List of (bbox, embedding) tuples
        entities: List of Entity objects to match against
        similarity_threshold: Minimum score to accept a match
        verbose: Print debug information
    
    Returns:
        (matches, unmatched_detections, unmatched_entities)
        - matches: List of (detection_idx, entity_id, score)
        - unmatched_detections: List of detection indices with no match
        - unmatched_entities: List of entity IDs with no match
    """
    if not detections or not entities:
        return [], list(range(len(detections))), [e.id for e in entities]
    
    num_detections = len(detections)
    num_entities = len(entities)
    
    # Build cost matrix (convert similarity to cost: cost = 1 - similarity)
    cost_matrix = np.ones((num_detections, num_entities), dtype=np.float32)
    score_matrix = np.zeros((num_detections, num_entities), dtype=np.float32)
    
    if verbose:
        print(f"[Hungarian] Building cost matrix: {num_detections} detections × {num_entities} entities")
    
    for det_idx, (bbox, embedding) in enumerate(detections):
        for ent_idx, entity in enumerate(entities):
            score, debug_info = compute_match_score(entity, bbox, embedding, verbose=False)
            score_matrix[det_idx, ent_idx] = score
            cost_matrix[det_idx, ent_idx] = 1.0 - score  # Convert to cost
            
            if verbose:
                print(f"  Det[{det_idx}] ↔ Entity[{entity.id}]: score={score:.3f}, "
                      f"emb={debug_info.get('embedding_sim', 0):.3f}, "
                      f"iou={debug_info.get('iou', 0):.3f}")
    
    # Run Hungarian algorithm
    det_indices, ent_indices = linear_sum_assignment(cost_matrix)
    
    # Filter matches by threshold
    matches = []
    matched_det_indices = set()
    matched_ent_ids = set()
    
    for det_idx, ent_idx in zip(det_indices, ent_indices):
        score = score_matrix[det_idx, ent_idx]
        entity_id = entities[ent_idx].id
        
        if score >= similarity_threshold:
            matches.append((det_idx, entity_id, score))
            matched_det_indices.add(det_idx)
            matched_ent_ids.add(entity_id)
            
            if verbose:
                print(f"[Hungarian] MATCHED: Det[{det_idx}] → Entity[{entity_id}] (score={score:.3f})")
        else:
            if verbose:
                print(f"[Hungarian] REJECTED: Det[{det_idx}] ↔ Entity[{entity_id}] "
                      f"(score={score:.3f} < threshold={similarity_threshold})")
    
    # Find unmatched detections and entities
    unmatched_detections = [i for i in range(num_detections) if i not in matched_det_indices]
    unmatched_entities = [e.id for e in entities if e.id not in matched_ent_ids]
    
    if verbose:
        print(f"[Hungarian] Results: {len(matches)} matches, "
              f"{len(unmatched_detections)} unmatched detections, "
              f"{len(unmatched_entities)} unmatched entities")
    
    return matches, unmatched_detections, unmatched_entities


def clean_inactive_entities(entities: List[Entity], ttl_threshold: int) -> List[Entity]:
    return [entity for entity in entities if entity.last_seen < ttl_threshold]
