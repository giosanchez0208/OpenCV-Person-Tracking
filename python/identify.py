from typing import Dict, List, Tuple, Optional
import numpy as np
import python.model as model
from python.identify_utils import Entity, Memory, match_detections_to_entities
from python.kalman import BBoxKalmanFilter

SIMILARITY_THRESHOLD = 0.20
TTL_THRESHOLD = 1000
OCCLUSION_TTL = 30  # More lenient TTL for occluded entities (Kalman keeps them alive)
VERBOSE_LOGGING = False

memory = Memory()
next_id_counter = 0


def identify(frame, curr_bboxes, next_bboxes):
    global memory, next_id_counter
    
    # Initialize memory on first call
    if memory.curr_entities is None:
        memory.curr_entities = []
        print(f"[ReID] Initialized - Similarity threshold: {SIMILARITY_THRESHOLD}, TTL: {TTL_THRESHOLD} frames")
        print(f"[ReID] Collaborative tracking: Kalman (motion) + Embeddings (appearance)")
    
    # Step 1: PREDICT - Use Kalman filters to predict where entities should be
    for entity in memory.curr_entities:
        entity.last_seen += 1
        
        if entity.kalman_filter is not None:
            entity.predicted_bbox = entity.kalman_filter.predict()
            if VERBOSE_LOGGING:
                quality = entity.track_quality()
                print(f"[Kalman] Entity {entity.id}: predicted bbox={entity.predicted_bbox}, quality={quality:.3f}")
        else:
            entity.predicted_bbox = None
    
    # Step 2: MATCH - Find best matches using GLOBAL ASSIGNMENT (Hungarian algorithm)
    # This prevents ID swapping when people cross paths!
    
    # Generate embeddings for all detections
    detections: List[Tuple[Tuple[float, ...], Optional[np.ndarray]]] = []
    for bbox_idx, bbox in enumerate(next_bboxes):
        try:
            embedding = model.generate_resnet_embedding(frame, bbox)
        except Exception as e:
            print(f"[ReID] Error generating embedding for bbox {bbox_idx}: {e}")
            embedding = None
        detections.append((bbox, embedding))
    
    # Global assignment: Find optimal detection-entity pairing
    matches, unmatched_detections, unmatched_entities = match_detections_to_entities(
        detections, memory.curr_entities, SIMILARITY_THRESHOLD, VERBOSE_LOGGING
    )
    
    identified_bbox_ids: Dict[int, int] = {}
    
    # Step 3: UPDATE MATCHED ENTITIES
    for det_idx, entity_id, score in matches:
        bbox, embedding = detections[det_idx]
        identified_bbox_ids[det_idx] = entity_id
        
        # Find and update the entity
        for entity in memory.curr_entities:
            if entity.id == entity_id:
                # Update entity state
                entity.bbox = bbox
                entity.last_seen = 0
                entity.missed_detections = 0  # Reset (successfully detected)
                
                # Update embedding
                if embedding is not None:
                    entity.add_embedding(embedding)
                
                # Update or initialize Kalman filter
                if entity.kalman_filter is None:
                    entity.kalman_filter = BBoxKalmanFilter(bbox[:4])
                    if VERBOSE_LOGGING:
                        print(f"[ReID] Entity {entity_id}: Initialized Kalman filter")
                else:
                    entity.kalman_filter.update(bbox[:4])
                
                if VERBOSE_LOGGING:
                    print(f"[ReID] Matched Det[{det_idx}] → Entity {entity_id} (score: {score:.3f})")
                break
    
    # Step 4: CREATE NEW ENTITIES for unmatched detections
    for det_idx in unmatched_detections:
        bbox, embedding = detections[det_idx]
        
        new_id = next_id_counter
        next_id_counter += 1
        identified_bbox_ids[det_idx] = new_id
        
        new_entity = Entity(id=new_id, bbox=bbox)
        if embedding is not None:
            new_entity.add_embedding(embedding)
        
        # Initialize Kalman filter for new entity
        new_entity.kalman_filter = BBoxKalmanFilter(bbox[:4])
        new_entity.predicted_bbox = bbox[:4]
        
        memory.curr_entities.append(new_entity)
        
        print(f"[ReID] NEW Entity {new_id}")
    
    # Step 5: OCCLUSION HANDLING - Mark unmatched entities as possibly occluded
    matched_entity_ids = set(entity_id for _, entity_id, _ in matches)
    
    for entity in memory.curr_entities:
        if entity.id not in matched_entity_ids:
            entity.missed_detections += 1
            
            if entity.missed_detections == 1 and entity.kalman_filter is not None:
                print(f"[ReID] Entity {entity.id} possibly occluded (Kalman tracking continues)")
            elif entity.missed_detections > OCCLUSION_TTL and entity.kalman_filter is not None:
                print(f"[ReID] Entity {entity.id} lost (missed {entity.missed_detections} frames)")
    
    # Step 6: CLEANUP - Remove truly inactive entities
    old_count = len(memory.curr_entities)
    
    # Use different TTL based on whether entity has Kalman tracking
    memory.curr_entities = [
        entity for entity in memory.curr_entities
        if entity.last_seen < (OCCLUSION_TTL if entity.is_tracked() else TTL_THRESHOLD)
    ]
    
    new_count = len(memory.curr_entities)
    if old_count != new_count:
        removed_count = old_count - new_count
        print(f"[ReID] Removed {removed_count} inactive entities (TTL expired)")
    
    return identified_bbox_ids
