import cv2
from python.detect import *
from python.annotate import *
from python.track import *

curr_frame = None
curr_tracked_entities = []
newest_id = 0

def tracking_pipeline(next_frame):
    global curr_frame
    global curr_tracked_entities
    global newest_id

    if next_frame is None:
        return None
        
    if curr_frame is None:
        # First frame - initialize tracking
        bboxes = detect_people_bboxes(next_frame)
        curr_tracked_entities = [TrackedEntity(newest_id + i, bbox) for i, bbox in enumerate(bboxes)]
        newest_id += len(bboxes)
        curr_frame = next_frame.copy()
        annotated_frame = annotate_bbox(next_frame, [(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]) for bbox in bboxes])
        return annotated_frame

    # Detect people in the new frame
    next_bboxes = detect_people_bboxes(next_frame)
    
    # Calculate midpoints for matching
    next_midpoints = []
    for bbox in next_bboxes:
        x1, y1, x2, y2, conf = bbox
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        next_midpoints.append((mx, my))
    
    # Update existing tracked entities
    used_detections = set()
    for entity in curr_tracked_entities:
        match_idx = entity.predict_next_location(next_midpoints)
        if match_idx is not None and match_idx not in used_detections:
            entity.update(next_bboxes[match_idx])
            used_detections.add(match_idx)
        else:
            entity.increment_age()
    
    # Remove old entities
    curr_tracked_entities = [e for e in curr_tracked_entities if e.age <= MAX_AGE]
    
    # Add new detections as new entities
    for i, bbox in enumerate(next_bboxes):
        if i not in used_detections:
            curr_tracked_entities.append(TrackedEntity(newest_id, bbox))
            newest_id += 1
    
    # Prepare bboxes for annotation
    annotated_bboxes = []
    for entity in curr_tracked_entities:
        if entity.age == 0:
            x1, y1, x2, y2 = entity.bbox[:4]
            conf = entity.bbox[4] if len(entity.bbox) > 4 else 0.5
            annotated_bboxes.append((x1, y1, x2, y2, conf))
    
    # Annotate the frame
    annotated_frame = annotate_bbox(next_frame, annotated_bboxes, label=lambda idx: f"{curr_tracked_entities[idx].id}")
    # Update current frame
    curr_frame = next_frame.copy()
    
    return annotated_frame