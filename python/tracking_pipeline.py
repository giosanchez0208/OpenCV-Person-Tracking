import cv2
from python.detect import *
from python.annotate import *
from python.track import MultiObjectTracker

# Global variables
tracker = MultiObjectTracker()
curr_frame = None

def tracking_pipeline(next_frame):
    global tracker
    global curr_frame

    if next_frame is None:
        return None
        
    # Detect people in the current frame
    detected_bboxes = detect_people_bboxes(next_frame)
    
    # Update tracker with new detections
    tracker.update(detected_bboxes)
    
    # Get currently tracked entities
    tracked_entities = tracker.get_tracked_entities()
    
    # Prepare bboxes for annotation (only show active tracks)
    annotated_bboxes = []
    labels = []
    
    for entity in tracked_entities:
        # Only annotate entities that were recently updated (age 0) or have high confidence
        if entity.age == 0 or (entity.hit_streak > 3 and entity.age <= 5):
            x1, y1, x2, y2 = entity.bbox[:4]
            conf = entity.bbox[4] if len(entity.bbox) > 4 else 0.5
            annotated_bboxes.append((x1, y1, x2, y2, conf))
            labels.append(entity.get_label())  # Uses confusion-aware labeling
    
    # Create label function for annotation
    def get_entity_label(idx):
        if idx < len(labels):
            return labels[idx]
        return "?"
    
    # Annotate the frame
    if annotated_bboxes:
        annotated_frame = annotate_bbox(next_frame, annotated_bboxes, label=get_entity_label)
    else:
        annotated_frame = next_frame.copy()
    
    # Update current frame
    curr_frame = next_frame.copy()
    
    return annotated_frame

def reset_tracker():
    """Reset the tracker - useful for new video sequences"""
    global tracker
    tracker = MultiObjectTracker()

def get_tracking_stats():
    """Get current tracking statistics"""
    global tracker
    entities = tracker.get_tracked_entities()
    active_tracks = len([e for e in entities if e.age == 0])
    total_tracks = len(entities)
    confused_tracks = len([e for e in entities if e.confused_with])
    
    return {
        'active_tracks': active_tracks,
        'total_tracks': total_tracks,
        'confused_tracks': confused_tracks,
        'next_id': tracker.next_id
    }