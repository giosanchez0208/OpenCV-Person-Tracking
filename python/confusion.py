import math

OVERLAP_THRESHOLD = 0.5

class ConfusedEntity:
    def __init__(self, possible_ids, bbox):
        self.possible_ids = possible_ids
        self.bbox = bbox
        self.midpoint = calculate_midpoint(bbox)
        
    def get_label(self):
        return "? ".join(map(str, self.possible_ids)) + "?"

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_midpoint(bbox):
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def detect_confusion(entities):
    confused_groups = []
    processed = set()
    
    for i, e1 in enumerate(entities):
        if i in processed:
            continue
            
        confused_with = [e1]
        processed.add(i)
        
        for j, e2 in enumerate(entities[i+1:], i+1):
            if calculate_iou(e1.bbox, e2.bbox) > OVERLAP_THRESHOLD:
                confused_with.append(e2)
                processed.add(j)
        
        if len(confused_with) > 1:
            confused_groups.append(confused_with)
    
    return confused_groups

def resolve_confusion(confused_entities, kalman_trackers):
    resolved = []
    
    for entity in confused_entities:
        # get predicted positions from Kalman filters
        predictions = {}
        for id in entity.possible_ids:
            if id in kalman_trackers:
                pred_pos = kalman_trackers[id].predict()
                predictions[id] = pred_pos
        
        # find best match based on predicted positions
        best_id = None
        min_dist = float("inf")
        
        for id, pred_pos in predictions.items():
            dist = math.dist(pred_pos, entity.midpoint)
            if dist < min_dist:
                min_dist = dist
                best_id = id
        
        if best_id is not None:
            resolved.append((best_id, entity.bbox))
    
    return resolved