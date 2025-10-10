# Kalman Filter for Person Tracking

## Overview

The new `kalman.py` provides a robust 4-point bounding box tracker that:
- Tracks all bbox coordinates (x1, y1, x2, y2) with velocities
- Predicts future positions based on motion patterns
- Combines with embedding similarity for better re-identification
- Tracks prediction quality and uncertainty

## Key Features

### 1. **BBoxKalmanFilter Class**

Tracks a bounding box through time using constant velocity model.

**State Vector** (8 dimensions):
- `[x1, y1, x2, y2, vx1, vy1, vx2, vy2]`
- Positions of top-left (x1,y1) and bottom-right (x2,y2)
- Velocities for both corners

**Key Methods**:

```python
# Create filter
kf = BBoxKalmanFilter(initial_bbox=(100, 200, 150, 300))

# Predict next position
predicted_bbox = kf.predict()  # Returns (x1, y1, x2, y2)

# Update with measurement
kf.update(detected_bbox)

# Get current state
current_bbox = kf.get_state()

# Get motion info
velocity = kf.get_velocity()  # (vx1, vy1, vx2, vy2)
center_vel = kf.get_center_velocity()  # (vx, vy) of center

# Get quality metrics
uncertainty = kf.get_uncertainty()  # Lower = more confident
quality = kf.prediction_quality()  # 0-1, higher = better
```

### 2. **Helper Functions**

```python
# IoU between boxes
iou = compute_iou(box1, box2)  # 0-1

# Distance between box centers
dist = compute_bbox_distance(box1, box2)  # pixels

# Combined matching score (Kalman + embedding)
score = compute_combined_score(
    kalman_filter=kf,
    detected_bbox=new_bbox,
    embedding_similarity=0.85,
    iou_weight=0.3,        # How much to trust spatial overlap
    distance_weight=0.2,   # How much to trust proximity
    embedding_weight=0.5   # How much to trust appearance
)
```

## Integration Plan with identify.py

### Current Flow (Embedding Only)
```
1. Get detections → bboxes
2. Generate embeddings
3. Match by embedding similarity alone
4. Assign IDs
```

### Future Flow (Kalman + Embedding)
```
1. Get detections → bboxes
2. Predict next positions using Kalman filters
3. Generate embeddings
4. Match using COMBINED score:
   - Embedding similarity (appearance)
   - IoU with prediction (spatial overlap)
   - Distance to prediction (proximity)
   - Prediction quality (how reliable is Kalman)
5. Update Kalman filters with matched bboxes
6. Assign IDs
```

### Integration Approach

**Add to Entity class** (in `identify_utils.py`):
```python
@dataclass
class Entity:
    id: int
    bbox: Tuple[float, ...]
    kalman_filter: Optional[BBoxKalmanFilter] = None  # NEW
    state_vector_history: Deque[Tuple[float, ...]] = ...
    resnet_embedding_history: Deque[Any] = ...
    last_seen: int = 0
```

**Modify identify() function**:

1. **On first detection** (create new entity):
```python
new_entity = Entity(id=new_id, bbox=bbox)
new_entity.kalman_filter = BBoxKalmanFilter(bbox[:4])  # Initialize Kalman
new_entity.add_embedding(embedding)
```

2. **Before matching** (predict positions):
```python
# Predict where each entity should be
for entity in memory.curr_entities:
    if entity.kalman_filter:
        entity.predicted_bbox = entity.kalman_filter.predict()
```

3. **During matching** (use combined score):
```python
# Instead of just embedding similarity
matched_id, similarity = find_best_match(embedding, ...)

# Use combined score
for entity in memory.curr_entities:
    if entity.id in already_matched:
        continue
    
    # Compute combined score
    combined_score = compute_combined_score(
        kalman_filter=entity.kalman_filter,
        detected_bbox=bbox,
        embedding_similarity=cosine_similarity(embedding, entity_embedding),
        iou_weight=0.3,
        distance_weight=0.2,
        embedding_weight=0.5
    )
    
    if combined_score > best_score:
        best_score = combined_score
        best_match_id = entity.id
```

4. **After matching** (update Kalman):
```python
# When match is found
if matched_id is not None:
    for entity in memory.curr_entities:
        if entity.id == matched_id:
            entity.bbox = bbox
            entity.last_seen = 0
            entity.add_embedding(embedding)
            if entity.kalman_filter:
                entity.kalman_filter.update(bbox[:4])  # Update with measurement
```

## Advantages

### 1. **Better Occlusion Handling**
- When person is occluded, Kalman predicts where they should be
- Can re-identify when they reappear near prediction

### 2. **Smoother Tracking**
- Filters out detection noise
- Bboxes don't jump around frame-to-frame

### 3. **Predictive Matching**
- Knows where person is moving
- Can handle fast motion better

### 4. **Multi-Modal Matching**
- Appearance (embedding) + Motion (Kalman) + Position (IoU/distance)
- More robust than any single cue

### 5. **Quality-Aware**
- Tracks prediction quality
- Can adjust trust based on track history

## Tuning Parameters

### Kalman Filter Initialization
```python
BBoxKalmanFilter(
    bbox,
    process_noise=1.0,      # ↑ if motion is erratic
    measurement_noise=10.0  # ↑ if detections are noisy
)
```

### Combined Score Weights
```python
compute_combined_score(
    ...,
    iou_weight=0.3,        # ↑ for crowded scenes
    distance_weight=0.2,   # ↑ for smooth motion
    embedding_weight=0.5   # ↑ for reliable ReID model
)
```

**Recommended for different scenarios**:

**Crowded scene** (people close together):
- `iou_weight=0.4, distance_weight=0.1, embedding_weight=0.5`

**Fast motion**:
- `iou_weight=0.2, distance_weight=0.3, embedding_weight=0.5`

**Occlusions**:
- `iou_weight=0.2, distance_weight=0.3, embedding_weight=0.5`

## Testing the Kalman Filter

```python
# Simple test
from python.kalman import BBoxKalmanFilter

# Person walking right
bbox = (100, 100, 150, 200)
kf = BBoxKalmanFilter(bbox)

# Simulate motion
for i in range(10):
    # Predict
    pred = kf.predict()
    print(f"Frame {i}: Predicted = {pred}")
    
    # Simulate detection (moving right)
    measured = (100 + i*5, 100, 150 + i*5, 200)
    kf.update(measured)
    
    print(f"  Velocity: {kf.get_center_velocity()}")
    print(f"  Quality: {kf.prediction_quality():.3f}")
```

## Next Steps

1. ✅ Kalman filter implementation complete
2. ⏳ Integrate into Entity class
3. ⏳ Modify find_best_match to use combined scoring
4. ⏳ Test on sample videos
5. ⏳ Tune weights for your use case
