"""
Test script to verify collaborative Kalman + Embedding ReID system
"""
import numpy as np
from python.identify_utils import Entity, compute_match_score, find_best_match
from python.kalman import BBoxKalmanFilter

def test_adaptive_weighting():
    """Test that weights adapt based on context"""
    print("=" * 70)
    print("TEST: Adaptive Weighting")
    print("=" * 70)
    
    # Create entity with high quality Kalman
    entity = Entity(id=0, bbox=(100, 100, 200, 200))
    entity.kalman_filter = BBoxKalmanFilter((100, 100, 200, 200))
    entity.predicted_bbox = (105, 100, 205, 200)  # Moved slightly right
    
    # Simulate several successful updates (high quality)
    for i in range(10):
        entity.kalman_filter.update((100 + i*5, 100, 200 + i*5, 200))
    
    # Add embedding
    entity.add_embedding(np.random.randn(512))
    
    print(f"\nEntity Kalman Quality: {entity.track_quality():.3f}")
    
    # Test 1: Detection very close to prediction
    print("\n--- Scenario 1: Close proximity (high IoU) ---")
    detected_bbox = (106, 100, 206, 200)
    embedding = np.random.randn(512)
    score, debug = compute_match_score(entity, detected_bbox, embedding, verbose=True)
    print(f"Expected: High IoU weight, lower embedding weight")
    
    # Test 2: Detection far from prediction
    print("\n--- Scenario 2: Far from prediction ---")
    detected_bbox = (300, 100, 400, 200)
    score, debug = compute_match_score(entity, detected_bbox, embedding, verbose=True)
    print(f"Expected: High embedding weight, lower spatial weights")
    
    # Test 3: After occlusion (missed detections)
    print("\n--- Scenario 3: Reappearance after occlusion ---")
    entity.missed_detections = 5
    detected_bbox = (105, 100, 205, 200)
    score, debug = compute_match_score(entity, detected_bbox, embedding, verbose=True)
    print(f"Expected: Favor Kalman prediction for reappearance")
    

def test_occlusion_handling():
    """Test occlusion detection and recovery"""
    print("\n" + "=" * 70)
    print("TEST: Occlusion Handling")
    print("=" * 70)
    
    entity = Entity(id=1, bbox=(100, 100, 200, 200))
    entity.kalman_filter = BBoxKalmanFilter((100, 100, 200, 200))
    entity.add_embedding(np.random.randn(512))
    
    print("\nInitial state:")
    print(f"  missed_detections: {entity.missed_detections}")
    print(f"  last_seen: {entity.last_seen}")
    
    # Simulate occlusion
    print("\n--- Simulating 3 frames of occlusion ---")
    for frame in range(3):
        entity.last_seen += 1
        entity.missed_detections += 1
        predicted = entity.kalman_filter.predict()
        print(f"  Frame {frame+1}: missed={entity.missed_detections}, "
              f"predicted_bbox={tuple(int(x) for x in predicted)}")
    
    # Reappearance
    print("\n--- Reappearance near prediction ---")
    detected_bbox = (115, 100, 215, 200)  # Near prediction
    entity.predicted_bbox = entity.kalman_filter.get_state()
    
    embedding = entity.resnet_embedding_history[0]  # Same appearance
    score, debug = compute_match_score(entity, detected_bbox, embedding, verbose=True)
    
    print(f"\nMatch Score: {score:.3f}")
    print(f"Expected: High score due to proximity to Kalman prediction")
    

def test_no_kalman_fallback():
    """Test that system falls back to pure embedding when no Kalman"""
    print("\n" + "=" * 70)
    print("TEST: Fallback to Pure Embedding (No Kalman)")
    print("=" * 70)
    
    entity = Entity(id=2, bbox=(100, 100, 200, 200))
    # No Kalman filter
    entity.kalman_filter = None
    
    embedding1 = np.random.randn(512)
    entity.add_embedding(embedding1)
    
    # Same embedding
    detected_bbox = (300, 300, 400, 400)  # Far away
    score_same, debug_same = compute_match_score(entity, detected_bbox, embedding1, verbose=False)
    
    # Different embedding
    embedding2 = np.random.randn(512)
    score_diff, debug_diff = compute_match_score(entity, detected_bbox, embedding2, verbose=False)
    
    print(f"\nSame embedding score: {score_same:.3f}")
    print(f"Different embedding score: {score_diff:.3f}")
    print(f"Embedding weight: {debug_same['weights']['embedding']:.2f}")
    print(f"Expected: weight=1.0 (pure embedding, no spatial component)")
    

def test_quality_progression():
    """Test that Kalman quality improves with successful updates"""
    print("\n" + "=" * 70)
    print("TEST: Track Quality Progression")
    print("=" * 70)
    
    entity = Entity(id=3, bbox=(100, 100, 200, 200))
    entity.kalman_filter = BBoxKalmanFilter((100, 100, 200, 200))
    
    print("\nQuality over time with successful updates:")
    for i in range(15):
        quality = entity.track_quality()
        print(f"  Update {i}: quality={quality:.3f}, hits={entity.kalman_filter.hits}, "
              f"age={entity.kalman_filter.age}")
        
        # Update with smooth motion
        entity.kalman_filter.predict()
        entity.kalman_filter.update((100 + i*3, 100, 200 + i*3, 200))
    
    print(f"\nFinal quality: {entity.track_quality():.3f}")
    print(f"Expected: Quality should increase as track matures")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COLLABORATIVE REID SYSTEM - UNIT TESTS")
    print("=" * 70)
    
    test_adaptive_weighting()
    test_occlusion_handling()
    test_no_kalman_fallback()
    test_quality_progression()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print("\nNext: Run 'python main.py --video videos/kalman_test.mp4' to test on real video")
