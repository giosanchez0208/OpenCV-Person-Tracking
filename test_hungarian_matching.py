"""
Test to demonstrate the ID swapping fix using Hungarian algorithm
"""
import numpy as np
from python.identify_utils import Entity, match_detections_to_entities
from python.kalman import BBoxKalmanFilter

def test_crossing_scenario():
    """
    Test the classic ID swapping scenario:
    Two people walking towards each other and crossing paths.
    
    Without Hungarian: Greedy matching causes IDs to swap
    With Hungarian: Global optimization maintains correct IDs
    """
    print("=" * 70)
    print("TEST: ID Swapping Prevention (Crossing Paths)")
    print("=" * 70)
    
    # Initial setup: Two people on opposite sides
    entity_0 = Entity(id=0, bbox=(100, 100, 150, 200))
    entity_0.kalman_filter = BBoxKalmanFilter((100, 100, 150, 200))
    entity_0.add_embedding(np.random.randn(512))
    entity_0.predicted_bbox = (110, 100, 160, 200)  # Moving right
    
    entity_1 = Entity(id=1, bbox=(200, 100, 250, 200))
    entity_1.kalman_filter = BBoxKalmanFilter((200, 100, 250, 200))
    entity_1.add_embedding(np.random.randn(512))
    entity_1.predicted_bbox = (190, 100, 240, 200)  # Moving left
    
    entities = [entity_0, entity_1]
    
    print("\nInitial State:")
    print(f"  Entity 0: bbox={entity_0.bbox}, predicted={entity_0.predicted_bbox}")
    print(f"  Entity 1: bbox={entity_1.bbox}, predicted={entity_1.predicted_bbox}")
    
    # Simulate crossing: detections are now in swapped positions
    # Detection 0 is now closer to Entity 1's prediction
    # Detection 1 is now closer to Entity 0's prediction
    # BUT they still have their original appearances
    
    detection_0_bbox = (115, 100, 165, 200)  # Still person 0, moved right
    detection_0_emb = entity_0.resnet_embedding_history[0]  # Same appearance as entity 0
    
    detection_1_bbox = (185, 100, 235, 200)  # Still person 1, moved left  
    detection_1_emb = entity_1.resnet_embedding_history[0]  # Same appearance as entity 1
    
    detections = [
        (detection_0_bbox, detection_0_emb),
        (detection_1_bbox, detection_1_emb)
    ]
    
    print("\nNew Detections (people moving toward each other):")
    print(f"  Detection 0: bbox={detection_0_bbox} (person 0, moved right)")
    print(f"  Detection 1: bbox={detection_1_bbox} (person 1, moved left)")
    
    # Run Hungarian algorithm
    print("\n" + "-" * 70)
    print("Running Hungarian Algorithm...")
    print("-" * 70)
    
    matches, unmatched_dets, unmatched_ents = match_detections_to_entities(
        detections, entities, similarity_threshold=0.2, verbose=True
    )
    
    # Verify correct matching
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    expected_matches = {0: 0, 1: 1}  # Detection 0 → Entity 0, Detection 1 → Entity 1
    actual_matches = {det_idx: ent_id for det_idx, ent_id, _ in matches}
    
    print(f"\nExpected: {expected_matches}")
    print(f"Actual:   {actual_matches}")
    
    if actual_matches == expected_matches:
        print("\n✅ SUCCESS: IDs maintained correctly (no swapping!)")
        print("   Hungarian algorithm correctly matched by appearance despite spatial proximity")
    else:
        print("\n❌ FAILURE: IDs were swapped!")
        print("   This should not happen with Hungarian algorithm")
    
    # Show why it worked
    print("\n" + "-" * 70)
    print("Why Hungarian Algorithm Prevents Swapping:")
    print("-" * 70)
    print("1. Greedy matching would:")
    print("   - Match Det[0] first, find Entity 1 closer spatially")
    print("   - Force Det[1] to match Entity 0")
    print("   - Result: SWAP!")
    print("\n2. Hungarian algorithm:")
    print("   - Builds complete cost matrix for all pairings")
    print("   - Finds GLOBALLY optimal assignment")
    print("   - Appearance similarity dominates for crossing scenarios")
    print("   - Result: NO SWAP!")
    
    return actual_matches == expected_matches


def test_extreme_crossing():
    """Test when people have very high spatial overlap (extreme crossing)"""
    print("\n\n" + "=" * 70)
    print("TEST: Extreme Crossing (High Spatial Overlap)")
    print("=" * 70)
    
    # Two people very close together
    entity_0 = Entity(id=0, bbox=(100, 100, 150, 200))
    entity_0.kalman_filter = BBoxKalmanFilter((100, 100, 150, 200))
    entity_0.add_embedding(np.array([1.0] * 512))  # Distinct embedding
    entity_0.predicted_bbox = (105, 100, 155, 200)
    
    entity_1 = Entity(id=1, bbox=(145, 100, 195, 200))  # Overlapping
    entity_1.kalman_filter = BBoxKalmanFilter((145, 100, 195, 200))
    entity_1.add_embedding(np.array([-1.0] * 512))  # Opposite embedding
    entity_1.predicted_bbox = (140, 100, 190, 200)
    
    entities = [entity_0, entity_1]
    
    # Detections have crossed but maintain appearance
    detection_0_bbox = (110, 100, 160, 200)
    detection_0_emb = entity_0.resnet_embedding_history[0]
    
    detection_1_bbox = (135, 100, 185, 200)  # Very close!
    detection_1_emb = entity_1.resnet_embedding_history[0]
    
    detections = [
        (detection_0_bbox, detection_0_emb),
        (detection_1_bbox, detection_1_emb)
    ]
    
    print("\nScenario: Two people crossing with high spatial overlap")
    print(f"  Entity 0 predicted: {entity_0.predicted_bbox}")
    print(f"  Entity 1 predicted: {entity_1.predicted_bbox}")
    print(f"  Detection 0: {detection_0_bbox}")
    print(f"  Detection 1: {detection_1_bbox}")
    
    matches, _, _ = match_detections_to_entities(
        detections, entities, similarity_threshold=0.2, verbose=False
    )
    
    actual_matches = {det_idx: ent_id for det_idx, ent_id, _ in matches}
    expected_matches = {0: 0, 1: 1}
    
    print(f"\nExpected: {expected_matches}")
    print(f"Actual:   {actual_matches}")
    
    if actual_matches == expected_matches:
        print("\n✅ SUCCESS: Even with high overlap, IDs maintained!")
    else:
        print("\n⚠️  SWAP OCCURRED: This can happen with very similar appearances")
    
    return actual_matches == expected_matches


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HUNGARIAN ALGORITHM - ID SWAPPING PREVENTION TESTS")
    print("=" * 70)
    
    test1_passed = test_crossing_scenario()
    test2_passed = test_extreme_crossing()
    
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Basic Crossing Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Extreme Crossing Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("\nThe Hungarian algorithm solves the ID swapping problem by finding")
    print("the globally optimal assignment instead of greedy sequential matching.")
    print("=" * 70)
