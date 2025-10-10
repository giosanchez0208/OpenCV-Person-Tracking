"""Simple test to verify identify function works correctly"""
import numpy as np
from python.identify import identify, memory

# Mock frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Test 1: First call with 2 bboxes
print("=" * 50)
print("TEST 1: First call with 2 bboxes")
print("=" * 50)
bboxes1 = [(100, 100, 200, 200), (300, 100, 400, 200)]
result1 = identify(frame, [], bboxes1)
print(f"Result 1: {result1}")
print(f"Expected: 2 IDs assigned (0 and 1 or similar)")
print()

# Test 2: Second call with same 2 bboxes (should reuse IDs if embeddings match)
print("=" * 50)
print("TEST 2: Second call with same 2 bboxes")
print("=" * 50)
result2 = identify(frame, bboxes1, bboxes1)
print(f"Result 2: {result2}")
print(f"Expected: Should have 2 IDs")
print()

# Test 3: Third call with 3 bboxes (1 new)
print("=" * 50)
print("TEST 3: Third call with 3 bboxes (1 new)")
print("=" * 50)
bboxes3 = [(100, 100, 200, 200), (300, 100, 400, 200), (500, 100, 600, 200)]
result3 = identify(frame, bboxes1, bboxes3)
print(f"Result 3: {result3}")
print(f"Expected: Should have 3 IDs, with a new one for the 3rd bbox")
print()

# Test 4: Call with no bboxes
print("=" * 50)
print("TEST 4: Call with no bboxes")
print("=" * 50)
result4 = identify(frame, bboxes3, [])
print(f"Result 4: {result4}")
print(f"Expected: Empty dict {{}}")
print(f"Memory entities count: {len(memory.curr_entities)}")
print()

print("=" * 50)
print("TESTS COMPLETE")
print("=" * 50)
print(f"Final memory state: {len(memory.curr_entities)} entities")
