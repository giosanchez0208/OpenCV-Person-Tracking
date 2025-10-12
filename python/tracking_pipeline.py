from dataclasses import dataclass
from python.detect import detect_people_bboxes
from python.annotate import annotate_bbox
from python.identify import identify


@dataclass
class Tracker:
    curr_frame = None
    curr_bboxes: list = None
    curr_bbox_ids: dict = None


tracker = Tracker()


def tracking_pipeline(next_frame):
    global tracker

    if next_frame is None:
        return None

    # 0. initialize tracker on first frame
    if tracker.curr_frame is None:
        tracker.curr_frame = next_frame  # keep reference; avoid full copy unless needed
        tracker.curr_bboxes = detect_people_bboxes(next_frame)

        # Initialize memory with first frame detections
        tracker.curr_bbox_ids = identify(next_frame, [], tracker.curr_bboxes)

        # Annotate once (single copy) and draw in-place per bbox
        final_frame = next_frame.copy()
        for i, bbox in enumerate(tracker.curr_bboxes):
            bbox_id = tracker.curr_bbox_ids.get(i, i)
            annotate_bbox(final_frame, bbox, label=str(bbox_id))

        return final_frame

    # 1. detect new bboxes in next frame
    next_bboxes = detect_people_bboxes(next_frame)

    # 2. identify (match) IDs for next_bboxes
    next_bbox_ids = identify(next_frame, tracker.curr_bboxes, next_bboxes)

    # 3. annotate bboxes — single frame copy then annotate in-place per bbox
    final_frame = next_frame.copy()
    for i, bbox in enumerate(next_bboxes):
        bbox_id = next_bbox_ids.get(i, i)
        annotate_bbox(final_frame, bbox, label=str(bbox_id))

    # 4. update tracker
    # Keep a reference to the current frame to avoid expensive copies; copy only when needed elsewhere
    tracker.curr_frame = next_frame
    tracker.curr_bboxes = next_bboxes
    tracker.curr_bbox_ids = next_bbox_ids

    return final_frame
