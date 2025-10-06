from dataclasses import dataclass
import cv2
from python.detect import detect_people_bboxes
from python.annotate import annotate_bbox

@dataclass
class Tracker:
    curr_frame = None
    curr_bboxes: list = None

tracker = Tracker()  # make tracker global

def tracking_pipeline(next_frame):
    global tracker

    if next_frame is None:
        return None

    if tracker.curr_frame is None:
        tracker.curr_frame = next_frame.copy()
        tracker.curr_bboxes = detect_people_bboxes(next_frame)
        return next_frame

    next_bboxes = detect_people_bboxes(next_frame)
    final_frame = next_frame.copy()

    for bbox in next_bboxes:
        final_frame = annotate_bbox(final_frame, bbox, label="New")

    tracker.curr_frame = next_frame.copy()
    tracker.curr_bboxes = next_bboxes

    return final_frame
