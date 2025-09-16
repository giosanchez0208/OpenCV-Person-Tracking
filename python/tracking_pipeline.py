import cv2
from python.detect import *
from python.annotate import *

curr_frame = None
curr_detections = []

def tracking_pipeline(next_frame):
    
    global curr_frame
    
    # no incoming or outgoing frame
    if next_frame is None:
        return None
    if curr_frame is None:
        curr_frame = next_frame.copy()
        return next_frame

    final_frame = None
    # actual pipeline with curr and next frame
    bboxes = detect_people_bboxes(curr_frame)
    final_frame = annotate_bbox(curr_frame, bboxes)

    # update current frame
    curr_frame = next_frame.copy()

    # return final frame
    return final_frame