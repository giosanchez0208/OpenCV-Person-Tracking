def identify(frame, curr_bboxes, next_bboxes):
    identified_bbox_ids = {i: i for i in range(len(next_bboxes))}
    return identified_bbox_ids