import cv2

curr_frame = None

def tracking_pipeline(next_frame):
    
    global curr_frame
    
    # no incoming or outgoing frame
    if next_frame is None:
        return None
    if curr_frame is None:
        curr_frame = next_frame.copy().astype("float32")
        return next_frame

    # actual pipeline with curr and next frame
    diff = cv2.absdiff(next_frame, curr_frame.astype("uint8"))

    # update current frame
    curr_frame = next_frame.copy().astype("float32")
    return diff