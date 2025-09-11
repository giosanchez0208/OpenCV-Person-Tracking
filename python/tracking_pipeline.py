import cv2

curr_frame = None

def tracking_pipeline(next_frame):
    global curr_frame
    if next_frame is None:
        return None

    # Initialize baseline on first call
    if curr_frame is None:
        curr_frame = next_frame.copy().astype("float32")
        return next_frame

    # Subtract baseline from current next_frame
    diff = cv2.absdiff(next_frame, curr_frame.astype("uint8"))

    return diff