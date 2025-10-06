import cv2
from python.detect import *

# Global variables
curr_frame = None

def tracking_pipeline(next_frame):
    global tracker
    global curr_frame

    if next_frame is None:
        return None

    # Update current frame
    curr_frame = next_frame.copy()
    
    return next_frame