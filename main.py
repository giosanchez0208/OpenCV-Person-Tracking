import argparse
import sys
import time
import cv2

from python.camera_manager import CameraManager
from python.process import process

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group()
    group.add_argument("--video", "-v", type=str)
    group.add_argument("--camera", "-c", type=int, nargs='?', const=0)
    return p.parse_args()


def main():
    args = parse_args()

    # THIS PART IS ONLY FOR TESTING. SIMULATE CAMERA ENVIRONMENT WITH CAMERA MANAGER
    # No, you don't actually need this when piloting Icarus.
    # ============================================================
    
    # determine source
    if args.video:
        use_camera = False
        video_path = args.video
        camera_index = 0
    else:
        use_camera = True
        video_path = None
        camera_index = args.camera if args.camera is not None else 0

    # create camera manager
    manager = CameraManager(
        camera_index=camera_index,
        use_camera=use_camera,
        video_path=video_path,
        window_name="DEMO"
    )

    # THIS PART BELOW IS THE ACTUAL USAGE
    # ============================================================
    
    # NOTE: we call process(frame) which internally manages the playback manager
    # and runs its default processor on the frame.

    try:
        while True:
            try:
                next_frame = manager.get_frame()
            except ValueError as e:
                # temporary grab error; keep trying
                print("Frame grab failed:", e)
                time.sleep(0.01)
                continue

            # hand the raw frame to the module-level process() function
            # process() handles lazy creation of the internal manager and runs processing.
            out_frame = process(next_frame)

            # display via CameraManager (pass the frame so it won't call get_frame() internally)
            manager.display_frame(out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                # toggle pause/resume
                if not manager.paused:
                    manager.pause_playback()
                    print("[paused]")
                else:
                    manager.resume_playback()
                    print("[resumed]")

            # quit if window closed
            if cv2.getWindowProperty(manager.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # tiny sleep to avoid pegging CPU
            time.sleep(0.001)

    finally:
        manager.release()


if __name__ == "__main__":
    main()
