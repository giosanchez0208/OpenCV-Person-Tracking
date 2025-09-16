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
    p.add_argument(
        "--no-adaptive-scaling",
        action="store_true",
        help="Disable automatic frame downscaling (default: enabled)"
    )

    return p.parse_args()


def main():
    args = parse_args()

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
        window_name="DEMO",
        adaptive_scaling=not args.no_adaptive_scaling  # <- Hook into CameraManager
    )

    try:
        while True:
            try:
                next_frame = manager.get_frame()
            except ValueError as e:
                print("Frame grab failed:", e)
                time.sleep(0.01)
                continue

            out_frame = process(next_frame)
            manager.display_frame(out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                if not manager.paused:
                    manager.pause_playback()
                    print("[paused]")
                else:
                    manager.resume_playback()
                    print("[resumed]")

            if cv2.getWindowProperty(manager.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            time.sleep(0.001)

    finally:
        manager.release()


if __name__ == "__main__":
    main()
