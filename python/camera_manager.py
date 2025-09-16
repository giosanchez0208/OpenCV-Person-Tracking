import time
import cv2
from collections import deque
from typing import Optional
import tkinter as tk

class CameraManager:
    def __init__(self, camera_index=0, use_camera=True, video_path: Optional[str]=None,
                 paused_buffer_max=2000, window_name="Camera", adaptive_scaling=True):
        
        # for initialization
        self.use_camera = bool(use_camera)
        self.video_path = video_path
        self.camera_index = camera_index
        self.window_name = window_name
        self.adaptive_scaling = adaptive_scaling  # NEW PARAMETER

        if (not self.use_camera) and (self.video_path is not None):
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            self.starttime = time.time()
            self.get_frame = self.get_video_frame
        elif self.use_camera:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.fps = None
            self.frame_count = None
            self.starttime = None
            self.get_frame = self.get_camera_frame
        else:
            raise ValueError("Invalid configuration: no camera or video source provided.")
        self._original_get_frame = self.get_frame

        if not self.cap.isOpened():
            raise ValueError("Camera/Video could not be opened.")

        # for pausing
        self.skip_this_frame = False
        self.paused_frames = deque(maxlen=paused_buffer_max)
        self.paused_frames = deque()
        self.paused_buffer_max = paused_buffer_max
        self.pause_pressed = False
        self.paused = False

        # for catching up
        self.catch_up_frame = None
        
        # for window centering
        self.window_initialized = False

    def _get_screen_resolution(self):
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            return screen_width, screen_height
        except:
            return 1920, 1080

    def _center_window(self, frame):
        if not self.window_initialized and frame is not None:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            frame_height, frame_width = frame.shape[:2]
            screen_width, screen_height = self._get_screen_resolution()
            x = (screen_width - frame_width) // 2
            y = (screen_height - frame_height) // 2
            x = max(0, x)
            y = max(0, y)
            cv2.moveWindow(self.window_name, x, y)
            self.window_initialized = True

    # FRAME UTILS =======================================
    def _downscale_if_needed(self, frame):
        if frame is None:
            return frame
        if not self.adaptive_scaling:
            return frame
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800.0 / w
            new_size = (800, int(h * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return frame

    # NORMAL PLAYBACK =======================================
    def get_video_frame(self):
        if not self.cap.isOpened():
            raise ValueError("Video capture not open.")
        if self.frame_count and self.frame_count > 1:
            elapsed = time.time() - self.starttime
            frame_index = int(elapsed * self.fps) % self.frame_count
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        if not ret or frame is None:
            raise ValueError("Failed to grab video frame.")
        return self._downscale_if_needed(frame)

    def get_camera_frame(self):
        if not self.cap.isOpened():
            raise ValueError("Camera capture not open.")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise ValueError("Failed to grab camera frame.")
        return self._downscale_if_needed(frame)

    def display_frame(self, frame=None):
        """Display frame with centered window"""
        if frame is None:
            frame = self.get_frame()
        
        # Center the window on first display
        self._center_window(frame)
        
        # Display the frame
        cv2.imshow(self.window_name, frame)
        return frame

    # PAUSE PLAYBACK =======================================
    def start_pause(self):
        if not self.pause_pressed:
            self.pause_frame = self._original_get_frame()
            self.pause_pressed = True
            self.pause_start_time = time.time()
            self.delta = 1.0 / (self.fps or 30.0)
            self.last_bin = -1
            self.paused_frames.clear()
        return self.pause_frame

    def record_paused_frame(self):
        
        # Use deterministic adaptive frame skipping
        if not self.pause_pressed:
            return None
        if self.skip_this_frame:
            return self.pause_frame
        try:
            frame = self._original_get_frame()
        except ValueError:
            return self.pause_frame
        if frame is None:
            return self.pause_frame
        
        # adaptive spacing
        elapsed = time.time() - self.pause_start_time
        bin_id = int(elapsed // self.delta)
        if bin_id > self.last_bin:
            self.paused_frames.append(frame)
            self.last_bin = bin_id
            if len(self.paused_frames) > self.paused_buffer_max:
                self.paused_frames = deque(list(self.paused_frames)[::2])
                self.delta *= 2
                self.last_bin = int(elapsed // self.delta)
        return self.pause_frame

    def stop_pause(self):
        self.pause_pressed = False
        return
    
    def pause_playback(self):
        if not self.paused:
            self.paused = True
            self.start_pause() 
            self.get_frame = self.record_paused_frame

    # CATCH UP PLAYBACK =======================================
    def catch_up_to_feed(self):
        frame = None
        for _ in range(3):
            if self.paused_frames:
                frame = self.paused_frames.popleft()
        
        if not self.paused_frames:
            self.get_frame = self._original_get_frame
            self.clear_pause_state()
        
        self.catch_up_frame = frame
        return self._downscale_if_needed(
            self.catch_up_frame if self.catch_up_frame is not None else self._original_get_frame()
        )

    def clear_pause_state(self):
        self.paused_frames.clear()
        self.pause_pressed = False
        self.pause_frame = None
        self.catch_up_frame = None

    def release(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def resume_playback(self):
        if self.paused:
            self.paused = False
            self.stop_pause()
            
            if self.paused_frames:
                self.get_frame = self.catch_up_to_feed
            else:
                self.get_frame = self._original_get_frame
                self.clear_pause_state()
