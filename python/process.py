from python.tracking_pipeline import tracking_pipeline
from typing import Optional, Callable

class PlaybackManager:
    def __init__(self, processor: Optional[Callable] = None):
        self.current_frame = None
        self._processor = processor

    def update(self, frame):
        self.current_frame = frame

    def set_processor(self, processor: Optional[Callable]):
        self._processor = processor

    def clear_processor(self):
        self._processor = None

    def get_output(self):
        if self.current_frame is None:
            return None
        if self._processor is None:
            return self.current_frame
        try:
            processed = self._processor(self.current_frame.copy())
            return processed
        except Exception:
            # If processing fails, fallback to the unmodified frame.
            return self.current_frame
        
_playback_manager: Optional[PlaybackManager] = None

# set tracking pipeline as default processor
def process(frame):
    global _playback_manager
    if _playback_manager is None:
        _playback_manager = PlaybackManager(processor=tracking_pipeline)

    # first pass (no initial frame yet)
    if frame is None:
        if _playback_manager.current_frame is None:
            return None
        return _playback_manager.get_output()

    # normal flow
    _playback_manager.update(frame)
    return _playback_manager.get_output()