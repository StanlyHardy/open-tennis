import cv2

from src import AppContext


class SessionContext(AppContext):

    def __init__(self):
        self.frame_count = 0

    def set_overlay_frame(self, frame):
        self.overlayframe = frame.copy()

    def set_detection_frame(self, frame):
        self.detection_frame = frame.copy()

    def get_detection_frame(self):
        return self.detection_frame

    def get_overlay_frame(self):
        return self.overlayframe

    def is_interrupted(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True
        return False