import os.path

import cv2

from src import AppContext


class SessionContext(AppContext):
    """
    Root session that is inherited by either the image playback or video playback sessions
    """

    def __init__(self):
        self._frame_count = 0

    def _set_detection_frame(self, frame):
        """
        Set the frame that shall be analyzed
        :param frame:
        :return:
        """
        self._detection_frame = frame.copy()

    @property
    def detection_frame(self):
        """
        Retrieves the detection frame
        :return:
        """
        return self._detection_frame

    def is_interrupted(self) -> bool:
        """
        Handle interruptions by key press
        :return:
        """
        k = cv2.waitKey(1)
        if k == ord('q'):
            return True
        return False
