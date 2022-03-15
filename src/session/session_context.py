import cv2

from src import AppContext


class SessionContext(AppContext):
    """
    Root session that is inherited by either the image playback or video playback sessions
    """
    def __init__(self):
        self.frame_count = 0

    def _set_detection_frame(self, frame):
        """
        Set the frame that shall be analyzed
        :param frame:
        :return:
        """
        self.detection_frame = frame.copy()

    def get_detection_frame(self):
        """
        Retrieves the detection frame
        :return:
        """
        return self.detection_frame

    def is_interrupted(self):
        """
        Handle interruptions by key press
        :return:
        """
        k = cv2.waitKey(1)
        if k == ord('q'):
            self.csv_logger.persist(self.gt_ann)
            return True
        return False