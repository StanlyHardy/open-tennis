import cv2

from src import AppContext


class SessionContext(AppContext):

    def __init__(self):
        self.frame_count = 0

    def set_detection_frame(self, frame):
        self.detection_frame = frame.copy()

    def get_detection_frame(self):
        return self.detection_frame

    def is_interrupted(self):
        k = cv2.waitKey(1)
        if k == ord('q'):
            self.csv_logger.persist(self.gt_ann)
            return True
        return False