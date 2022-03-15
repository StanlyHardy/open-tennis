import cv2

from src import AppContext
from src.controllers.detector.score_detector import ScoreDetector
from src.session.image_session import ImageStreamer
from src.session.videostreamer import VideoStreamer
from src.utils.daos import InputFrame


class ScoreManager(AppContext):
    """
    ScoreManager binds itself with the session
and dispatches them to the processors that analyses the frame.
    """

    def __init__(self):
        if self.streamer_profile["evaluation"]:
            self.session = ImageStreamer()
        else:
            self.session = VideoStreamer()

        self.score_detector = ScoreDetector()

    def run(self):
        """
        Retrieves the frames from the session and passes it forward to do recognition.
        :return:
        """
        while not self.session.is_interrupted():
            self.session.update()
            detector_frame = self.session.get_detection_frame()
            self.score_detector.detect(InputFrame(detector_frame, self.session.get_frame_count()))
            cv2.imshow("World", detector_frame)
