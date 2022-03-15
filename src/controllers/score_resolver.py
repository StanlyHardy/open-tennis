import cv2

from src import AppContext
from src.controllers.score_detector import ScoreDetector
from src.session.image_session import ImageStreamer
from src.session.videosession import VideoSession
from src.utils.daos import InputFrame


class ScoreResolver(AppContext):
    def __init__(self):
        if self.streamer_profile["evaluation"]:
            self.camera = ImageStreamer()
        else:
            self.camera = VideoSession()

        self.score_detector = ScoreDetector()

    def run(self):
        while not self.camera.is_interrupted():
            self.camera.update_frames()
            detector_frame = self.camera.get_detection_frame()
            self.score_detector.detect(InputFrame(detector_frame, self.camera.get_frame_count()))
            cv2.imshow("World", detector_frame)
