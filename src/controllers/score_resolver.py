from concurrent.futures import ThreadPoolExecutor
import cv2

from src import AppContext
from src.controllers.score_detector import ScoreDetector
from src.interfaces.streamer import Streamer
from src.utils.daos import InputFrame


class ScoreResolver(AppContext):
    def __init__(self):
        self.camera = Streamer()
        self.score_detector = ScoreDetector()
        self.frame_count = 0

    def run(self):
        with ThreadPoolExecutor(8) as executor:
            while not self.camera.is_interrupted():
                self.camera.update_frames()
                detector_frame = self.camera.get_detection_frame()
                self.score_detector.detect(InputFrame(detector_frame, self.frame_count))

                self.frame_count += 1
                cv2.imshow("World", detector_frame)
