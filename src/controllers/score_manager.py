import cv2
import numpy as np
from tqdm import tqdm

from src import AppContext
from src.controllers.detector.score_detector import ScoreDetector
from src.session.image_streamer import ImageStreamer
from src.session.videostreamer import VideoStreamer
from src.utils.daos import InputFrame


class ScoreManager(AppContext):
    """
    ScoreManager binds itself with the session
    and dispatches them to the processors that analyses the frame.
    """

    def __init__(self):
        if self.app_profile["streamer"]["evaluation"]:
            self.session = ImageStreamer()
        else:
            self.session = VideoStreamer()

        self.score_detector = ScoreDetector()

    def run(self):
        self._warmup()
        """
        Retrieves the frames from the session and passes it forward to do recognition.
        :return:
        """
        while not self.session.is_interrupted():
            self.session.update()
            detector_frame = self.session.get_detection_frame()
            self.score_detector.detect(InputFrame(detector_frame, self.session.get_frame_count(), False))
            cv2.imshow("World", detector_frame)

    def _warmup(self):
        for i in tqdm(range(self.detector_config["model"]["warm_up"]),desc="Warming up..."):
            blank_frame = np.zeros((self.score_detector.in_h, self.score_detector.in_w, 3), np.uint8)
            self.score_detector.detect(InputFrame(blank_frame, self.session.get_frame_count(), True))