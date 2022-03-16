import os.path

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
        self.score_detector = ScoreDetector()
        self._warmup()

        if self.app_profile["streamer"]["evaluation"]:
            self.session = ImageStreamer()
        else:
            self.session = VideoStreamer()

        if self.app_profile["streamer"]["save_stream"]:
            self.out = cv2.VideoWriter(os.path.expanduser(self.app_profile["paths"]["output_video_path"]), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (self.session.width, self.session.height))

    def run(self):

        """
        Retrieves the frames from the session and passes it forward to do recognition.
        :return:
        """
        while not self.session.is_interrupted():
            self.session.update()
            det_frame = self.session.detection_frame
            self.score_detector.detect(InputFrame(det_frame, self.session.frame_count, False))
            if self.app_profile["streamer"]["view_imshow"]:
                cv2.imshow("World", det_frame)
            if self.app_profile["streamer"]["save_stream"]:
                self.out.write(det_frame)

    def _warmup(self):
        for i in tqdm(range(self.detector_config["model"]["warm_up"]),desc="Warming up..."):
            blank_frame = np.zeros((self.score_detector.in_h, self.score_detector.in_w, 3), np.uint8)
            self.score_detector.detect(InputFrame(blank_frame, 0, True))