import os.path

import cv2
import numpy as np
from tqdm import tqdm

from src import AppContext
from src.controllers.detector.score_detector import ScoreDetector
from src.session.image_streamer import ImageStreamer
from src.session.videostreamer import VideoStreamer
from src.utils.daos import InputFrame, Result


class OpenTennis(AppContext):
    """
    OpenTennis binds itself with the session
    and dispatches them to the processors that analyses the frame.
    """

    def __init__(self):
        self.score_detector = ScoreDetector()
        self._warmup()

        if self.app_profile["streamer"]["evaluation"]:
            self.session = ImageStreamer()
        else:
            self.session = VideoStreamer()
        self.observer = self.notif_center.add_observer(with_block=self.detection_result,
                                                       for_name="OpenTennis")
        self.observer = self.notif_center.add_observer(with_block=self.detection_result,
                                                       for_name="TxPoints")
        if self.app_profile["streamer"]["save_stream"]:
            self.out = cv2.VideoWriter(os.path.expanduser(self.app_profile["paths"]["output_video_path"]),
                                       cv2.VideoWriter_fourcc(*'MP4V'), 25, (self.session.width, self.session.height))

    def detection_result(self, sender, event_name, result: Result):
        if event_name == "OpenTennis":
            self.scoreboard_result = result
        elif event_name == "TxPoints":
            self.court_points = result
        if result is None:
            self.scoreboard_result, self.court_points = None, None

    def run(self):

        """
        Retrieves the frames from the session and passes it forward to do recognition.
        """
        while not self.session.is_interrupted():

            self.session.update()
            det_frame = self.session.detection_frame

            self.score_detector.detect(InputFrame(det_frame, self.session.frame_count, False))

            if self.court_points is not None:
                det_frame = self.renderer.render_court_points(det_frame, self.court_points)
            if self.scoreboard_result is not None:
                # TODO PIL based rendering
                self.renderer.render_result(det_frame, self.scoreboard_result)

            if self.app_profile["streamer"]["view_imshow"]:
                cv2.imshow("World", det_frame)
            if self.app_profile["streamer"]["save_stream"]:
                self.out.write(det_frame)

    def _warmup(self):
        """
        Warm up since the first few frames tends to be slow
        """
        for i in tqdm(range(self.detector_config["model"]["warm_up"]), desc="Warming up..."):
            blank_frame = np.zeros((self.detector_config["model"]["img_size"],
                                    self.detector_config["model"]["img_size"], 3), np.uint8)
            self.score_detector.detect(InputFrame(blank_frame, 0, True))
