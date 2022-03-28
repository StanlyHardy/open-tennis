import os.path

import cv2
import numpy as np
from tqdm import tqdm

from src import AppContext
from src.controllers.detector.score_detector import ScoreDetector
from src.session.image_streamer import ImageStreamer
from src.session.videostreamer import VideoStreamer
from src.utils.daos import InputFrame, Result


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
            self.out = cv2.VideoWriter(os.path.expanduser(self.app_profile["paths"]["output_video_path"]),
                                       cv2.VideoWriter_fourcc(*'MP4V'), 25, (self.session.width, self.session.height))

        self.observer = self.notif_center.add_observer(with_block=self.detection_result,
                                                       for_name="ScoreManager")

    def detection_result(self, sender, event_name, result: Result):
        self.scoreboard_result = result

    def draw(self, det_frame, result: Result):
        """

        :param score_board: Scoreboard object with its metadata
        :param result: Processed result
        """
        h, w = det_frame.shape[:2]
        tl = round(0.002 * (w + h) / 2) + 1
        det_frame = \
            self.render.draw_canvas(det_frame.copy(), det_frame)

        det_frame = \
            self.render.draw_canvas(det_frame.copy(), det_frame)

        bbox = self.scoreboard_result.score_board.bbox
        x1, y1, x2, y2 = map(int, bbox)
        self.render.text(det_frame, "scoreboard", (x1 + 3, y1 - 4), 0, tl / 3)
        self.render.rect(
            det_frame, (x1, y1), (x2, y2),
            thickness=max(
                int((w + h) / 600), 1)
        )
        self.render.text(det_frame, "Player 1: {}".format(result.name_1.title()),
                         coordinate=(870, 940))
        if len(result.score_1) > 0:
            self.render.text(det_frame, "Score:    {}".format(result.score_1),
                             coordinate=(870, 980))
        else:
            self.render.text(det_frame, "Score:    {}".format("Recognizing"),
                             coordinate=(870, 980))

        self.render.text(det_frame, "Player 2: {}".format(result.name_2.title()),
                         coordinate=(1370, 940))
        if len(result.score_2) > 0:
            self.render.text(det_frame, "Score:    {}".format(result.score_2),
                             coordinate=(1370, 990))
        else:
            self.render.text(det_frame, "Score:    {}".format("Recognizing"),
                             coordinate=(1370, 990))

        if result.serving_player == "unknown":
            draw_text = "Recognizing..."
        elif result.serving_player == "name_1":
            draw_text = result.name_1.title()
        else:
            draw_text = result.name_2.title()

        self.render.text(det_frame, "Serving Player: {}".format(draw_text),
                         coordinate=(880, 870))

    def run(self):

        """
        Retrieves the frames from the session and passes it forward to do recognition.
        :return:
        """
        while not self.session.is_interrupted():
            self.session.update()
            det_frame = self.session.detection_frame
            if self.enable_threading:
                self.executor.submit(self.score_detector.detect, InputFrame(det_frame, self.session.frame_count, False))
            else:
                self.score_detector.detect(InputFrame(det_frame, self.session.frame_count, False))

            if self.scoreboard_result is not None:
                self.draw(det_frame, self.scoreboard_result)

            if self.app_profile["streamer"]["view_imshow"]:
                cv2.imshow("World", det_frame)
            if self.app_profile["streamer"]["save_stream"]:
                self.out.write(det_frame)

    def _warmup(self):
        for i in tqdm(range(self.detector_config["model"]["warm_up"]), desc="Warming up..."):
            blank_frame = np.zeros((self.score_detector.in_h, self.score_detector.in_w, 3), np.uint8)
            self.score_detector.detect(InputFrame(blank_frame, 0, True))
