import difflib

import cv2
import numpy as np

from src import AppContext
from src.utils.daos import ScoreBoard, Result


class OCRCore(AppContext):

    def __init__(self):
        """
        Core OCR class that handles significant common functionalities shared by CRNN
        as well as the TesserOCR based recognizer.
        """
        mapped_players = (map(lambda x: x.lower().strip(), self.playersLines))
        self.players = list(mapped_players)

    def sanitize(self, name: str) -> str:
        """
        Sanitize the predicted player name based on the closest possible match over the list of
        the existing player.
        :param name: name of the player
        :return: matched player name
        """
        stripped_name = name.lower().strip()
        matching_name = difflib.get_close_matches(stripped_name, self.players)
        if len(matching_name) > 0:
            return matching_name[0]
        return "Recognizing..."

    def _divide_image(self, image: np.ndarray) -> dict:
        """
        Divide the cropped scoreboard into two patches.
            1. Upper patch has the Player 1 details
            2. Lower patch has the Player 2 details
        :param image: cropped image of the score board
        :return: patches that are embedded in a dictionary.
        """
        buf_image = image.copy()
        if len(buf_image.shape) > 2:
            h, w, c = buf_image.shape
        else:
            h, w = buf_image.shape
        start_x, start_y = (1, 1)
        end_x, end_y = (w, h // 2)

        lower_startx, lower_start_y = (0, h // 2)
        lower_end_x, lower_end_y = (w, h)

        upper_part = buf_image[start_y:end_y + 6, start_x:end_x]

        lower_part = buf_image[lower_start_y:lower_end_y, lower_startx:lower_end_x]

        patches = {"upper_patch": upper_part, "lower_patch": lower_part}

        return patches

    def draw(self, score_board: ScoreBoard, result: Result):
        """

        :param score_board: Scoreboard object with its metadata
        :param result: Processed result
        """
        score_board.raw_img = \
            self.render.draw_canvas(score_board.raw_img.copy(), score_board.raw_img)

        score_board.raw_img = \
            self.render.draw_canvas(score_board.raw_img.copy(), score_board.raw_img)

        self.render.text(score_board.raw_img, "Player 1: {}".format(result.name_1.title()),
                         coordinate=(870, 940))
        if len(result.score_1) > 0:
            self.render.text(score_board.raw_img, "Score:    {}".format(result.score_1),
                             coordinate=(870, 980))
        else:
            self.render.text(score_board.raw_img, "Recognizing...",
                             coordinate=(870, 980))

        self.render.text(score_board.raw_img, "Player 2: {}".format(result.name_2.title()),
                         coordinate=(1370, 940))
        if len(result.score_2) > 0:
            self.render.text(score_board.raw_img, "Score:    {}".format(result.score_2),
                             coordinate=(1370, 990))
        else:
            self.render.text(score_board.raw_img, "Recognizing...",
                             coordinate=(1370, 990))

        if result.serving_player == "unknown":
            draw_text = "Recognizing..."
        elif result.serving_player == "name_1":
            draw_text = result.name_1.title()
        else:
            draw_text = result.name_2.title()

        self.render.text(score_board.raw_img, "Serving Player: {}".format(draw_text),
                         coordinate=(880, 870))

    def process_result(self, result: dict, score_board: ScoreBoard):
        """

        :param result: unchecked result
        :param score_board: Scoreboard data object
        """
        result["bbox"] = score_board.bbox.tolist()
        result["frame_count"] = score_board.frame_count
        if "serving_player" not in result.keys():
            result["serving_player"] = "unknown"
        result = Result(score_board=score_board,
                        name_1=result["name_1"],
                        name_2=result["name_2"],
                        serving_player=result["serving_player"],
                        score_1=result["score_1"],
                        score_2=result["score_2"])
        self.draw(score_board, result)
        if self.app_profile["streamer"]["evaluation"]:
            if str(score_board.frame_count) in self.gt_ann.keys():
                self.csv_logger.store(result)

    def recognize(self, score_board: ScoreBoard):
        """
        Implementation overriden by the Tesseract or CRNN based recognizer
        :return:
        """
        pass
