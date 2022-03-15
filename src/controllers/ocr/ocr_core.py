import difflib

import cv2

from src import AppContext
from src.utils.daos import ScoreBoard, Result


class OCRCore(AppContext):

    def __init__(self):
        mapped_players = (map(lambda x: x.lower().strip(), self.playersLines))
        self.players = list(mapped_players)

    def sanitize(self, name):
        stripped_name = name.lower().strip()
        matching_name = difflib.get_close_matches(stripped_name, self.players)
        if len(matching_name) > 0:
            return matching_name[0]
        return name

    def divide_image(self, image):
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

    def enlarge_scoreboard_images(self, patch, enlarge_ratio):
        patch = cv2.resize(
            patch, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)
        return patch

    def draw(self, score_board: ScoreBoard, result: Result):
        score_board.raw_img = \
            self.render.draw_boundary(score_board.raw_img.copy(), score_board.raw_img)

        score_board.raw_img = \
            self.render.draw_boundary(score_board.raw_img.copy(), score_board.raw_img)

        self.render.text(score_board.raw_img, "Player 1: {}".format(result.name_1.title()),
                         coordinate=(870, 940))
        self.render.text(score_board.raw_img, "Score:    {}".format(result.score_1),
                         coordinate=(870, 980))

        self.render.text(score_board.raw_img, "Player 2: {}".format(result.name_2.title()),
                         coordinate=(1370, 940))
        self.render.text(score_board.raw_img, "Score:    {}".format(result.score_2),
                         coordinate=(1370, 990))
        if result.serving_player == "unknown":
            draw_text = "Recognizing..."
        elif result.serving_player == "name_1":
            draw_text = result.name_1
        else:
            draw_text = result.name_2

        self.render.text(score_board.raw_img, "Serving Player: {}".format(draw_text),
                         coordinate=(880, 870))