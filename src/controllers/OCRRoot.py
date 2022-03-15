import difflib

import cv2

from src import AppContext
from src.controllers.evaluator import Evaluator


class OCRRoot(AppContext):

    def __init__(self):
        players_file_path = open('assets/data/gt/players.csv', 'r')
        self.playersLines = players_file_path.read().splitlines()
        mapped_players = (map(lambda x: x.lower().strip(), self.playersLines))
        self.players = list(mapped_players)
        self.evaluator = Evaluator()

    def sanitize(self, name):
        stripped_name = name.lower().strip()
        matching_name = difflib.get_close_matches(stripped_name, self.players)
        if len(matching_name) > 0:
            return matching_name[0]
        return name

    def divide_image(self, image):
        buf_image = image.copy()
        h, w,c = buf_image.shape
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