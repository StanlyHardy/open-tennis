import re

import cv2

import tesserocr

from src.controllers.OCRRoot import OCRRoot
from src.utils.daos import ScoreBoard, Result
from PIL import Image


class TesserTextRecognizer(OCRRoot):
    def __init__(self):
        super().__init__()
        self.api = tesserocr.PyTessBaseAPI()

    def recognition(self, patches, score_board: ScoreBoard):
        pass

    def run(self, score_board: ScoreBoard):
        patches = self.divide_image(score_board.image.copy())
        self.recognition(patches, score_board)
