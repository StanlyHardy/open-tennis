import re

import cv2

import tesserocr

from src.controllers.ocr.ocr_core import OCRCore
from src.utils.daos import ScoreBoard
from PIL import Image


class TesserTextRecognizer(OCRCore):
    def __init__(self):
        """
        OCR Recognition based on TesserOCR.
        """
        super().__init__()
        self.api = tesserocr.PyTessBaseAPI()
        self.symbol_pattern = re.compile("[A-Za-z0-9]+")

    def get_preprocessed_image(self, patch):
        """
        Preprocess the input patch
        :param patch:
        :return: cropped preprocessed patches
        """
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # binarize the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # find the position of the rectangular block in the score so it can be inverted specifically.
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        count = 0
        res = []
        for c in contours:
            sub_h, sub_w = thresh.shape
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            # Usually the rectangular block tends to be in the right half of the image
            if 60 < h < 80 and w < 60 and x > (sub_w // 2):
                count += 1
                res.append([x, y, x + w, y + h])
        if count == 1:
            res = res[0]
            sub_crop = thresh[res[1]:res[3], res[0]:res[2]]
            # inverted the cropped block so the bg is white and number is black.
            sub_crop = ~sub_crop
            # paste the image back to the original cropped image.
            thresh[res[1]:res[3], res[0]:res[2]] = sub_crop
        patches = self.divide_image(thresh)
        return patches

    def analyze(self, patches, score_board: ScoreBoard):
        """
        Recognize the text in the cropped score image
        :param patches: patches that were cut previously
        :param score_board: scoreboard with it's metadata
        :return:
        """
        name = ["unknown" for i in range(2)]
        result = {}
        for pos, (patch_position, patch) in enumerate(patches.items()):
            pil_image = Image.fromarray(patch)
            self.api.SetImage(pil_image)
            text = self.api.GetUTF8Text()

            # grab the score from the noisy text
            score_match = re.search(r"\d", text)
            if score_match:
                score_match_pos = score_match.start()
            else:
                score_match_pos = -1

            # extract the noisy name based on the position in which the score begins.
            name[pos] = text[:score_match_pos]
            # determine the serving player for they tend to have a symbol in the beginning
            if self.symbol_pattern.fullmatch(name[pos][:len(name[pos]) - 1]) is not None:
                if patch_position == "upper_patch":
                    result["serving_player"] = "name_1"
                else:
                    result["serving_player"] = "name_2"
            # the score tends to have symbols sometimes. Clean such scores.
            reg_score = text[score_match_pos:]
            score = re.sub(r'\W+', '-', reg_score)
            score = score[:len(score) - 1]

            # pick the closest possible name from the stored player data
            noisy_name = text[:score_match_pos]
            if patch_position == "upper_patch":
                result["name_1"] = self.sanitize(noisy_name)
            else:
                result["name_2"] = self.sanitize(noisy_name)

            # determine the score
            if patch_position == "upper_patch":
                result["score_1"] = score.lower().strip()
            else:
                result["score_2"] = score.lower().strip()
        self.process_result(result, score_board)

    def recognize(self, score_board: ScoreBoard):
        """
        :param score_board: scoreboard that contains it's meta data
        """
        patches = self.get_preprocessed_image(score_board.image)
        self.analyze(patches, score_board)
