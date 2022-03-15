import re

import cv2

import tesserocr

from src.controllers.ocr.OCRRoot import OCRRoot
from src.utils.daos import ScoreBoard, Result
from PIL import Image


class TesserTextRecognizer(OCRRoot):
    def __init__(self):
        super().__init__()
        self.api = tesserocr.PyTessBaseAPI()

    def get_preprocessed_image(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        count = 0
        res = []
        for c in contours:
            sub_h, sub_w = thresh.shape
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            if h > 60 and h < 80 and w < 60 and x > (sub_w // 2):
                count += 1
                res.append([x, y, x + w, y + h])
        if count == 1:
            res = res[0]
            sub_crop = thresh[res[1]:res[3], res[0]:res[2]]
            sub_crop = ~sub_crop
            thresh[res[1]:res[3], res[0]:res[2]] = sub_crop
        patches = self.divide_image(thresh)
        return patches

    def recognition(self, patches, score_board: ScoreBoard):
        name = ["unknown" for i in range(2)]
        result = {}
        for pos, (k, patch) in enumerate(patches.items()):
            yo = Image.fromarray(patch)
            self.api.SetImage(yo)
            text = self.api.GetUTF8Text()
            score_match = re.search(r"\d", text)
            if score_match:
                score_match_pos = score_match.start()
            else:
                score_match_pos = -1
            name[pos] = text[:score_match_pos]

            pattern = re.compile("[A-Za-z0-9]+")

            if pattern.fullmatch(name[pos][:len(name[pos]) - 1]) is not None:
                if k == "upper_patch":
                    result["serving_player"] = "name_1"
                else:
                    result["serving_player"] = "name_2"

            reg_score = text[score_match_pos:]
            score = re.sub(r'\W+', '-', reg_score)
            score = score[:len(score) - 1]

            noisy_name = text[:score_match_pos]
            if k == "upper_patch":
                result["name_1"] = self.sanitize(noisy_name)
            else:
                result["name_2"] = self.sanitize(noisy_name)

            if k == "upper_patch":
                result["score_1"] = score.lower().strip()
            else:
                result["score_2"] = score.lower().strip()

        if str(score_board.frame_count) in self.gt_ann.keys():

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
            self.csv_logger.store(result)

    def run(self, score_board: ScoreBoard):
        patches = self.get_preprocessed_image(score_board.image)
        self.recognition(patches, score_board)
