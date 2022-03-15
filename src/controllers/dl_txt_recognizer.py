import difflib

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from src import AppContext
from src.controllers.OCRRoot import OCRRoot
from src.controllers.evaluator import Evaluator
from src.controllers.ocr import crnn
from src.utils import ocr_utils
from src.utils.csv_logger import CSV_Logger
from src.utils.daos import ScoreBoard, Result


class DLTextRecognizer(OCRRoot):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.text_rec_model = crnn.get_crnn(self.text_rec_config).to(self.device)

        checkpoint = torch.load(self.streamer_profile["text_rec_model"])
        if 'state_dict' in checkpoint.keys():
            self.text_rec_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.text_rec_model.load_state_dict(checkpoint)
        self.text_rec_model.eval()
        self.converter = ocr_utils.strLabelConverter(self.text_rec_config.DATASET.ALPHABETS)

    def recognition(self, patches, score_board: ScoreBoard):

        result = {}
        for k, patch in patches.items():
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            h, w = patch.shape

            img = cv2.resize(patch, (0, 0), fx=self.text_rec_config.MODEL.IMAGE_SIZE.H / h,
                             fy=self.text_rec_config.MODEL.IMAGE_SIZE.H / h,
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape
            w_cur = int(
                img.shape[1] / (self.text_rec_config.MODEL.IMAGE_SIZE.OW / self.text_rec_config.MODEL.IMAGE_SIZE.W))
            img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
            img = np.reshape(img, (self.text_rec_config.MODEL.IMAGE_SIZE.H, w_cur, 1))

            img = img.astype(np.float32)
            img = (img / 255. - self.text_rec_config.DATASET.MEAN) / self.text_rec_config.DATASET.STD
            img = img.transpose([2, 0, 1])

            img = torch.from_numpy(img)

            img = img.to(self.device)
            img = img.view(1, *img.size())
            preds = self.text_rec_model(img)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            preds_size = Variable(torch.IntTensor([preds.size(0)]))
            sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
            # print('results: {0}'.format(sim_pred))
            name_score_partition = sim_pred.partition("_")
            name = name_score_partition[0]
            score = name_score_partition[2]
            if ">" in name:
                name = name[1:]
                if k == "upper_patch":
                    result["serving_player"] = "name_1"
                else:
                    result["serving_player"] = "name_2"

            if k == "upper_patch":
                result["name_1"] = self.sanitize(name)
                result["score_1"] = score
            else:
                result["name_2"] = self.sanitize(name)
                result["score_2"] = score
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
            cv2.putText(
                img=score_board.raw_img,
                text="name1: " + result.name_1 + " " + result.name_2,
                org=(200, 200),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=3.0,
                color=(125, 246, 55),
                thickness=3
            )
            cv2.putText(
                img=score_board.raw_img,
                text="serving: " + result.serving_player,
                org=(200, 350),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=3.0,
                color=(125, 246, 55),
                thickness=3
            )
            cv2.putText(
                img=score_board.raw_img,
                text="score_1: " + result.score_1 + " score_2 " + result.score_2,
                org=(200, 500),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=3.0,
                color=(125, 246, 55),
                thickness=3
            )
            cv2.putText(
                img=score_board.raw_img,
                text="frame_count " + score_board.frame_count,
                org=(200, 600),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=3.0,
                color=(125, 246, 55),
                thickness=3
            )
            cv2.imwrite("assets/result/" + str(score_board.frame_count) + ".jpg", score_board.raw_img)

            self.evaluator.trigger(result)
            self.csv_logger.store(result)

    def run(self, score_board: ScoreBoard):
        patches = self.divide_image(score_board.image.copy())
        self.recognition(patches, score_board)
