import cv2
import numpy as np
import torch
from torch.autograd import Variable

from src.controllers.ocr.ocr_core import OCRCore
from src.controllers.ocr.crnn import crnn
from src.controllers.ocr.crnn import ocr_utils
from src.utils.daos import ScoreBoard, Result


class DLTextRecognizer(OCRCore):
    """
    CRNN based scoreboard text recognizer.
    """
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
        self.converter = ocr_utils.strLabelConverter(self.text_rec_config.preprocessing.ALPHABETS)

    def _preprocess(self, patch):
        """
        Preprocess the input image patch.
        :param patch: patch that needs to be preprocessed.
        :return:
        """
        h, w = patch.shape
        img = cv2.resize(patch, (0, 0), fx=self.text_rec_config.model.img_size.h / h,
                         fy=self.text_rec_config.model.img_size.h / h,
                         interpolation=cv2.INTER_CUBIC)
        h, w = img.shape
        w_cur = int(
            img.shape[1] / (self.text_rec_config.model.img_size.ow / self.text_rec_config.model.img_size.w))
        img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (self.text_rec_config.model.img_size.h, w_cur, 1))

        img = img.astype(np.float32)
        img = (img / 255. - self.text_rec_config.preprocessing.mean) / self.text_rec_config.preprocessing.std
        img = img.transpose([2, 0, 1])

        img = torch.from_numpy(img)

        img = img.to(self.device)
        img = img.view(1, *img.size())
        return img

    def _analyze(self, patches, score_board: ScoreBoard):
        """
        Analyze the input patch for the occurrence of the scoreboard
        :param patches: patches that were cut previously
        :param score_board: scoreboard data object
        """
        result = {}
        for k, patch in patches.items():

            img = self._preprocess(patch)
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
                result["score_1"] = score.lower().strip()
            else:
                result["name_2"] = self.sanitize(name)
                result["score_2"] = score.lower().strip()

        self.process_result(result, score_board)

    def recognize(self, score_board: ScoreBoard):
        """
        Recognize the player data in the input scoreboard
        :param score_board: Scoreboard that has got the player data.
        """
        score_board_img = cv2.cvtColor(score_board.image.copy(), cv2.COLOR_BGR2GRAY)
        patches = self._divide_image(score_board_img)
        self._analyze(patches, score_board)
