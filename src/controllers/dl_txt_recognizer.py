
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from src import AppContext
from src.controllers.ocr import crnn
from src.utils import ocr_utils
from src.utils.daos import ScoreBoard


class DLTextRecognizer(AppContext):
    def __init__(self):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.text_rec_model = crnn.get_crnn(self.text_rec_config).to(self.device)

        checkpoint = torch.load(self.streamer_profile["text_rec_model"])
        if 'state_dict' in checkpoint.keys():
            self.text_rec_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.text_rec_model.load_state_dict(checkpoint)
        self.text_rec_model.eval()
        self.converter = ocr_utils.strLabelConverter(self.text_rec_config.DATASET.ALPHABETS)

    def divide_image(self, image):
        h, w = image.shape
        start_x, start_y = (1, 1)
        end_x, end_y = (w, h // 2)

        lower_startx, lower_start_y = (0, h // 2)
        lower_end_x, lower_end_y = (w, h)

        upper_part = image[start_y:end_y + 6, start_x:end_x]

        lower_part = image[lower_start_y:lower_end_y, lower_startx:lower_end_x]

        return [upper_part, lower_part]

    def recognition(self,  patches, score_board : ScoreBoard):

        for patch in patches:
            h, w = patch.shape

            img = cv2.resize(patch, (0, 0), fx=self.text_rec_config.MODEL.IMAGE_SIZE.H / h, fy=self.text_rec_config.MODEL.IMAGE_SIZE.H / h,
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape
            w_cur = int(img.shape[1] / (self.text_rec_config.MODEL.IMAGE_SIZE.OW / self.text_rec_config.MODEL.IMAGE_SIZE.W))
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
            print('results: {0}'.format(sim_pred))
            result = str(score_board.frame_count) + "^" +sim_pred
            with open('assets/recognition.txt', 'a') as fd:
                fd.write(f'{result}\n')

    def run(self, score_board: ScoreBoard):
        patches = self.divide_image(score_board.image)
        self.recognition(patches, score_board)
