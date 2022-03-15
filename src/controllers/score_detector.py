import cv2
import numpy as np
import torch
from PIL import Image

from src import AppContext
from src.controllers.dl_txt_recognizer import DLTextRecognizer
from src.controllers.tesseract_ocr import TesserTextRecognizer
from src.utils import background_job
from src.utils.daos import InputFrame, ScoreBoard
from src.utils.detector_utils import letterbox_image, non_max_suppression, scale_coords


class ScoreDetector(AppContext):

    def __init__(self):
        self.text_recognizer = TesserTextRecognizer()

    def preprocess_image(self, pil_image, in_size=(640, 640)):
        """preprocesses PIL image and returns a norm np.ndarray
        pil_image = pillow image
        in_size: in_width, in_height
        """
        in_w, in_h = in_size
        resized = letterbox_image(pil_image, (in_w, in_h))
        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        img_in /= 255.0
        img_in = np.expand_dims(img_in, axis=0)
        return img_in

    def post_processing(self, detections, image_src, threshold, frame_count):
        boxs = detections[..., :4].numpy()
        confs = detections[..., 4].numpy()
        if isinstance(image_src, str):
            image_src = cv2.imread(image_src)
        elif isinstance(image_src, np.ndarray):
            image_src = image_src

        h, w = image_src.shape[:2]
        boxs[:, :] = scale_coords((self.in_h, self.in_w), boxs[:, :], (h, w)).round()
        tl = round(0.002 * (w + h) / 2) + 1
        for i, box in enumerate(boxs):
            if confs[i] >= threshold:
                x1, y1, x2, y2 = map(int, box)
                cropped_image = image_src[y1:y2, x1:x2]
                self.text_recognizer.run(ScoreBoard(cropped_image,frame_count,box, image_src))

                cv2.rectangle(image_src, (x1, y1), (x2, y2), (0, 255, 0), thickness=max(
                    int((w + h) / 600), 1), lineType=cv2.LINE_AA)

                cv2.putText(image_src, "scoreboard", (x1 + 3, y1 - 4), 0, tl / 3, [255, 255, 255],
                            thickness=1, lineType=cv2.LINE_AA)
    def detect(self, data: InputFrame):
        pil_img = Image.fromarray(
            cv2.cvtColor(data.image, cv2.COLOR_BGR2RGB))
        norm_image = self.preprocess_image(pil_img)
        outputs = self.session.run(None, {self.input_name: norm_image})

        batch_detections = torch.from_numpy(np.array(outputs[0]))

        batch_detections = non_max_suppression(
            batch_detections, conf_thres=0.4, iou_thres=0.5, agnostic=False)
        self.result = batch_detections[0]

        self.post_processing(batch_detections[0], data.image, 0.6, data.frame_count)
