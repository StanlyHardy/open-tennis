import cv2
import numpy as np
import onnxruntime as rt
import torch
from PIL import Image

from src.controllers.model_manager import ModelManager
from src.controllers.detector.detector_utils import letterbox_image, non_max_suppression, scale_coords
from src.utils.daos import InputFrame, ScoreBoard


class ScoreDetector(ModelManager):
    """
    Detect the location of the scoreboard
    """

    def __init__(self):
        super().__init__()
        self.model_batch_size = self.detector_session.get_inputs()[0].shape[0]
        model_h = self.detector_session.get_inputs()[0].shape[2]
        model_w = self.detector_session.get_inputs()[0].shape[3]
        self.in_w = 640 if (model_w is None or isinstance(model_w, str)) else model_w
        self.in_h = 640 if (model_h is None or isinstance(model_h, str)) else model_h
        self.input_name = self.detector_session.get_inputs()[0].name
        self._model_check()

    def _model_check(self):
        if self.app_profile["streamer"]["debug"]:
            print("Input Layer: ", self.input_name)
            print("Output Layer: ", self.detector_session.get_outputs()[0].name)
            print("Model Input Shape: ", (self.in_w,self.in_h))
            print("Model Output Shape: ", self.detector_session.get_outputs()[0].shape)
        print("Host Device: ", rt.get_device())

    def preprocess_image(self, pil_image) -> np.ndarray:
        """
        Preprocess the input frame
        :param pil_image: image on which the detection has to be made
        :param in_size: size of the input
        :return: Preprocessed imagedata
        """
        resized = letterbox_image(pil_image, (self.in_w, self.in_h))
        img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)
        img_in /= 255.0
        img_in = np.expand_dims(img_in, axis=0)
        return img_in

    def post_processing(self, detections, image_src, threshold, frame_count):
        """
        Postprocess the detection and pass it to the Player recognizers :func:`~src.ocr.ocr_core.recognize
        :param detections:
        :param image_src:
        :param threshold:
        :param frame_count:
        :return:
        """
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
                self.text_recognizer.recognize(ScoreBoard(cropped_image, frame_count, box, image_src))

                self.render.rect(
                    image_src, (x1, y1), (x2, y2),
                    thickness=max(
                        int((w + h) / 600), 1)
                )
                self.render.text(image_src, "scoreboard", (x1 + 3, y1 - 4), 0, tl / 3)

    def detect(self, data: InputFrame):
        pil_img = Image.fromarray(
            cv2.cvtColor(data.image, cv2.COLOR_BGR2RGB))
        norm_image = self.preprocess_image(pil_img)
        outputs = self.detector_session.run(None, {self.input_name: norm_image})

        batch_detections = torch.from_numpy(np.array(outputs[0]))

        batch_detections = non_max_suppression(
            batch_detections, conf_thres=self.detector_config["model"]["conf_thresh"],
            iou_thres=self.detector_config["model"]["iou_thres"], agnostic=False)
        self.result = batch_detections[0]

        self.post_processing(batch_detections[0], data.image, self.detector_config["model"]["conf_thresh"],
                             data.frame_count)


