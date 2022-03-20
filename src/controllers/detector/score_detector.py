import cv2
import numpy as np
import torch

from src.controllers.detector.detector_utils import letterbox, non_max_suppression, scale_coords
from src.controllers.model_manager import ModelManager
from src.utils.daos import InputFrame, ScoreBoard


class ScoreDetector(ModelManager):
    """
    Detect the location of the scoreboard
    """

    def __init__(self):
        super().__init__()

    def normalize(self, img):
        """
        Normalize the frame before detection
        Args:
            img: resized image that needs to be normalized

        Returns:

        """
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        return img

    def infer(self, img):
        """
        Run inference to determine the coords of the scoreboard
        Args:
            img: image to detect the scoreboard

        Returns:

        """
        assert img.shape == self.bindings['images'].shape, (img.shape, self.bindings['images'].shape)
        self.binding_addrs['images'] = int(img.data_ptr())

        self.context.execute_v2(list(self.binding_addrs.values()))
        pred = self.bindings['output'].data
        pred = pred.cpu().numpy()
        return pred

    def post_process(self, pred, processed_image, original_img, frame_count):
        """
        Post process the detected result and pass it to the player information extractor
        Args:
            pred: predicted result
            processed_image: resized image
            original_img: original input frame
            frame_count: the id of the current frame

        """
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(processed_image.shape[2:], det[:, :4], original_img.shape).round()

        output = pred[0]
        boxes = output[:, :4]
        scores = output[:, 4]
        classes = output[:, 5]
        h, w = original_img.shape[:2]
        tl = round(0.002 * (w + h) / 2) + 1
        for box, score, c in zip(boxes, scores, classes):
            top_left, bottom_right = box[:2].astype(np.int64).tolist(), box[2:4].astype(np.int64).tolist()

            x1, y1, x2, y2 = top_left[0], top_left[1], bottom_right[0], bottom_right[1]
            cropped_image = original_img[y1:y2, x1:x2]
            scoreboard = ScoreBoard(cropped_image, frame_count, box, original_img)
            self.text_recognizer.recognize(scoreboard)

            self.render.rect(
                original_img, (x1, y1), (x2, y2),
                thickness=max(
                    int((w + h) / 600), 1)
            )
            self.render.text(original_img, "scoreboard", (x1 + 3, y1 - 4), 0, tl / 3)

    def detect(self, data: InputFrame):
        """
        Detect the scoreboard
        Args:
            data: data that has got the input frame

        Returns:

        """
        processed_image = letterbox(data.image, new_shape=self.imgsz, auto=False)[0]
        processed_image = self.normalize(processed_image)
        pred = self.infer(processed_image)
        pred = non_max_suppression(pred, self.detector_config["model"]["conf_thresh"],
                                   self.detector_config["model"]["iou_thresh"],
                                   classes=None,
                                   agnostic=False)

        self.post_process(pred, processed_image, data.image, data.frame_count)
