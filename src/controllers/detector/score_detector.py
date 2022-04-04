import math
import operator
from collections import OrderedDict
from functools import reduce

import cv2
import numpy as np
import torch

from src.controllers.detector.detector_utils import letterbox, non_max_suppression, scale_coords
from src.controllers.model_manager import ModelManager
from src.utils.daos import InputFrame, ScoreBoard
from src.utils.math_utils import MathUtils


class ScoreDetector(ModelManager):
    """
    Detect the location of the scoreboard
    """

    def __init__(self):
        super().__init__()
        self.all_labels = self.detector_config["model"]["class_labels"]

        self.reference_keys = [str(i + 1) for i in range(6)]

        self.reference_pts = MathUtils.group_pts(self.app_profile.thresholds.ref_points, 2)
        self.src = MathUtils.group_pts(self.app_profile.thresholds.src_points, 2)

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

    def _regulate_coordinates(self, centroids):
        """
        Remove the missing indices from the src points
        @param centroids: detected centroids of the centers of the court
        @return: current source point and the missing point indices list
        """
        present_keys = list(centroids.keys())

        missing_points = set(self.reference_keys).difference(set(present_keys))
        missing_points = [int(i) for i in missing_points]
        current_src = self.src
        if 0 < len(missing_points) <= 2:
            current_src = []
            for src_key, src_pt in enumerate(self.src):
                if int(src_key) + 1 not in missing_points:
                    current_src.append(src_pt)
        return [current_src, missing_points]

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
        court_yard = {}
        court_yard_centroids = {}
        if len(boxes) == 0:
            self.notif_center.post_notification(sender=self.__class__.__name__,
                                                with_name="ScoreManager", with_info=None)
        for box, score, c in zip(boxes, scores, classes):
            top_left, bottom_right = box[:2].astype(np.int64).tolist(), box[2:4].astype(np.int64).tolist()
            current_label = self.all_labels[int(c)]
            x1, y1, x2, y2 = top_left[0], top_left[1], bottom_right[0], bottom_right[1]
            if current_label == "scoreboard":
                if score > 0.90:
                    cropped_image = original_img[y1:y2, x1:x2]
                    scoreboard = ScoreBoard(cropped_image, frame_count, box, original_img)
                    self.text_recognizer.recognize(scoreboard)
            elif current_label == "central":
                # TODO Run keypoint search algorithm over the central patch for added verification.
                pass
            elif current_label == "ball":
                # TODO Trigger ball tracking.
                pass
            else:
                court_yard[current_label] = [(x1, y1), (x2, y2)]
                court_yard_centroids[current_label] = [(x1 + x2) // 2, (y1 + y2) // 2]
        sorted_centroids = OrderedDict(sorted(court_yard_centroids.items()))
        current_src, missing_points = self._regulate_coordinates(court_yard_centroids)

        if len(missing_points) <= 2:
            s_cent_list = list(sorted_centroids.values())
            center = tuple(
                map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), s_cent_list), [len(s_cent_list)] * 2))
            s_cent_list = sorted(s_cent_list, key=lambda coord: (-180 - math.degrees(
                math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
            dst_pts = []
            for centroid in s_cent_list :
                dst_pts.append(centroid)
            current_tx, _ = cv2.findHomography(np.float32(current_src), np.float32(dst_pts))
            final_pts = []
            for pt in self.reference_pts:
                res = MathUtils.apply_tx(pt, current_tx)
                final_pts.append((int(res[0]), int(res[1])))
            self.notif_center.post_notification(sender=self.__class__.__name__,
                                                with_name="TxPoints", with_info=final_pts)
        else:
            self.notif_center.post_notification(sender=self.__class__.__name__,
                                                with_name="TxPoints", with_info=None)

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
