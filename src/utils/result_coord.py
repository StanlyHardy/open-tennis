import json

from src.controllers.evaluator import Evaluator
from src.utils.daos import Result


class ResultCoordinator(object):
    def __init__(self):
        """
        Coordinates and persists the downstream results.
        """
        self.evaluator = Evaluator()
        self.buff_repo = {}

    def store(self, data: Result):
        """
        Stack the data temporarily
        :param data:
        :return:
        """
        bbox = data.score_board.bbox.tolist()
        frame_count = data.score_board.frame_count

        data_dict = data.__dict__
        data_dict.pop("score_board")
        data_dict["bbox"] = bbox
        self.buff_repo[str(frame_count)] = data_dict

    def persist(self, path, gt_annotation: dict, total_frame_count=0):
        """
        Persist the data to a json file and evaluate further.
        :param gt_annotation: ground truth annotation
        :param total_frame_count:  total frame count
        :return:
        """
        if total_frame_count:
            with open(path, 'w') as outfile:
                json.dump(self.buff_repo, outfile)
            self.evaluator.evaluate(self.buff_repo, gt_annotation, total_frame_count)
