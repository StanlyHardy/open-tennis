import json

from src.controllers.evaluator import Evaluator
from src.utils.daos import Result


class CSV_Logger():
    def __init__(self):
        self.evaluator = Evaluator()
        with open('assets/data/result/rec_result.txt', 'w') as fd:
            fd.write(f'')
        self.json_result = {}

    def store(self, data: Result):
        bbox = data.score_board.bbox.tolist()
        frame_count = data.score_board.frame_count
        result = str(frame_count) + "," + \
                 data.name_1.lower().strip() + "," + data.name_2.lower().strip()  + "," + \
                 data.serving_player.lower().strip()  + "," + \
                 data.score_1.lower().strip()  + "," + data.score_2.lower().strip()

        with open('assets/data/result/rec_result.txt', 'a') as fd:
            fd.write(f'{result}\n')
        data_dict = data.__dict__
        data_dict.pop("score_board")
        data_dict["bbox"] = bbox
        self.json_result[str(frame_count)] = data_dict

    def persist(self, gt_annotation, total_frame_count = 0):
        with open('assets/data/result/rec_result.json', 'w') as outfile:
            json.dump(self.json_result, outfile)
        self.evaluator.evaluate( self.json_result,gt_annotation, total_frame_count)
