import json

from src.utils.daos import Result


class CSV_Logger():
    def __init__(self):
        with open('assets/data/result/rec_result.txt', 'w') as fd:
            fd.write(f'')
        self.json_result = {}

    def store(self, data: Result):
        bbox = data.score_board.bbox.tolist()
        frame_count = data.score_board.frame_count
        result = str(frame_count) + "," + \
                 data.name_1 + "," + data.name_2 + "," + \
                 data.serving_player + "," + \
                 data.score_1 + "," + data.score_2

        with open('assets/data/result/rec_result.txt', 'a') as fd:
            fd.write(f'{result}\n')
        data_dict = data.__dict__
        data_dict.pop("score_board")
        data_dict["bbox"] = bbox
        self.json_result[str(frame_count)] = data_dict



    def persist(self):
        with open('assets/data/result/rec_result.json', 'w') as outfile:
            json.dump(self.json_result, outfile)
