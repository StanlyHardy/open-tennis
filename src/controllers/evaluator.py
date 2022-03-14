import json

from src import AppContext
from src.utils.daos import Result


class Evaluator(AppContext):
    def __init__(self):
        with open(self.GT_FILE_PATH, "r") as file:
            self.gt_ann = json.load(file)

    def trigger(self, data : Result):
        pass
