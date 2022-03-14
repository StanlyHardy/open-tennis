from src import AppContext
from src.utils.daos import Result


class CSV_Logger(AppContext):
    def __init__(self):
        with open('assets/recognition.txt', 'w') as fd:
            fd.write(f'')

    def store(self, result: Result):
        score_board = result.scoreboard
        result = str(
            score_board.frame_count) + "," + result.name_1 + "," + result.name_2 + "," + result.serving_player + ","+ result.score

        with open('assets/rec_result.txt', 'a') as fd:
            fd.write(f'{result}\n')