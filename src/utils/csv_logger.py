from src.utils.daos import Result


class CSV_Logger():
    def __init__(self):
        with open('assets/result/rec_result.txt', 'w') as fd:
            fd.write(f'')

    def store(self, result: Result):
        score_board = result.scoreboard
        result = str(
            score_board.frame_count) + "," + \
                 result.name_1 + "," + result.name_2 + "," + \
                 result.serving_player + ","+\
                 result.score_1 +"," + result.score_2

        with open('assets/data/result/rec_result.txt', 'a') as fd:
            fd.write(f'{result}\n')

