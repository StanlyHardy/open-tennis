import numpy as np
from dataclasses import dataclass


@dataclass
class InputFrame(object):
    image: np.ndarray
    frame_count: int


@dataclass
class ScoreBoard:
    image: np.ndarray
    frame_count: int
    bbox: list


@dataclass
class Result:
    scoreboard: ScoreBoard
    name_1: str
    name_2: str
    serving_player: str
    score : str
