from dataclasses import dataclass

import numpy as np


@dataclass
class InputFrame(object):
    """
    Input frame that has been retrieved during each session
    """
    image: np.ndarray
    frame_count: int
    is_warm: bool


@dataclass
class ScoreBoard:
    """
    It holds the meta data of the scoreboard such as the image,
    bounding box coordinate and the original image in which it was extracted
    """
    image: np.ndarray
    frame_count: int
    bbox: np.ndarray
    raw_img: np.ndarray


@dataclass
class CourtYard:
    bboxes: dict
    centroids: dict


@dataclass
class Result:
    """
    Final result that has got the player data
    """
    score_board: ScoreBoard
    name_1: str
    name_2: str
    serving_player: str
    score_1: str
    score_2: str
