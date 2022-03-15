import json
import os
from pathlib import Path

import onnxruntime as rt
import yaml
from easydict import EasyDict as edict

from src.controllers.ocr import crnn
from src.controllers.ocr.crnn import alphabets
from src.utils.app_utils import AppUtils
from src.utils.result_coord import ResultCoordinator
from src.utils.renderer import Renderer

ROOT_DIR = Path(__file__).parents[1]
CONFIG_DIR = os.path.join(ROOT_DIR, "assets/configs")
CONFIGURATION_FILE = os.path.join(CONFIG_DIR, "app_config.yaml")
PLAYERS_FILE_PATH = os.path.join(ROOT_DIR, "assets/data/gt/players.csv")
GT_FILE_PATH = os.path.join(ROOT_DIR, "assets/data/gt/groundtruth.json")


class AppContext(object):
    # Load App Profile
    app_profile = AppUtils.load_config(CONFIGURATION_FILE)
    detector_config = AppUtils.load_config(app_profile["models"]["detector_config"])
    text_rec_config = AppUtils.load_config(app_profile["models"]["text_rec_config"])
    playersLines = AppUtils.load_players(PLAYERS_FILE_PATH)
    text_rec_config.preprocessing.alphabets = alphabets.alphabet
    text_rec_config.model.num_classes = len(text_rec_config.preprocessing.alphabets)
    render = Renderer(app_profile)
    csv_logger = ResultCoordinator()

    total_frame_count = 0
    with open(GT_FILE_PATH, "r") as file:
        gt_ann = json.load(file)
