import json
import os
from pathlib import Path

import onnxruntime as rt
import yaml
from easydict import EasyDict as edict

from src.controllers.ocr import crnn
from src.controllers.ocr.crnn import alphabets
from src.utils.app_utils import AppUtils
from src.utils.resultcoordinator import ResultCoordinator
from src.utils.renderer import Renderer

ROOT_DIR = Path(__file__).parents[1]
CONFIG_DIR = os.path.join(ROOT_DIR, "assets/configs")
CONFIGURATION_FILE = os.path.join(CONFIG_DIR, "app_config.yaml")
TEXT_REC_CONFIG_FILE = os.path.join(CONFIG_DIR, "text_rec_config.yaml")
DETECTOR_CONFIG_FILE = os.path.join(CONFIG_DIR, "detector_config.yaml")
PLAYERS_FILE_PATH = os.path.join(CONFIG_DIR, "data/gt/players.csv")
GT_FILE_PATH = os.path.join(ROOT_DIR, "assets/data/gt/groundtruth.json")


class AppContext(object):
    # Load App Profile
    app_profile = AppUtils.load_config(CONFIGURATION_FILE)
    streamer_profile = app_profile["streamer"]
    detector_config = AppUtils.load_config(DETECTOR_CONFIG_FILE)
    text_rec_config = AppUtils.load_config(TEXT_REC_CONFIG_FILE)

    total_frame_count = 0

    render = Renderer(streamer_profile["should_draw"])
    csv_logger = ResultCoordinator()

    players_file_path = open('assets/data/gt/players.csv', 'r')
    playersLines = players_file_path.read().splitlines()

    with open(GT_FILE_PATH, "r") as file:
        gt_ann = json.load(file)
