import json
import os
from pathlib import Path


from src.controllers.ocr import crnn
from src.controllers.ocr.crnn import alphabets
from src.utils.app_utils import AppUtils
from src.utils.result_coord import ResultCoordinator
from src.utils.renderer import Renderer

ROOT_DIR = Path(__file__).parents[1]
CONFIG_DIR = os.path.join(ROOT_DIR, "assets/configs")
CONFIGURATION_FILE = os.path.join(CONFIG_DIR, "app_config.yaml")
PLAYERS_FILE_PATH = os.path.join(ROOT_DIR, "assets/data/gt/players.csv")


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
    ground_truth_path = os.path.expanduser(app_profile["paths"]["groundtruth_path"])
    if not os.path.exists(ground_truth_path):
        print("Please verify the ground-truth path {}".format(ground_truth_path))
        exit()
    with open(os.path.expanduser(ground_truth_path), "r") as file:
        gt_ann = json.load(file)
