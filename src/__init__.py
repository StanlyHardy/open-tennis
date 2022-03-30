import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from notificationcenter import *

from src.controllers.ocr import crnn
from src.controllers.ocr.crnn import alphabets
from src.utils.app_utils import AppUtils
from src.utils.renderer import Renderer
from src.utils.result_coord import ResultCoordinator

ROOT_DIR = Path(__file__).parents[1]
CONFIG_DIR = os.path.join(ROOT_DIR, "assets/configs")
CONFIGURATION_FILE = os.path.join(CONFIG_DIR, "app_config.yaml")


class AppContext(object):
    # Load App Profile
    app_profile = AppUtils.load_config(CONFIGURATION_FILE)
    detector_config = AppUtils.load_config(app_profile["models"]["detector_config"])
    text_rec_config = AppUtils.load_config(app_profile["models"]["text_rec_config"])
    playersLines = AppUtils.load_players(app_profile["paths"]["players_path"])

    if app_profile["streamer"]["evaluation"]:
        ground_truth_path = os.path.expanduser(app_profile["paths"]["groundtruth_path"])
        if not os.path.exists(ground_truth_path):
            print("Please verify the ground-truth path {}".format(ground_truth_path))
            exit()
        with open(os.path.expanduser(ground_truth_path), "r") as file:
            gt_ann = json.load(file)

    # Initialize CRNN classes
    text_rec_config.preprocessing.alphabets = alphabets.alphabet
    text_rec_config.model.num_classes = len(text_rec_config.preprocessing.alphabets)

    renderer = Renderer(app_profile)
    result_coordinator = ResultCoordinator()
    total_frame_count = 0
    tl = 0
    scoreboard_result = None
    notif_center = NotificationCenter()
    executor = ThreadPoolExecutor()

    if app_profile["models"]["ocr_engine"] != "PyTesseract":
        enable_threading = True
    else:
        enable_threading = False
