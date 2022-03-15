import json
import os
from pathlib import Path

import onnxruntime as rt
import yaml
from easydict import EasyDict as edict

from src.controllers.ocr import crnn
from src.controllers.ocr.crnn import alphabets
from src.utils.csv_logger import CSV_Logger
from src.utils.renderer import Renderer

ROOT_DIR = Path(__file__).parents[1]
CONFIG_DIR = os.path.join(ROOT_DIR, "assets/configs")
CONFIGURATION_FILE = os.path.join(CONFIG_DIR, "app_config.yaml")
TEXT_REC_CONFIG_FILE = os.path.join(CONFIG_DIR, "text_rec_config.yaml")
DETECTOR_CONFIG_FILE = os.path.join(CONFIG_DIR, "detector_config.yaml")
PLAYERS_FILE_PATH = os.path.join(CONFIG_DIR, "data/gt/players.csv")
GT_FILE_PATH = os.path.join(ROOT_DIR, "assets/data/gt/groundtruth.json")


class AppContext(object):
    stream = open(CONFIGURATION_FILE, 'r')
    streamer_profile = yaml.load(stream, Loader=yaml.Loader)["streamer"]
    stream.close()
    csv_logger = CSV_Logger()
    total_frame_count = 0
    render = Renderer(streamer_profile["should_draw"])

    with open(TEXT_REC_CONFIG_FILE, 'r') as f:
        text_rec_config = yaml.load(f, Loader=yaml.FullLoader)
        text_rec_config = edict(text_rec_config)

    with open(DETECTOR_CONFIG_FILE, 'r') as f:
        detector_config = yaml.load(f, Loader=yaml.FullLoader)
        detector_config = edict(detector_config)

    text_rec_config.preprocessing.ALPHABETS = alphabets.alphabet
    text_rec_config.model.num_classes = len(text_rec_config.preprocessing.ALPHABETS)
    players_file_path = open('assets/data/gt/players.csv', 'r')
    playersLines = players_file_path.read().splitlines()
    sess_options = rt.SessionOptions()

    session = rt.InferenceSession(streamer_profile["score_det_model"],
                                  providers=["CUDAExecutionProvider"],
                                  sess_options=sess_options)

    model_batch_size = session.get_inputs()[0].shape[0]
    model_h = session.get_inputs()[0].shape[2]
    model_w = session.get_inputs()[0].shape[3]
    in_w = 640 if (model_w is None or isinstance(model_w, str)) else model_w
    in_h = 640 if (model_h is None or isinstance(model_h, str)) else model_h
    input_name = session.get_inputs()[0].name

    with open(GT_FILE_PATH, "r") as file:
        gt_ann = json.load(file)
    if streamer_profile["debug"]:
        print("Input Layer: ", session.get_inputs()[0].name)
        print("Output Layer: ", session.get_outputs()[0].name)
        print("Model Input Shape: ", session.get_inputs()[0].shape)
        print("Model Output Shape: ", session.get_outputs()[0].shape)
    print("Host Device: ", rt.get_device())
