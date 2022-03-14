import os
from pathlib import Path

import onnxruntime as rt
import torch
import yaml
from easydict import EasyDict as edict

from src.controllers.ocr import alphabets, crnn

ROOT_DIR = Path(__file__).parents[1]
ASSETS_DIR = os.path.join(ROOT_DIR, "assets/configs")
CONFIGURATION_FILE = os.path.join(ASSETS_DIR, "app_config.yaml")



class AppContext(object):
    GT_FILE_PATH = "assets/data/top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json"
    stream = open(CONFIGURATION_FILE, 'r')
    streamer_profile = yaml.load(stream, Loader=yaml.Loader)["streamer"]
    stream.close()

    with open("assets/configs/text_rec_config.yaml", 'r') as f:
        text_rec_config = yaml.load(f, Loader=yaml.FullLoader)
        text_rec_config = edict(text_rec_config)

    text_rec_config.DATASET.ALPHABETS = alphabets.alphabet
    text_rec_config.MODEL.NUM_CLASSES = len(text_rec_config.DATASET.ALPHABETS)

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

    if streamer_profile["debug"]:
        print("Input Layer: ", session.get_inputs()[0].name)
        print("Output Layer: ", session.get_outputs()[0].name)
        print("Model Input Shape: ", session.get_inputs()[0].shape)
        print("Model Output Shape: ", session.get_outputs()[0].shape)
    print("Host Device: ", rt.get_device())
