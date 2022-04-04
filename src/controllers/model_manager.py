import math
from collections import OrderedDict, namedtuple

import numpy as np
import tensorrt as trt
import torch

from src import AppContext
from src.controllers.ocr.dl_txt_recognizer import DLTextRecognizer
from src.controllers.ocr.ocr_core import OCRCore
from src.controllers.ocr.tesseract_ocr import TesserTextRecognizer


class ModelManager(AppContext):
    def __init__(self):
        """
        Manages the models for detection and recognition
        """
        self.device = self.detector_config["model"]["execution_env"]
        self.init_detector()
        self.text_recognizer = self.load_text_recognizer()

    def init_detector(self):
        """
        Initialize the detector
        :return:
        """

        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(self.app_profile["models"]["score_det_model"], 'rb') as f, trt.Runtime(
                logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):
            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(
                self.device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()
        self.batch_size = self.bindings['images'].shape[0]

        self.imgsz = self.check_img_size(self.detector_config["model"]["img_size"],
                                         s=self.detector_config["model"]["stride"])

    def make_divisible(self, x, divisor):
        # Returns x evenly divisible by divisor
        return math.ceil(x / divisor) * divisor

    def check_img_size(self, imgsz, s=32, floor=0):
        # Verify image size is a multiple of stride s in each dimension
        if isinstance(imgsz, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(imgsz, int(s)), floor)
        else:  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in imgsz]
        if new_size != imgsz:
            print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
        return new_size

    def load_text_recognizer(self) -> OCRCore:
        """
        Load the recognizer model
        :return:
        """
        if self.app_profile["models"]["ocr_engine"] == "PyTesseract":
            print("Initializing OCR Backend : PyTesseract")
            text_recognizer = TesserTextRecognizer()
        else:
            print("Initializing OCR Backend : CRNN")
            text_recognizer = DLTextRecognizer()
        return text_recognizer
