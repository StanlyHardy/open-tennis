import multiprocessing

from src import AppContext
import onnxruntime as rt

from src.controllers.ocr.dl_txt_recognizer import DLTextRecognizer
from src.controllers.ocr.ocr_core import OCRCore
from src.controllers.ocr.tesseract_ocr import TesserTextRecognizer


class ModelManager(AppContext):
    def __init__(self):
        """
        Manages the models for detection and recognition
        """
        self.detector_session = self.load_detector()
        self.text_recognizer = self.load_text_recognizer()

    def load_detector(self):
        """
        Load the detector Model
        :return:
        """
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.detector_config["model"]["execution_env"] == "cpu":
            sess_options.intra_op_num_threads = multiprocessing.cpu_count()
            sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL

        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Prioritize CUDA so it can pick CUDA if available.
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = rt.InferenceSession(self.app_profile["models"]["score_det_model"],
                                      providers=EP_list,
                                      sess_options=sess_options)

        return session

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
